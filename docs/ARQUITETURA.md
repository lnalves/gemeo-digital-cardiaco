# Arquitetura de implementação

Tradução da arquitetura do Cap. 4 do TC (Seções 4.1–4.4) em módulos de código, contratos e fluxos.
Parâmetros citados entre crases (`assim`) estão em `configs/default.yaml`.

## 1. Visão geral

```
                   ┌──────────────────────── servidor Python (FastAPI) ────────────────────────┐
 MIT-BIH (.dat)    │                                                                            │
      │            │   engine.StreamingPipeline  (um por sessão / registro)                     │
      ▼            │   ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐     │
 acquisition/  ──WS──▶ │ signal/    │─▶│ model/       │─▶│ decision/   │─▶│ storage/     │     │
 simulator.py  /ingest │ filtro     │  │ BeatClassif. │  │ regras Q5   │  │ SQLite       │     │
 (360 Hz, 1 s) │   │   │ picos R    │  │ (atraso 1 RR)│  │             │  │              │     │
               │   │   │ batimentos │  └──────────────┘  └──────▲──────┘  └──────────────┘     │
               │   │   │ features   │─▶ profile/ (perfil Q, KL, L, EMA) ─────┘                  │
               │   │   └────────────┘                                                          │
               │   └──────────────────────────────── WS /monitor + REST ────────────────────────┘
               │                                              │
               │                                              ▼
               │                                  dashboard/ (React) — especialista
               │                                  confirma/descarta → POST /alerts/{id}/decision
               └──────────── avaliação offline: evaluation/replay.py usa o MESMO StreamingPipeline
```

Correspondência com Grieves (2014): entidade física = paciente (registro); representação virtual =
`BeatClassifier` + `PatientProfile`; conexões = `/ingest` (físico→virtual) e alertas/decisões (virtual→físico).

## 2. Estrutura de diretórios

```
src/cardiac_twin/
  config.py            # carrega YAML → dataclasses imutáveis (Config)
  provenance.py        # hash do git (com -dirty) gravado nos artefatos
  data/
    mitbih.py          # download (wfdb.dl_database), leitura de registro (canal MLII), anotações
    labels.py          # mapa símbolo MIT-BIH → classe AAMI; símbolos de não-batimento ignorados
    splits.py          # DS1, DS2, PACED; split DS1 → DS1-train / DS1-val por REGISTRO
    builder.py         # registro → batimentos + features (mesmas funções de signal/ usadas no streaming)
  signal/
    filters.py         # CausalBandpass (sos + estado zi), streaming e em lote com saída idêntica
    rpeaks.py          # StreamingRPeakDetector (Pan-Tompkins simplificado) + matching ±150 ms
    segmentation.py    # recorte de batimento [R-pre, R+post], normalização, contexto de K batimentos
    features.py        # RR pré/pós/local, largura QRS (onset/offset por derivada), amplitude R
  model/
    architecture.py    # build_model(cfg) → keras.Model (saídas: probs, attention)
    losses.py          # focal loss categórica; pesos de classe
    dataset.py         # npz → tf.data (sequências de K batimentos + features RR)
    train.py           # laço de treino, early stopping, salvamento em artifacts/runs/
    inference.py       # BeatClassifier: carrega .keras, predict_batch → (classe, prob, attention)
  profile/
    histogram.py       # bins fixos por feature (percentis da calibração + bins de transbordo)
    kl.py              # kl_divergence(p, q, eps) e soma das marginais
    profile.py         # PatientProfile: build(calib) · threshold() · score(window) · update(P, alpha)
  decision/
    rules.py           # decide_beat(...), decide_window(...), VT detector (≥3 V consecutivos)
  engine/
    pipeline.py        # StreamingPipeline: estado da sessão; push(chunk) → list[Event]
    events.py          # dataclasses: BeatEvent, KLUpdate, DecisionEvent, Alert, StateChange
  storage/
    db.py              # SQLite (sqlite3); tabelas abaixo
  acquisition/
    simulator.py       # cliente WS: lê registro, envia blocos de `chunk_seconds` em tempo real (ou acelerado)
  server/
    app.py             # FastAPI: REST + WebSockets; uma StreamingPipeline por sessão
    schemas.py         # modelos Pydantic das mensagens
  evaluation/
    metrics.py         # métricas por classe/macro, AUC OvR, matriz de confusão
    replay.py          # roda StreamingPipeline sobre registros sem dormir; mede latência de processamento
    compare.py         # config A (classificador isolado) × B (classificador + perfil); Wilcoxon pareado
scripts/               # CLIs finas que só chamam funções do pacote
dashboard/             # React + Vite + TS
tests/
```

## 3. Dados e pré-processamento (offline e online)

- **Fonte:** MIT-BIH Arrhythmia Database (PhysioNet `mitdb`), 48 registros de 30 min, 360 Hz, 2 canais.
  Usa-se apenas o canal **MLII** (índice obtido por nome do sinal, não por posição).
- **Exclusões:** registros com marca-passo `102, 104, 107, 217`.
- **DS1 (treino):** 101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230
- **DS2 (teste):** 100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234
- **DS1-val:** subconjunto de registros de DS1 (`data.val_records` = 101, 124, 205, 207), sorteado por
  `scripts/select_val_records.py` entre as combinações com S, V e F suficientes sem tirar a maior parte de nenhuma
  classe do treino (D-15); usado para early stopping e para ajustar `k` e janela do perfil.
- **Mapa AAMI (EC57):** N = {N, L, R, e, j}; S = {A, a, J, S}; V = {V, E}; F = {F}; Q = {/, f, Q}.
- **Filtro:** Butterworth passa-faixa `signal.bandpass_hz` (0,5–40 Hz), ordem `signal.filter_order`, forma SOS,
  aplicado causalmente com estado. Teste obrigatório: filtrar em blocos de 360 amostras ≡ filtrar o sinal inteiro.
- **Picos R:** treino usa as posições anotadas. Execução usa `StreamingRPeakDetector` (configurável:
  `engine.rpeak_source: detector | annotation`). Na avaliação, batimentos detectados são casados com anotações
  em ±150 ms (EC57); não casados contam como detecções espúrias, anotações sem par como perdidas.
  Nos dois caminhos o R é **refinado** para o máximo de |x| filtrado em ±`segment.r_refine_ms` (D-16): sem isso,
  o atraso de fase do filtro causal deslocaria as janelas do treino (anotação) em relação às da execução (detector).
  Parâmetros do detector em `rpeaks.*`.
- **Batimento:** janela `[R - pre, R + post]` (`segment.pre_samples`=90, `post_samples`=166 → 256 amostras),
  normalização z-score por batimento.
- **Features por batimento:**
  - RR pré, RR pós, RR local médio (últimos `features.local_rr_beats`), razões pré/local e pós/local (entrada do modelo).
  - Largura do QRS: onset/offset onde |derivada| do sinal filtrado cai abaixo de `features.qrs_slope_frac` × máx,
    busca em ±`features.qrs_search_ms`. Validar faixa plausível (60–200 ms) em amostra de registros.
  - Amplitude R: valor do sinal filtrado no pico menos a mediana da linha de base local.
  - O **perfil** usa: RR (pré), largura QRS, amplitude R.

## 4. Classificador (`model/`)

Entrada para classificar o batimento *i*: sequência de K = `model.context_beats` (default 5) batimentos
`[i-3, i-2, i-1, i, i+1]` — 1 batimento futuro, o que gera o atraso de ~1 RR descrito na Seção 4.4.

```
beats  (K, 256, 1) ─▶ TimeDistributed(CNN encoder) ─▶ (K, 128)
                       Conv1D(32,7)-BN-ReLU-MaxPool
                       Conv1D(64,5)-BN-ReLU-MaxPool
                       Conv1D(128,3)-BN-ReLU-GlobalAvgPool
rr     (K, 5)  ──────────────────────────────────────▶ concat ─▶ (K, 133)
                                                       ─▶ BiLSTM(64, return_sequences) ─▶ (K, 128)
                                                       ─▶ atenção aditiva (Bahdanau) sobre K ─▶ contexto (128), α (K)
                                                       ─▶ Dense(64)-Dropout(0.3)-Dense(5, softmax)
saídas: probs (5), attention (K)
```

- Os pesos α indicam quais batimentos da sequência mais pesaram → no dashboard, destacam os trechos do sinal
  (RNF03). Se houver tempo, Grad-CAM 1D sobre o encoder para regiões intra-batimento (opcional).
- Desbalanceamento: comparar `weighted_ce` (pesos inversos à frequência, com teto) e `focal` (γ=2). Registrar ambos.
- Otimizador Adam (lr `model.lr`), batch `model.batch_size`, early stopping por macro-F1 em DS1-val
  (`model.patience`). Seeds fixas.
- Modelo salvo como `.keras`; `BeatClassifier` é a única porta de entrada usada por `engine/`.

## 5. Perfil do paciente (`profile/`)

- **Calibração:** primeiros `profile.calibration_seconds` (300 s) do registro → valores das 3 features.
- **Histogramas:** para cada feature f, bins fixos definidos na calibração: `profile.n_bins` bins entre os percentis
  1 e 99 **mais dois bins de transbordo** (< p1 e > p99). Sem eles, valores fora da faixa da calibração —
  justamente os anômalos — seriam descartados e a KL não os perceberia.
- **Suavização:** soma `profile.eps` a todos os bins antes de normalizar (Seção 2.3.4).
- **Divergência:** `D = Σ_f D_KL(P_f ‖ Q_f)` (soma das marginais). Histograma conjunto 3D é inviável com 60–100
  batimentos por janela.
- **Limiar:** subjanelas da calibração com o mesmo tamanho e passo da monitorização (`profile.window_seconds`,
  `profile.step_seconds`) → valores de D → `L = μ + k·σ` (Eq. 11), `k = profile.k` ajustado em DS1-val.
- **Atualização (Eq. 10):** `Q_f ← (1-α)·Q_f + α·P_f` só para alertas descartados; `α = profile.alpha`.
  Desligável por `engine.feedback_enabled`. Versão do perfil incrementa a cada atualização (persistida).

## 6. Decisão (`decision/`)

Implementa o Quadro 5. Para tornar o falso positivo mensurável **por batimento**, a decisão é feita em dois pontos:

| Gatilho | Regra | Saída |
|---|---|---|
| Batimento *i* classificado | `cls == V` e é o 3º V consecutivo | **alerta crítico** (imediato, ignora perfil) |
| Batimento *i* classificado (`cls ≠ N`) | KL atual (última janela) > L | **alerta** |
| Batimento *i* classificado (`cls ≠ N`) | KL atual ≤ L | **atenção** |
| Atualização de janela (a cada 10 s) | sem batimentos ≠ N na janela e KL > L | **atenção** |
| Atualização de janela | todos N e KL ≤ L | normal (só atualiza painel) |

- "KL atual" = valor da janela de 60 s mais recente. Cada batimento é decidido **uma única vez**, evitando dupla
  contagem por janelas sobrepostas.
- Durante a calibração: só alerta crítico (perfil inexistente).
- **Configuração A (classificador isolado):** todo batimento ≠ N gera alerta (+ regra crítica). Mesmo pipeline,
  `engine.use_profile: false`.
- Agrupamento para a UI: alertas a menos de `decision.episode_gap_s` do anterior entram no mesmo episódio
  pendente (combate fadiga de alarmes). Métricas primárias continuam por batimento.

## 7. Engine (`engine/pipeline.py`)

```python
class StreamingPipeline:
    def __init__(self, cfg, classifier: BeatClassifier, record_id: str, *,
                 use_profile: bool, feedback_enabled: bool, rpeak_source: str): ...
    def push(self, chunk: np.ndarray, t0_sample: int, recv_ts: float) -> list[Event]: ...
    def apply_decision(self, alert_id: str, decision: Literal["confirm", "discard"]) -> list[Event]: ...
    @property
    def state(self) -> Literal["calibrating", "monitoring", "paused", "finished"]: ...
```

Ciclo de `push`: filtro causal → detector de R → fila de batimentos pendentes → quando chega o batimento i+1,
classifica i (lote) → features → decisão por batimento → a cada `step_seconds` recalcula KL → eventos.
Cada evento carrega `recv_ts` do bloco que o originou e `emit_ts` → base da medição de latência.

**Latência (ver D-01 em DECISOES.md):** reportar separadamente
(a) *latência de processamento* = `emit_ts − recv_ts` do bloco que completou o evento (alvo RNF01 < 500 ms) e
(b) *atraso algorítmico* = tempo de sinal entre o pico R do batimento e o fim do bloco que permitiu classificá-lo
(≈ 1 RR + até 1 bloco).

## 8. Armazenamento (`storage/db.py`, SQLite)

- `records(record_id, split, duration_s, fs)`
- `sessions(id, record_id, use_profile, feedback_enabled, started_at, ended_at, model_run)`
- `profiles(id, session_id, version, bins_json, q_json, mu, sigma, k, L, created_at)`
- `beats(session_id, sample, cls, prob, attention_json, true_label NULL)` — `true_label` só em avaliação
- `events(id, session_id, sample, level, cls, kl, L, recv_ts, emit_ts)`
- `alerts(id, session_id, event_id, episode_id, status[pending|confirmed|discarded], decided_at)`

## 9. API e WebSocket (`server/`)

REST:
- `GET  /api/records` → registros disponíveis (DS2 na demo; DS1 marcado como "treino").
- `POST /api/sessions` `{record_id, use_profile, feedback_enabled, speed}` → `{session_id}`; dispara o simulador.
- `POST /api/sessions/{id}/pause | resume | stop`
- `POST /api/alerts/{alert_id}/decision` `{decision: "confirm" | "discard"}`
- `GET  /api/sessions/{id}/history` → eventos, alertas e decisões.

WebSockets:
- `/ws/ingest/{session_id}` (simulador → servidor): `{"type":"chunk","t0":int,"fs":360,"samples":[...]}`
- `/ws/monitor/{session_id}` (servidor → dashboard), mensagens com `type`:
  `state` · `signal` (amostras filtradas) · `beat` `{sample, cls, prob, attention}` ·
  `kl` `{t, value, L}` · `event` `{level, sample, cls, kl, L}` · `alert` `{id, episode_id, level, sample, segment, cls, prob, kl, L, attention}` ·
  `profile_updated` `{version}` · `hr` `{bpm}`

O simulador é um processo separado que conversa só por WebSocket → trocar por um dispositivo real não altera
os demais módulos (RNF05).

## 10. Dashboard (`dashboard/`)

React + Vite + TypeScript; gráficos de ECG com **uPlot** (desempenho em streaming).
- **Seleção:** lista de registros, toggles "usar perfil" e "realimentação", velocidade (1×, 5×).
- **Monitor:** faixa de ECG rolando (10 s) com marcadores coloridos por classe; FC; gráfico KL × L;
  selo de estado (calibrando com contagem regressiva / monitorando); lista de alertas pendentes.
- **Detalhe do alerta:** trecho do sinal, classe e probabilidade, KL vs L, batimentos destacados pela atenção,
  botões **Confirmar** / **Descartar**; aviso fixo "ferramenta de apoio à decisão" (RNF07).
- **Histórico:** tabela de eventos/alertas/decisões com filtros.

## 11. Avaliação (`evaluation/`)

1. **Classificador (DS2, por batimento, picos anotados):** acurácia, precisão, sensibilidade, especificidade, F1
   por classe e macro; AUC OvR; matriz de confusão. Q tem pouquíssimos exemplos em DS2 → reportar, mas destacar
   também a média macro sobre N, S, V, F.
2. **Sistema (DS2, replay, feedback desligado, 5 min iniciais excluídos):** configurações A × B com:
   nº de alertas sobre batimentos N (FP), FP/hora, sensibilidade de alerta para S e V, alertas totais/hora,
   episódios/hora, latência de processamento (p50/p95/máx) e atraso algorítmico. Teste de Wilcoxon pareado por
   registro (22 pares) para FP.
3. **Testes funcionais:** sessão completa com especialista simulado (descarta se anotação = N, confirma caso
   contrário) e feedback ligado → mostra queda de alertas repetidos e evolução da versão do perfil.
