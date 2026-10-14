# Plano de implementação

**Prazo duro:** 28/10/2026 — entrega do TC até o Cap. 4 inclusive (Seções 4.5 Implementação e 4.6 Testes precisam
de conteúdo real). Prioridade: **números da 4.6 primeiro** (F1–F4), interface depois (F5–F6).

Marque `[x]` ao concluir. Cada fase termina com testes verdes e um "Resumo para o TC".

| Fase | Entrega | Janela | Bloqueia |
|---|---|---|---|
| F0 | Ambiente + dados baixados | 13/10 | tudo |
| F1 | Dados, filtro, batimentos, features, splits | 13–15/10 | F2, F3 |
| F2 | Classificador treinado + métricas DS2 | 15–19/10 | F4 |
| F3 | Perfil KL + regras de decisão | 16–19/10 (paralelo a F2) | F4 |
| F4 | Engine streaming + comparação A×B | 19–22/10 | **números da 4.6** |
| F5 | Servidor, simulador, SQLite | 21–23/10 | F6 |
| F6 | Dashboard | 23–26/10 | screenshots da 4.5 |
| F7 | Testes funcionais, figuras, textos 4.5/4.6 | 25–28/10 | entrega |

---

## F0 — Ambiente

- [x] Mac (desenvolvimento/servidor): `uv sync --extra dev` (Python 3.11, CPU).
- [x] Máquina de treino: clonar o repo, `uv sync --extra dev --extra gpu-cuda` (WSL2 se for Windows) e confirmar
      `tf.config.list_physical_devices('GPU')` não vazio. Ver D-10. GTX 1050 Ti no WSL2; TF 2.20 + cuDNN 9.3 e
      `check_env.py --require-gpu` com passo de treino real (D-14).
- [x] `scripts/download_data.py`: baixa `mitdb` via `wfdb.dl_database` para `data/raw/mitdb` (idempotente).
- [x] `config.py`: carrega `configs/default.yaml` em dataclasses congeladas; `--config` sobrescreve.
- [x] `ruff` e `pytest` rodando (mesmo sem testes).

**Aceite:** 48 registros presentes nas duas máquinas; `uv run pytest` executa; `python -c "import tensorflow"` ok.

## F1 — Dados e sinal

- [x] `data/labels.py` — mapa AAMI; símbolos de não-batimento (`+ ~ | " x [ ] ! ...`) ignorados.
- [x] `data/splits.py` — DS1, DS2, PACED; `assert` disjunção; `select_val_records(ds1, seed)` garante S, V e F em DS1-val (D-15).
- [x] `data/mitbih.py` — `load_record(id) -> (signal_mlii, fs, r_samples, symbols)`.
- [x] `signal/filters.py` — `CausalBandpass` com `process(chunk)` (estado) e `process_full(x)`.
- [x] `signal/segmentation.py` — recorte [R-90, R+166], z-score, descarte de bordas; montagem de contexto K.
- [x] `signal/features.py` — RR (pré, pós, local, razões), largura QRS, amplitude R.
- [x] `signal/rpeaks.py` — `StreamingRPeakDetector` + `match_peaks(det, ann, tol=150ms)` + refino do R (D-16).
- [x] `scripts/build_dataset.py` — gera `data/processed/ds1.npz` e `ds2.npz` (beats, rr_feats, labels, record_id,
      r_sample) + `stats.json` com contagem por classe e registro.
- [ ] Notebook `notebooks/01_eda.ipynb` (opcional): distribuição de classes, exemplos de batimentos por classe,
      histogramas das 3 features em 2–3 pacientes (vira figura do TC).

**Testes:** mapa AAMI; DS1∩DS2=∅ e PACED fora; filtro em blocos ≡ filtro inteiro (tolerância 1e-9); detector com
sensibilidade > 99 % e VPP > 99 % em 3 registros limpos de DS1 (112, 115, 122 — D-16); largura QRS mediana em 60–200 ms em
≥ 90 % dos registros; contagens por classe batem com a literatura (DS1 ≈ 51 mil, DS2 ≈ 49,7 mil batimentos).

**Resumo para o TC:** contagem de batimentos por classe em DS1/DS2 (tabela), parâmetros do filtro, janela do
batimento, método de largura do QRS.

## F2 — Classificador

- [x] `model/architecture.py` — conforme ARQUITETURA §4; `build_model(cfg)` e `build_inference_model` (expõe atenção).
- [x] `model/losses.py` — focal loss; pesos de classe com teto `model.class_weight_cap`.
- [x] `model/dataset.py` — `tf.data` com sequências K; embaralhamento só no treino.
- [x] `model/train.py` + `scripts/train.py --loss {weighted_ce,focal}` — early stopping por macro-F1 em DS1-val;
      salva `model.keras`, `history.json`, `config.yaml`, `git_hash`.
- [x] `model/inference.py` — `BeatClassifier.predict(beats, rr) -> (cls, prob, attention)`.
- [x] `evaluation/metrics.py` + `scripts/evaluate_classifier.py` — métricas Eq. 4–8 por classe, macro (5 e 4
      classes), AUC OvR, matriz de confusão (PNG + CSV).
- [x] Treinar na máquina com GPU e trazer o run para o Mac (fluxo em D-10). Runs em `artifacts/runs/` do Windows; portabilidade GPU→CPU conferida (D-18). Falta copiar para o Mac.
- [x] Rodar as duas perdas; escolher a de maior macro-F1 em **DS1-val** (não em DS2!) e registrar em DECISOES. Empate; focal pelo critério (D-21).

**Testes:** shapes de saída; atenção soma 1; um passo de treino reduz a perda em lote sintético; `BeatClassifier`
determinístico.

**Referência de sanidade (interpaciente, DS2):** literatura reporta acurácia ~ 0,90–0,95, sensibilidade de S
frequentemente < 0,6. Resultado muito acima disso → procurar vazamento antes de comemorar.

**Resumo para o TC:** figura da rede, hiperparâmetros, tempo de treino e hardware, tabela de métricas DS2,
matriz de confusão.

## F3 — Perfil e decisão

- [ ] `profile/histogram.py` — bins por feature (p1–p99 + transbordo), `histogram(values, bins, eps)`.
- [ ] `profile/kl.py` — `kl(p, q)`, `kl_sum(P, Q)`.
- [ ] `profile/profile.py` — `PatientProfile.build(calib_features)`, `.threshold(k)`, `.score(window_features)`,
      `.update(P, alpha)`, serialização JSON, `version`.
- [ ] `decision/rules.py` — `decide_beat(cls, recent_v_run, kl, L, use_profile)`, `decide_window(has_non_n, kl, L)`,
      contador de V consecutivos.
- [ ] `scripts/tune_profile.py` — grade de D-09 sobre DS1-val, usando o classificador de F2; salva tabela e escolha.

**Testes:** KL ≥ 0, KL(P,P)=0, finita com bins vazios; perfil atualizado soma 1 e fica entre Q e P; tabela-verdade
completa do Quadro 5 (parametrizada); 3 V consecutivos → crítico mesmo com KL ≤ L e durante calibração; A ignora KL.

**Resumo para o TC:** valores finais de n_bins, ε, janela, passo, k, α e o critério de escolha.

## F4 — Engine e avaliação do sistema  ← **números da Seção 4.6**

- [ ] `engine/events.py` e `engine/pipeline.py` — conforme ARQUITETURA §7; estados calibrating → monitoring.
- [ ] `evaluation/replay.py` — alimenta o pipeline em blocos de 1 s sem dormir; registra `recv_ts/emit_ts` e
      rótulos verdadeiros (casamento ±150 ms).
- [ ] `evaluation/compare.py` + `scripts/evaluate_system.py` — config A × B em todos os 22 registros DS2,
      feedback desligado, 5 min iniciais excluídos. Saídas: `per_record.csv`, `summary.json`, figuras
      (FP por registro A vs B, KL × L ao longo do tempo para 2 registros exemplares, boxplot de latência).
- [ ] Wilcoxon pareado unilateral (FP(B) < FP(A)) por registro, α = 0,05 — **entra na Seção 4.6** (A-03):
      reportar estatística W, p-valor e mediana das diferenças.

**Testes:** pipeline em blocos ≡ pipeline com sinal inteiro (mesmas classes); nenhum evento antes do fim da
calibração exceto crítico; FP de A ≥ FP de B por construção (B é subconjunto de A) — se falhar, há bug;
latência p95 < 500 ms em CPU.

**Resumo para o TC:** tabela A × B (FP, FP/h, sensibilidade S e V, alertas/h, latência p50/p95, atraso
algorítmico médio), p-valor, 2 figuras, 1 parágrafo honesto sobre o custo em sensibilidade.

## F5 — Servidor, simulador, armazenamento

- [ ] `storage/db.py` — esquema de ARQUITETURA §8; migração simples por `CREATE TABLE IF NOT EXISTS`.
- [ ] `server/schemas.py` + `server/app.py` — REST e WS de ARQUITETURA §9; uma pipeline por sessão; broadcast para
      vários dashboards; CORS para `localhost:5173`.
- [ ] `acquisition/simulator.py` + `scripts/run_simulator.py` — envia blocos de 1 s em tempo real (`--speed`).
- [ ] `POST /alerts/{id}/decision` chama `pipeline.apply_decision` e persiste; descarte com feedback ligado
      atualiza perfil e emite `profile_updated`.

**Testes:** `TestClient` FastAPI: criar sessão, enviar 10 s sintéticos, receber `beat`; decisão persiste;
trocar a fonte (cliente WS de teste) sem tocar no servidor (RNF05).

## F6 — Dashboard

- [ ] Scaffold `npm create vite@latest dashboard -- --template react-ts`; uPlot; cliente WS com reconexão.
- [ ] Telas: Seleção, Monitor, Detalhe do alerta, Histórico (ARQUITETURA §10). Textos em português.
- [ ] Cores por classe consistentes com as figuras do TC; aviso RNF07 sempre visível.
- [ ] Capturas de tela das 4 telas em `docs/figuras/` para a Seção 4.5.

**Aceite:** sessão de 30 min a 5× sem travar; navegador sem instalação (RNF04); latência visual aceitável.

## F7 — Testes funcionais e material do TC

- [ ] `tests/functional/test_feedback.py` — especialista simulado (descarta se anotação N), feedback ligado;
      mostra queda de alertas repetidos e versões do perfil.
- [ ] Atualizar `docs/RASTREABILIDADE.md` com o teste que cobre cada RF/RNF.
- [ ] `docs/TC_4.5_rascunho.md` e `docs/TC_4.6_rascunho.md` — texto em português, ABNT, pronto para colar no
      Google Doc, usando apenas números de `artifacts/runs/` (citar o run).
- [ ] Figuras finais em `docs/figuras/` (arquitetura da rede, matriz de confusão, A×B, KL×L, telas).

**Aceite:** `uv run pytest` verde; todo número dos rascunhos rastreável a um arquivo de run.
