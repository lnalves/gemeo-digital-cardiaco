# Registro de decisões

Formato: **D-NN — título** · status (`aceita`, `proposta`, `aberta`) · contexto · decisão · impacto no texto do TC.
Toda mudança de parâmetro ou desvio do PLANO entra aqui com data.

---

## Achados da revisão da arquitetura (12/10/2026)

### D-01 — RNF01 (latência < 500 ms) conflita com o atraso inerente da arquitetura · `aceita` (12/10)
- **Contexto:** a BiLSTM precisa do batimento seguinte (~1 RR ≈ 0,6–1 s), o sinal chega em blocos de 1 s e a KL é
  atualizada a cada 10 s. A Seção 4.4 diz que o atraso de 1 RR "é considerado na medição da latência", mas o RNF01
  exige < 500 ms. Do jeito que está escrito, o requisito não pode ser cumprido.
- **Decisão:** separar duas medidas: (a) **latência de processamento** (chegada do bloco → emissão do evento),
  alvo < 500 ms (RNF01); (b) **atraso algorítmico** (≈ 1 RR + até 1 bloco), reportado à parte. A decisão por
  batimento usa a KL mais recente, sem esperar a próxima janela (ver D-06).
- **Impacto no TC:** reescrever RNF01 como "processar cada bloco do sinal e emitir os eventos correspondentes em
  menos de 500 ms" e ajustar a frase da Seção 4.4 sobre o atraso de 1 RR.

### D-02 — Filtro causal em treino e execução · `aceita`
- **Contexto:** `filtfilt` (fase zero) não pode ser usado em streaming. Treinar com filtfilt e executar com filtro
  causal muda a morfologia vista pelo modelo (atraso de fase).
- **Decisão:** Butterworth SOS causal com estado em ambos. Teste: blocos ≡ sinal inteiro.
- **Impacto no TC:** Seção 3.5/4.5 — "filtro passa-faixa causal, aplicado de forma idêntica no treinamento e na execução".

### D-03 — Picos R: anotados no treino, detector na execução · `proposta`
- **Contexto:** o módulo de pré-processamento "localiza os picos R" (4.1), mas o treino usa as anotações (3.5).
- **Decisão:** `engine.rpeak_source = detector` como alvo, com casamento ±150 ms (EC57) na avaliação.
  `annotation` disponível como modo de referência e como plano B se o detector não ficar pronto até 22/10.
- **Impacto no TC:** citar o detector e a tolerância de 150 ms na Seção 4.5; se usar plano B, declarar como limitação.

### D-04 — Largura do QRS exige delineação · `proposta`
- **Contexto:** a MIT-BIH anota só o pico R; não há onset/offset do QRS.
- **Decisão:** método simples por limiar da derivada em torno do R (parâmetros no config), validado por faixa
  plausível. Alternativa se falhar: `neurokit2.ecg_delineate` offline para validar o método simples.
- **Impacto no TC:** uma frase na Seção 4.5 descrevendo a estimativa.

### D-05 — KL multivariada = soma das KL marginais + bins de transbordo · `proposta`
- **Contexto:** o texto fala em "distribuições das características" sem dizer como combinar as três; histograma
  conjunto 3D com 60–100 batimentos é esparso demais. Bins limitados à faixa da calibração descartariam os
  valores anômalos.
- **Decisão:** `D = Σ_f D_KL(P_f‖Q_f)`; bins entre p1 e p99 da calibração + 2 bins de transbordo; suavização ε.
- **Impacto no TC:** acrescentar 1–2 frases na Seção 4.4 (calibração) ou 2.3.4.

### D-06 — Decisão por batimento usando a KL mais recente · `proposta`
- **Contexto:** o Quadro 5 é definido por janela, mas o FP da avaliação é "alerta emitido sobre batimento normal".
  Janelas sobrepostas (60 s, passo 10 s) contariam o mesmo batimento até 6 vezes.
- **Decisão:** cada batimento ≠ N é decidido uma vez, no momento da classificação, contra a KL da janela mais
  recente. Janelas sem batimentos ≠ N e KL > L geram "atenção" na atualização da janela. Ver ARQUITETURA §6.
- **Impacto no TC:** frase curta na Seção 4.4 explicando que a regra é aplicada a cada batimento classificado.

### D-07 — Subjanelas do limiar com mesmo tamanho/passo da monitorização · `proposta`
- **Contexto:** "o período de calibração é dividido em subjanelas" — sem tamanho definido. Com 60 s sem
  sobreposição, seriam só 5 valores para estimar σ. KL de amostras pequenas tem viés positivo dependente do
  tamanho, então subjanelas menores que a janela de monitorização distorcem L.
- **Decisão:** subjanelas de 60 s com passo de 10 s (≈ 25 valores). Viés por a subjanela estar contida no perfil
  é compensado pelo ajuste de k em DS1-val.
- **Impacto no TC:** especificar na Seção 4.4.

### D-08 — Contexto do classificador: 5 batimentos, 1 futuro · `proposta`
- **Contexto:** o texto diz "sequência de batimentos consecutivos" e atraso de ~1 RR.
- **Decisão:** K = 5 (`i-3 … i+1`), CNN por batimento (TimeDistributed) + features RR, BiLSTM sobre os K
  batimentos, atenção sobre os K. Pesos de atenção = destaque por batimento no painel.
- **Impacto no TC:** detalhar na Seção 4.5 com figura da rede.

### D-09 — Ajuste de k e janela em DS1-val · `proposta`
- **Contexto:** a Seção 4.4 diz que o ajuste usa só DS1. Registros usados para treinar os pesos dariam predições
  otimistas.
- **Decisão:** usar os registros de **DS1-val** (fora do treino dos pesos). Grade: k ∈ {1; 1,5; 2; 2,5; 3},
  janela ∈ {30; 60; 90} s. Critério: minimizar FP sujeito a perda de sensibilidade de alerta (S+V) ≤ 5 p.p. em
  relação à configuração A.
- **Impacto no TC:** Seção 4.5/4.6 — descrever o critério.

### D-10 — Stack e máquina de treino · `aceita` (12/10)
- Python 3.11 · TensorFlow/Keras 3 · wfdb · SciPy · scikit-learn · FastAPI + Uvicorn (WebSocket) · SQLite ·
  React + Vite + TypeScript + uPlot · uv · pytest · ruff.
- **Treino em outra máquina com GPU** (não no MacBook). Desenvolvimento, servidor e dashboard rodam no Mac (CPU).
  - Linux com NVIDIA: `uv sync --extra dev --extra gpu-cuda` (instala `tensorflow[and-cuda]`).
  - Windows: TensorFlow não usa GPU nativamente desde a 2.11 → usar **WSL2** com o mesmo comando do Linux.
  - Fluxo: `git pull` na máquina de treino → `download_data.py` + `build_dataset.py` lá → `train.py` →
    copiar a pasta do run (`artifacts/runs/<run>/`) de volta para o Mac (scp, pendrive ou drive).
    O run precisa ser autocontido: `model.keras`, `config.yaml`, `git_hash`, `metrics.json`, `history.json`.
  - Registrar no run: GPU, versão de CUDA/driver, versão do TF e tempo de treino (vai para a Seção 4.5).
- Inferência no Mac em CPU; a latência (RNF01) é medida no Mac, que é onde o servidor roda.

### D-11 — Métricas com classe Q · `proposta`
- **Contexto:** DS2 tem pouquíssimos batimentos Q (ordem de unidades a dezenas), e F também é raro.
- **Decisão:** reportar as 5 classes e, à parte, macro sobre N, S, V, F, justificando.

### D-12 — Unidade do alerta para a UI: episódios · `proposta`
- Alertas próximos (`decision.episode_gap_s`) formam um episódio, para não inundar o especialista. Métricas
  primárias continuam por batimento; episódios/hora é métrica secundária.

---

## Resolvidas

- **A-01** (12/10) Nova redação do RNF01 aprovada → D-01 aceita.
- **A-02** (12/10) Treino em outra máquina com GPU → D-10.
- **A-03** (12/10) Teste de Wilcoxon pareado (FP por registro, 22 pares, A × B) **entra na Seção 4.6**, com
  estatística, p-valor e α = 0,05. Hipótese unilateral: FP(B) < FP(A).

## Em aberto

- (nenhuma)
