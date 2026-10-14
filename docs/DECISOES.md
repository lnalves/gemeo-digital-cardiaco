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

### D-13 — Carregamento da configuração e download paralelo (F0) · `aceita` (13/10)
- **Config:** `cardiac_twin.config.load_config(path)` lê `configs/default.yaml` em dataclasses congeladas (listas
  viram tuplas). `--config arquivo.yaml` é **mesclado** sobre o default (arquivo parcial basta). Chave desconhecida
  ou ausente → erro, para pegar erro de digitação. `config_to_dict(cfg)` grava a config efetiva no run (regra 7).
- **Download:** o PhysioNet limita a vazão a ~14 kB/s por conexão e o `wfdb.dl_database` usa só 2 conexões
  (~1,7 h para os 48 registros). Novo parâmetro `data.download_workers: 8`: um registro por thread, cada um baixado
  em pasta temporária e movido ao final — download interrompido não deixa arquivo truncado.
- **Ferramentas:** `scripts/check_env.py` confere Python, TF, GPU (`--require-gpu`) e os 48 registros (aceite F0
  nas duas máquinas). Ruff ≥ 0.9 formata blocos de código em Markdown; `*.md` foi excluído do `ruff format`.
- **Impacto no TC:** nenhum.

### D-14 — Máquina de treino: GTX 1050 Ti no WSL2, TensorFlow 2.20 e cuDNN 9.3 · `aceita` (14/10)
- **Hardware:** Windows 11 + WSL2 (Ubuntu 26.04), NVIDIA GeForce GTX 1050 Ti, **4 GB de VRAM**, compute capability
  6.1 (Pascal), driver 582.78; WSL com 12 núcleos e ~7 GB de RAM. Repositório clonado em `~/gemeo-digital-cardiaco`
  (filesystem Linux; `/mnt/c` é lento para I/O).
- **Problema 1 — TF 2.21:** `libtensorflow_framework.so.2` não tem `nvidia/cusolver/lib` no RUNPATH, então o
  `dlopen("libcusolver.so.11")` falha e o TF desiste da GPU ("Cannot dlopen some GPU libraries"), embora o wheel esteja
  instalado. → `tensorflow>=2.16,<2.21` (lock: **2.20.0**, mesma versão no Mac, para treino e inferência baterem).
- **Problema 2 — cuDNN:** o resolvedor escolheu `nvidia-cudnn-cu12` 9.27, que não executa convolução em Pascal
  (`No algorithm worked!`, status 5003) — a GPU aparece na lista, mas o primeiro `Conv1D` quebra. → extra `gpu-cuda`
  fixa `nvidia-cudnn-cu12==9.3.0.75` (versão contra a qual o TF 2.20 foi compilado). Só afeta Linux.
- **Checagem:** `scripts/check_env.py --require-gpu` agora roda um passo de gradiente Conv1D + LSTM na GPU e imprime
  o nome e a compute capability; só listar o dispositivo não detecta o problema 2.
- **Desempenho de referência:** modelo de teste (CNN por batimento + BiLSTM, K = 5, 4096 amostras, lote 256):
  0,64 s/época na GPU × 1,64 s/época na CPU do WSL.
- **Dados na máquina de treino:** o `download_data.py` no WSL foi interrompido por falta de memória no Windows
  (6/48 registros). Os 48 registros foram baixados manualmente do PhysioNet (zip da MIT-BIH 1.0.0), conferidos
  contra o `SHA256SUMS.txt` oficial (144 arquivos `.hea/.dat/.atr`, todos ok) e copiados para `data/raw/mitdb`.
  Conteúdo idêntico ao do `wfdb.dl_database`; os extras do zip (`.xws`, `x_mitdb/`, `mitdbdir/`) não são usados.
- **Consequência para F2:** com 4 GB de VRAM, tamanho de lote e largura da rede precisam caber; se faltar memória,
  reduzir `model.batch_size` (e registrar aqui) antes de mexer na arquitetura.
- **Impacto no TC:** Seção 4.5 — hardware de treino (GPU, VRAM, TF 2.20, CUDA 12.5, cuDNN 9.3).

---

## Resolvidas

- **A-01** (12/10) Nova redação do RNF01 aprovada → D-01 aceita.
- **A-02** (12/10) Treino em outra máquina com GPU → D-10.
- **A-03** (12/10) Teste de Wilcoxon pareado (FP por registro, 22 pares, A × B) **entra na Seção 4.6**, com
  estatística, p-valor e α = 0,05. Hipótese unilateral: FP(B) < FP(A).

## Em aberto

- (nenhuma)
