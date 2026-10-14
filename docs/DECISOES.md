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

### D-03 — Picos R: anotados no treino, detector na execução · `aceita` (14/10, ver D-16)
- **Contexto:** o módulo de pré-processamento "localiza os picos R" (4.1), mas o treino usa as anotações (3.5).
- **Decisão:** `engine.rpeak_source = detector` como alvo, com casamento ±150 ms (EC57) na avaliação.
  `annotation` disponível como modo de referência e como plano B se o detector não ficar pronto até 22/10.
- **Impacto no TC:** citar o detector e a tolerância de 150 ms na Seção 4.5; se usar plano B, declarar como limitação.

### D-04 — Largura do QRS exige delineação · `aceita` (14/10, ver D-17)
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
- **F1 (14/10):** `segmentation.context_indices` só gera sequências completas dentro de um registro → os 3 primeiros
  e o último batimento de cada registro não são classificados (~88 de ~50 mil em DS2). Confirmar na F2.
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

### D-15 — Escolha do DS1-val por restrições + sorteio · `aceita` (14/10)
- **Contexto:** as arritmias de DS1 estão concentradas: o 208 tem 373 dos 415 F; o 209 tem 383 dos 944 S. Um sorteio
  cego de 4 registros pode tirar quase todo o F do treino ou deixar o val sem S/V (early stopping por macro-F1 inútil).
- **Decisão:** `select_val_records` enumera as C(22,4) combinações de DS1, mantém as que têm **S ≥ 100, V ≥ 200,
  F ≥ 10** (`data.val_min_beats`) e **no máximo 35 % do S, V e F de DS1** (`data.val_max_fraction`), e sorteia uma com
  `seed` = 42. A primeira versão (sem mínimo de F) deu 1 único F no val → F1 da classe F seria 0 ou 1; acrescentado
  F ≥ 10. Só lê anotações de DS1 (`scripts/select_val_records.py`).
- **Resultado (fixado em `data.val_records`):** **101, 124, 205, 207** → val N = 7510, S = 144, V = 328, F = 16, Q = 2;
  treino (18 registros) fica com 85 % do S, 91 % do V e 96 % do F de DS1.
- **Consequência para F2:** Q tem só 2 batimentos no val → o early stopping deve usar macro-F1 sobre N, S, V, F (D-11).
- **Impacto no TC:** Seção 4.5 — uma frase sobre o critério de escolha do conjunto de validação.

### D-16 — Detector de R: parâmetros e refino do R para alinhar treino e execução · `aceita` (14/10)
- **Detector:** Pan-Tompkins simplificado e causal (derivada de 5 pontos → quadrado → integração de 150 ms; pico
  candidato = máximo em ±200 ms; limiar adaptativo SPK/NPK; busca de volta após 1,66·RR médio; rejeição de onda T
  em até 360 ms). Cada candidato só é decidido quando chegam 200 ms depois dele → saída idêntica para qualquer
  divisão em blocos (testado com blocos de 1, 250, 360 e 2520 amostras e num registro real).
- **Ajuste só em DS1:** grade `threshold_frac` ∈ {0,15; 0,2; 0,25} × `searchback_threshold_scale` ∈ {0,3; 0,5} ×
  `twave_slope_frac` ∈ {0,3; 0,5}. Escolhidos 0,15 / 0,3 / 0,5 (maior Se + VPP). Com 0,25 / 0,5 o 208 tinha Se = 94 %
  (V largos abaixo do limiar). **DS1 final: Se = 99,79 %, VPP = 99,11 %** (TP = 50 912, FP = 456, FN = 109); sem o
  207 (flutter ventricular, ondas anotadas como não batimento), VPP = 99,79 %.
- **Desvio do PLANO:** o teste de aceite (Se e VPP > 99 %) usa os registros limpos **112, 115, 122 (DS1)** em vez
  de 100, 103, 113, que são DS2 — ajustar o detector até passar neles violaria o RNF02. O desempenho em DS2 sai
  na avaliação da F4.
- **Refino do R (`segment.r_refine_ms` = 50):** o filtro causal atrasa o pico ~4 amostras (11 ms) em relação à
  anotação, feita no sinal bruto. Treinar com janelas na anotação e executar com janelas no pico detectado deslocaria
  a morfologia vista pelo modelo. Regra única nos dois caminhos: R = máximo de |x| filtrado em ±50 ms da posição de
  referência (anotação no treino, detector na execução). Em DS1, detector e anotação refinados coincidem na mesma
  amostra em 98,7 % dos batimentos casados (99,1 % com |Δ| ≤ 2 amostras), contra 0,4 % sem refino.
- **D-03:** passa a `aceita` com `engine.rpeak_source: detector`.
- **Impacto no TC:** Seção 4.5 — detector, tolerância de 150 ms e refino do R; números de DS1 acima (DS2 na 4.6).

### D-17 — Largura do QRS: limiar de inclinação 0,1 · `aceita` (14/10)
- **Método (D-04):** em ±100 ms do R, limiar = `qrs_slope_frac` × max |Δx|; início/fim = primeira/última amostra com
  inclinação acima dele.
- **Mudança:** `qrs_slope_frac` 0,2 → **0,1**. Com 0,2, a mediana do QRS de batimentos N em DS1 ficava em 42–69 ms na
  maioria dos registros (abaixo do fisiológico, ~80–100 ms) e só 55 % dos registros em 60–200 ms; com 0,1, 22/22
  registros de DS1 ficam em 60–200 ms (mediana por registro), bloqueios de ramo em 119–147 ms (109, 118, 124, 207) e
  V mais largos que N. Ajuste feito só em DS1.
- **Impacto no TC:** Seção 4.5 — uma frase com o método e o limiar.

### D-18 — Treino: perdas, early stopping e normalização das RR · `aceita` (14/10)
- **Perdas:** `weighted_ce` = entropia cruzada com pesos n / (C · n_c), teto `model.class_weight_cap` = 50 (Q, com 6
  batimentos no treino, fica no teto). `focal` = mesma perda com o fator (1 − p)^γ, γ = 2, e **os mesmos pesos como
  α**. Com γ = 0 as duas são idênticas, então a comparação isola o efeito de γ.
- **Early stopping:** macro-F1 sobre N, S, V, F (`evaluation.macro_classes`) no DS1-val ao fim de cada época;
  paciência 8; restaura os pesos da melhor época. Q fica fora (2 batimentos no val, D-11/D-15).
- **RR:** camada `Normalization` com média e variância do DS1-train, salva dentro do `model.keras` (sem estado
  externo). As RR entram no modelo pelas colunas de `model.rr_features`; o `.npz` guarda todas.
- **Seeds:** `keras.utils.set_random_seed(seed)`. Na GPU o treino não é bit a bit determinístico (cuDNN); por isso as
  comparações usam 2–3 seeds. A inferência é determinística e portátil: a matriz de confusão do DS1-val na CPU do
  Windows é idêntica à do treino na GPU do Linux.
- **Execução:** GTX 1050 Ti, ~20 s por época, 4–7 min por treino (registrado em cada `run.json`).

### D-19 — RR relativo ao ritmo habitual do paciente · `aceita` (14/10)
- **Contexto:** o primeiro treino (5 features RR do ARQUITETURA §3) tinha Se de S ≈ 0,04 no DS1-val. Diagnóstico por
  registro: 105 dos 142 S do val estão numa corrida de taquicardia supraventricular no 207. Dentro da corrida,
  RR pré / RR local ≈ 0,99: o batimento não é prematuro em relação aos vizinhos, só em relação ao ritmo habitual.
- **Decisão:** 3 features novas, RR pré, pós e local divididos pela média causal dos últimos
  `features.long_rr_beats` = 300 intervalos (~5 min). No 207, S tem RR pré / RR longo = 0,40 contra 1,02 nos N.
  É a mesma ideia de personalização do perfil (comparar o paciente com ele mesmo) aplicada à entrada do modelo, e
  funciona em streaming (só usa batimentos anteriores). `model.rr_features` passa a ter as 8 colunas.
- **Resultado (DS1-val, focal, média ± dp de 2 seeds):** macro-F1 0,469 ± 0,003 → **0,605 ± 0,016**; Se de S
  0,04 → 0,77. Runs: `20261014-151610_train_focal`, `…153038_train_focal_s43_e0_base`,
  `…153500_train_focal_s42_e1_long_rr`, `…154129_train_focal_s43_e1_long_rr`.
- **Impacto no TC:** Seção 4.5 — descrever as 8 features RR. Pode ser citado como evidência a favor da
  personalização (Seção 4.6, discussão), deixando claro que é medido no DS1-val.

### D-20 — Experimentos descartados: aumento de dados e diferença para o batimento habitual · `aceita` (14/10)
- **E2 — aumento de dados** (`augment_noise_std` = 0,05, `augment_baseline_std` = 0,1, só no treino): 0,613 ± 0,008
  contra 0,605 ± 0,016 do E1, dentro do ruído; uma seed parou na época 4. Não adotado (mantido desligado na config).
- **E3 — canal `beat_diff`** (batimento menos a mediana causal dos 50 anteriores, `segment.template_beats`): a feature
  separa V de N isoladamente (RMS mediano 0,65 × 0,08 no 207), mas o modelo piorou: 0,560 ± 0,025, melhor época 3,
  S caiu. Não adotado; o canal continua no `.npz` e em `model.beat_channels` para quem quiser repetir.
- **Cuidado com o val:** só 5 configurações foram comparadas, todas por hipóteses tiradas do diagnóstico do DS1-val,
  para limitar o sobreajuste ao val. Runs com tag `e2_long_rr_aug` e `e3_long_rr_template`.
- **Impacto no TC:** opcional — uma frase sobre o que foi testado e não ajudou.

### D-21 — Escolha da perda e do modelo final, antes de qualquer avaliação em DS2 · `aceita` (14/10)
- **Comparação (E1, 3 seeds cada, DS1-val, macro-F1 N, S, V, F):** focal **0,6021 ± 0,0126**
  (0,5934 / 0,6165 / 0,5963) × weighted_ce **0,6020 ± 0,0099** (0,5987 / 0,5941 / 0,6132). Empate: a diferença é
  muito menor que a variação entre seeds.
- **Decisão:** aplicar o critério fixado no PLANO (maior macro-F1 médio no DS1-val) → **focal**, declarando o
  empate. Modelo final = seed de maior macro-F1 no val: **`20261014-154129_train_focal_s43_e1_long_rr`**
  (0,6165; melhor época 14). `model.loss` e `model.rr_features` da config padrão passam a refletir essa escolha.
- **DS2:** avaliar o modelo final; avaliar também as outras 2 seeds do focal só para reportar a variação entre
  seeds (média ± dp), sem nova escolha. Nenhum ajuste depois de ver DS2.
- **Impacto no TC:** Seção 4.5 — "as duas perdas tiveram desempenho equivalente na validação; adotou-se a focal".

---

## Resolvidas

- **A-01** (12/10) Nova redação do RNF01 aprovada → D-01 aceita.
- **A-02** (12/10) Treino em outra máquina com GPU → D-10.
- **A-03** (12/10) Teste de Wilcoxon pareado (FP por registro, 22 pares, A × B) **entra na Seção 4.6**, com
  estatística, p-valor e α = 0,05. Hipótese unilateral: FP(B) < FP(A).

## Em aberto

- (nenhuma)
