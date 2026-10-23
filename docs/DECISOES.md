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

### D-09 — Ajuste de k e janela em DS1-val · `aceita` (14/10, critério revisto em D-26)
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

### D-22 — Resultado em DS2 e principal confusão (N → F) · `aceita` (14/10)
- **Modelo final em DS2** (run `20261014-163721_eval_classifier_focal`): acurácia 0,780; Se/+P — N 0,777/0,981,
  S 0,581/0,456, V 0,926/0,804, F 0,912/0,041; macro-F1 (N, S, V, F) 0,579. Três seeds: acurácia 0,804 ± 0,057,
  macro-F1 0,586 ± 0,010.
- **Achado:** 8 100 batimentos N (18,4 %) são classificados como F, concentrados em 213 (2 407), 233 (1 866) e
  212 (1 172). No DS1-val só 0,5 % dos N iam para F: nenhum registro do val tem esse padrão. A acurácia fica abaixo
  da faixa de referência (0,90–0,95) por causa disso; S e V estão na faixa ou acima do que a literatura
  interpaciente reporta. Não é sinal de vazamento (o resultado é pior, não melhor, que a referência).
- **Decisão:** nenhum ajuste depois de ver DS2 (D-21). O excesso de alarmes do classificador isolado é registrado
  como limitação e é exatamente o que a configuração B (perfil) deve reduzir na F4.
- **D-08:** confirmado — 88 batimentos de DS2 (3 primeiros e último de cada registro) ficam sem contexto e não são
  avaliados (49 580 de 49 668).
- **Inferência:** 0,50 ms por batimento na CPU do Windows (lote de 256), bem abaixo do orçamento do RNF01.
- **Impacto no TC:** Seção 4.6 — tabela por classe, matriz de confusão e um parágrafo sobre N → F.

### D-23 — μ e σ do limiar por leave-one-out nas subjanelas da calibração · `aceita` (14/10, A-04)
- **Problema:** com D de cada subjanela medido contra o Q da calibração inteira, a própria subjanela já está em Q:
  todo valor dela tem massa, e D sai baixo (μ ≈ 0,3–1,0 no DS1-val). Janelas novas trazem valores em bins vazios
  da calibração (Q ≈ ε) e D sobe (mediana ≈ 3–4). Resultado no DS1-val, k = 2: 100 % das janelas só com N e 97 % das
  com não-N acima de L — a configuração B quase não difere da A. A D-07 previa o viés, mas nenhum k da grade (1–3)
  o compensa.
- **Medido no DS1-val** (fração das janelas pós-calibração com D > L):

  | ε | leave-one-out | k | só N | com não-N |
  |---|---|---|---|---|
  | 10⁻⁶ | não | 2 | 100 % | 97 % |
  | 10⁻⁶ | **sim** | 2 | **18 %** | **36 %** |
  | 0,5 | sim | 2 | 85 % | 74 % |
  | 1 | sim | 2 | 86 % | 76 % |

- **Decisão (aprovada pelo autor, A-04):** cada subjanela é comparada com um Q montado com a calibração **sem** ela
  (leave-one-out). O Q usado na monitorização e os bins continuam vindo dos 5 min inteiros; ε, bins e a Eq. 11 não
  mudam. Teste `test_stationary_signal_rarely_exceeds_threshold`: num sinal sintético estacionário, 74 % das
  janelas passavam de L no método antigo e menos de 20 % no novo. ε maior (pseudocontagem) piora e não foi adotado.
- **Impacto no TC:** Seção 4.4 — 1–2 frases: "para estimar a variação natural de D sem viés, cada subjanela da
  calibração é comparada com o perfil construído sem ela".

### D-24 — Calibração igual nas duas configurações; features do perfil · `aceita` (14/10)
- Durante os 5 min de calibração só existe alerta crítico, **em A e em B**: as duas configurações ficam iguais nesse
  trecho, que é excluído das métricas de qualquer forma (regra 6).
- A primeira KL é calculada no fim da calibração (janela que termina em 300 s), para que os batimentos logo após a
  calibração já tenham "KL atual". Batimento ≠ N sem KL disponível em B → atenção.
- `profile.features` passa a usar os nomes de `signal.features.beat_features`: `rr_pre`, `qrs_width_ms`,
  `r_amplitude`. σ das subjanelas com ddof = 1 (25 valores).
- **`tune_profile.py` depois do pipeline:** para respeitar a regra 2, o ajuste de k e janela roda sobre o replay
  do `StreamingPipeline` (F4) no DS1-val, não sobre uma simulação própria. Ordem: F3 (módulos) → F4 (pipeline e
  replay) → `tune_profile.py` → avaliação em DS2.

### D-25 — StreamingPipeline: atraso de ~2 RR, "KL atual" e inferência compilada · `aceita` (14/10)
- **Atraso algorítmico ≈ 2 RR:** o batimento futuro da sequência (i+1) entra no modelo com o próprio RR pós, que só
  existe quando chega o batimento i+2. É o que o modelo viu no treino, então o streaming espera. No 205 (DS1-val),
  atraso médio entre o pico R e o fim do bloco que permitiu classificá-lo: **2,15 s**. Reportar à parte (D-01).
- **"KL atual" de um batimento** = última janela que termina **até o instante do batimento**, e não a última calculada
  no momento da decisão. Calibrando = nenhuma janela até esse instante. Assim o resultado não depende do tamanho do
  bloco (testado com blocos de 1 s, 7,04 s e sinal inteiro: mesmos batimentos, classes, níveis, KL e alertas).
- **Mesmo código que o treino:** em modo `annotation`, as entradas que o pipeline entrega ao modelo (janelas e RR)
  são iguais às do `ds1.npz` (tolerância de 1e-5) — teste `test_model_inputs_equal_the_training_dataset`.
- **Janela com não-N:** a decisão da janela (Quadro 5) sai quando todos os batimentos dela já foram classificados.
- **Episódios:** alerta a mais de `decision.episode_gap_s` do anterior abre episódio novo.
- **Buffer:** `engine.signal_buffer_seconds` = 10 s de sinal filtrado, para recortar batimentos detectados com atraso.
- **Inferência compilada (`tf.function`)** no `BeatClassifier`: replay de 30 min passou de 118 s para 4 s na CPU do
  Windows, com resultados idênticos. Latência de processamento por bloco de 1 s no 205: p50 2,1 ms, p95 2,5 ms.
- **Impacto no TC:** Seção 4.5 — atraso de ~2 RR (corrige a frase de "1 RR" da Seção 4.4); latência em 4.6.

### D-26 — Critério de k e janela: queda de até 15 p.p. na sensibilidade S+V · `aceita` (14/10, A-06)
- **Contexto:** no DS1-val, com o critério original da D-09 (queda ≤ 5 p.p.), nenhum ponto da grade k ∈ {1…3} ×
  janela ∈ {30, 60, 90} s era viável (queda mínima 11,7 p.p.). Ampliando para k ∈ {−0,5; 0; 0,5}, o único viável útil
  era k = 0 / 30 s, com FP só −4 %. O perfil troca FP por sensibilidade de forma contínua: um S ou V isolado entre
  batimentos normais quase não muda o histograma da janela, a KL fica abaixo de L e o alerta vira atenção.
- **Decisão (do autor, A-06):** limite de queda passa a **15 p.p.** (`profile.max_sensitivity_drop_pp`); grade
  `k` ∈ {0; 0,5; 1; 1,5; 2; 2,5; 3} × janela ∈ {30, 60, 90} s. Critério: menor FP entre os pontos viáveis.
- **Resultado (run `20261014-173844_tune_profile`, 4 registros do DS1-val, 1,67 h após a calibração, 350 S+V):**
  A: FP = 181, Se S+V = 0,823. Escolha: **k = 1,5, janela = 30 s** → FP = 48 (−73 %), Se S+V = 0,683 (−14,0 p.p.).
  A tabela completa (21 pontos) está em `tuning.csv` do run e é a curva de compromisso para o TC.
- **D-09:** passa a `aceita` com este critério. Runs exploratórios do mesmo dia: `…173002_tune_profile` (grade
  original) e `…173422_tune_profile` (k baixos).
- **Impacto no TC:** Seção 4.5 — critério de escolha; Seção 4.6 — mostrar a curva FP × sensibilidade do DS1-val e o
  ponto escolhido, e discutir que o ganho em FP tem custo em sensibilidade (parágrafo honesto pedido na F4).

### D-27 — Avaliação A × B em DS2: FP cai 69 %, sensibilidade de V cai 78 p.p. · `aceita` (22/10)
- **Protocolo:** 22 registros de DS2, detector de R (`engine.rpeak_source: detector`), k = 1,5 e janela de 30 s da
  D-26 sem novo ajuste, realimentação desligada, 300 s iniciais fora das métricas (9,20 h, 36 860 N, 1 608 S,
  2 686 V). Run `20261022-184514_eval_system`; o run `…183805_eval_system` tem os mesmos números e foi refeito só para
  gravar o método do Wilcoxon e o diagnóstico por registro.
- **Resultado:** FP 8 279 → 2 581 (−68,8 %; 900 → 281 por hora), menor em 20 de 22 registros, igual em 2 (100, 202).
  Wilcoxon pareado unilateral FP(B) < FP(A): W = 0, p = 4,4·10⁻⁵, mediana das diferenças B − A = −29. Sensibilidade
  de alerta: S 0,618 → 0,440; V 0,988 → 0,207; S+V 0,849 → 0,294 (−55,5 p.p., contra −14,0 p.p. no DS1-val).
- **Causa da queda de V (verificada, não é bug):** o perfil aprende o padrão dos 5 min iniciais, incluindo a
  ectopia habitual. Nove registros têm 10 ou mais V na calibração (105, 200, 210, 213, 214, 219, 221, 228, 233) e
  concentram 2 662 dos 2 686 V. Neles, a KL passa de L em só 9–60 % das janelas, e B alerta 539 V (20 %). Nos outros
  13 registros B alerta 16 de 24 V (67 %). O DS1-val não tinha esse padrão na mesma proporção, por isso a queda lá
  foi menor. Os alertas críticos (≥ 3 V seguidos) continuam, porque não dependem da KL.
- **Wilcoxon:** 2 diferenças nulas e 3 empatadas → aproximação normal (`method: approx` no `summary.json`). É
  conservadora: como as 20 diferenças não nulas são todas negativas, o p exato condicionado seria 2⁻²⁰ ≈ 9,5·10⁻⁷.
- **Latência:** tempo de processamento por bloco de 1 s, p50 2,2 ms, p95 2,7 ms, máx. 110 ms (A) e 10 ms (B), na
  CPU do Windows. Atraso algorítmico médio 2,38 s (D-25). O `BeatClassifier` agora traça o grafo ao carregar: antes,
  o primeiro bloco pagava a compilação (~480 ms) e quase estourava o orçamento.
- **Decisão:** nenhum ajuste depois de ver DS2. O custo em sensibilidade é resultado do TC e entra na Seção 4.6.
- **Impacto no TC:** Seção 4.6 — tabela A × B, p-valor, figuras `f4_*` e o parágrafo sobre o custo. Seção 5
  (conclusão) — a hipótese de redução de FP se confirma, mas o perfil, como definido, silencia arritmias que o
  paciente já apresentava na calibração.

### D-28 — Servidor, simulador e SQLite: contrato e escolhas de implementação · `aceita` (23/10)
- **Uma `StreamingPipeline` por sessão**, a mesma classe da avaliação offline (regra 2). O servidor usa sempre o
  detector de R: na execução não há anotações. O `push` roda numa thread de trabalho, com um lock por sessão; a
  decisão do especialista usa o mesmo lock.
- **Fonte da sessão:** `POST /api/sessions` com `source: "simulator"` (padrão) dispara `scripts/run_simulator.py` como
  outro processo; com `source: "external"` o servidor só espera a fonte. O simulador fala só por WebSocket, então um
  dispositivo real entra no lugar dele sem mudar o servidor (RNF05). O teste usa blocos de 0,37 s e de 1 s e recebe
  os mesmos batimentos.
- **Protocolo de `/ws/ingest` ampliado:** `hello` com `next_t0` (a fonte retoma de onde parou depois de cair),
  `control` (pause/resume/stop) e `end` (fim do registro encerra a sessão). Pausar pede à fonte que pare: os blocos
  que já estão a caminho ainda são processados, para não abrir buraco no sinal. Depois da pausa, o simulador
  recomeça o relógio, sem rajada para recuperar o atraso.
- **Alerta com trecho do sinal:** o evento `Alert` do engine passa a levar os picos R dos K batimentos de contexto e
  o sinal filtrado entre eles, para o Detalhe do alerta (atenção sobre os batimentos). Não muda nenhuma decisão.
- **Esquema (ARQUITETURA §8):** `events.kind` (beat | window) e `alerts.payload_json` (a mensagem `alert` inteira).
  Só eventos de nível ≠ normal vão para `events`; todo batimento vai para `beats` (`prob` = probabilidade da classe).
  O id do alerta no banco é `<sessão>-<amostra>`: o id do engine (`<registro>-<amostra>`) repete entre sessões.
- **Difusão:** uma fila por dashboard; a fila cheia (`server.monitor_queue_size` = 2000, ~6 min de mensagens a 5×)
  desconecta o dashboard lento (código 1013), que reconecta. A mensagem `signal` fecha as mensagens de cada bloco.
- **Parâmetros novos:** `server.default_speed` = 1,0, `server.monitor_queue_size` = 2000, `server.hr_beats` = 10
  (FC = média dos últimos 10 RR, só exibição).
- **Lista de registros:** `GET /api/records` monta a lista em `data/mitbih.record_catalog`. Assim o teste de
  vazamento continua sem mudança: o servidor não cita DS2. A demo usa DS2 só para mostrar o sistema, sem treino nem
  ajuste (RNF02).
- **Verificação:** 14 testes em `tests/test_server.py` (TestClient e um uvicorn real com o simulador), 4 em
  `tests/test_storage.py`. Ensaio manual com o registro 100 a 200× e classificador falso: o servidor dispara o
  simulador, a pausa segura o envio, os 1 806 blocos chegam, e a sessão termina com o perfil gravado.
- **Impacto no TC:** Seção 4.5 — protocolo de ingestão e monitoramento, persistência e troca da fonte (RNF05).

---

## Resolvidas

- **A-01** (12/10) Nova redação do RNF01 aprovada → D-01 aceita.
- **A-02** (12/10) Treino em outra máquina com GPU → D-10.
- **A-03** (12/10) Teste de Wilcoxon pareado (FP por registro, 22 pares, A × B) **entra na Seção 4.6**, com
  estatística, p-valor e α = 0,05. Hipótese unilateral: FP(B) < FP(A).
- **A-04** (14/10) μ e σ do limiar por leave-one-out nas subjanelas da calibração → D-23.
- **A-05** (14/10) Rodada de melhorias do classificador antes de tocar em DS2, só com DS1-val → D-19 a D-21.
- **A-06** (14/10) Queda máxima de sensibilidade S+V no ajuste de k passa de 5 para 15 p.p. → D-26.

## Em aberto

- (nenhuma)
