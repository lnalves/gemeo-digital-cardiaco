# CLAUDE.md — Gêmeo digital cardíaco (TC UNISC)

Instruções permanentes para o Claude Code neste repositório. Leia este arquivo inteiro antes de qualquer tarefa.

## O que é este projeto

Implementação do TC "Gêmeo digital cardíaco para detecção de arritmias em tempo real a partir de sinais de
eletrocardiograma com aprendizado profundo" (Leonardo Nunes Alves, orient. Profa. Dra. Rejane Frozza, UNISC).

- Classificador híbrido **CNN 1D + BiLSTM + atenção** por batimento (classes AAMI N, S, V, F, Q).
- **Perfil estatístico do paciente** (histogramas de RR, largura do QRS e amplitude R nos 5 min iniciais) +
  **divergência KL** em janelas deslizantes → núcleo da personalização / inovação do trabalho.
- **Módulo de decisão** (Quadro 5 do TC): alerta só quando classificador **e** desvio KL concordam; alerta crítico
  para ≥3 V consecutivos independente do perfil; descarte do especialista realimenta o perfil (EMA, Eq. 10).
- Ingestão simulada (MIT-BIH a 360 Hz via WebSocket) → servidor Python → dashboard web próprio.
- Hipótese a testar: a personalização reduz falsos positivos em relação ao classificador isolado (DS2, interpaciente).

Documentos de referência (fonte da verdade, nesta ordem):
1. `docs/PLANO.md` — fases, tarefas, critérios de aceite. **Trabalhe fase a fase.**
2. `docs/ARQUITETURA.md` — módulos, contratos de dados, protocolo WebSocket.
3. `docs/DECISOES.md` — decisões técnicas já tomadas e as que estão em aberto.
4. `docs/RASTREABILIDADE.md` — RF/RNF do TC ↔ módulo ↔ teste.
5. `configs/default.yaml` — todos os parâmetros numéricos.

## Regras inegociáveis

1. **Interpaciente (RNF02).** Nenhum registro de DS2 pode ser usado em treino, validação, ajuste de `k`, janela
   ou qualquer hiperparâmetro. DS2 só é tocado por `scripts/evaluate_*.py`. Existe teste que garante isso — não remova.
2. **Mesmo código offline e online.** A avaliação offline (`evaluation/`) e o servidor (`server/`) usam a mesma
   classe `engine.StreamingPipeline`. Proibido reimplementar lógica de decisão/perfil fora de `profile/` e `decision/`.
3. **Filtro causal em todo lugar.** Treino e execução usam o mesmo filtro passa-faixa causal (`scipy.signal.sosfilt`
   com estado `zi`). Nunca use `filtfilt` em código que alimenta o modelo — gera divergência treino/execução.
4. **Sem números mágicos.** Todo parâmetro (frequências de corte, tamanhos de janela, `k`, `alpha`, bins, épocas…)
   vem de `configs/default.yaml` via `cardiac_twin.config`. Mudou um valor? Registre em `docs/DECISOES.md`.
5. **Realimentação desligada na avaliação quantitativa** (Seção 4.3 do TC). Ela só é exercitada nos testes funcionais.
6. **Os 5 min iniciais de cada registro DS2 são excluídos das métricas** nas duas configurações comparadas.
7. **Reprodutibilidade.** Toda execução de treino/avaliação grava em `artifacts/runs/<AAAAMMDD-HHMMSS>_<nome>/`:
   config efetiva, hash do git, seed, métricas (JSON) e figuras. Nunca sobrescreva runs anteriores.
8. **Apoio à decisão, não diagnóstico (RNF07).** Textos da interface nunca afirmam diagnóstico.
9. **Dados não versionados.** `data/` e `artifacts/` estão no `.gitignore`. Não faça commit de `.dat/.hea/.atr`, `.npz`,
   `.keras` ou banco SQLite.

## Convenções

- Python 3.11, layout `src/`, pacote `cardiac_twin`. Gerenciador: **uv** (`uv sync`, `uv run ...`).
- Código, identificadores e docstrings em **inglês**; documentação em `docs/` e mensagens da UI em **português**.
- Tipagem em todas as funções públicas; `ruff` para lint/format; `pytest` para testes.
- Funções puras em `signal/`, `profile/`, `decision/` (sem I/O, sem TensorFlow) → fáceis de testar.
- TensorFlow/Keras 3 apenas em `model/`. O resto do sistema conversa com o modelo via `model.inference.BeatClassifier`.
- Front-end em `dashboard/` (React + Vite + TypeScript). Não misture código Python lá.
- Commits pequenos, mensagem em português no imperativo: `feat(profile): calcula limiar L = μ + kσ`.
- Do not add redundant code comments or explain obvious code. Write clean code with zero inline comments unless explaining complex algorithmic edge cases. Always use ASD-STE100, or Simplified Technical English. Follow Zinsser's four principles of quality writing:

   1. Simplicity
   2. Brevity
   3. Clarity
   4. Humanity

## Comandos

```bash
uv sync --extra dev                      # Mac (CPU). Máquina de treino: + --extra gpu-cuda (D-10)
uv run python scripts/download_data.py   # baixa MIT-BIH (PhysioNet) para data/raw/mitdb
uv run python scripts/build_dataset.py   # gera data/processed/{ds1,ds2}.npz
uv run python scripts/train.py --loss weighted_ce   # ou --loss focal
uv run python scripts/tune_profile.py    # ajusta k e janela em DS1-val
uv run python scripts/evaluate_classifier.py --run <dir>
uv run python scripts/evaluate_system.py --run <dir>   # classificador isolado vs + perfil (DS2)
uv run python scripts/run_server.py      # FastAPI em :8000
uv run python scripts/run_simulator.py --record 100
cd dashboard && npm install && npm run dev   # :5173
uv run pytest -q
uv run ruff check . && uv run ruff format .
```

## Como trabalhar

- Comece cada sessão lendo `docs/PLANO.md` e localizando a primeira tarefa não marcada.
- Antes de codar uma fase, releia a seção correspondente de `docs/ARQUITETURA.md`.
- Ao concluir uma tarefa: rode os testes da fase, marque `[x]` no PLANO e anote desvios em `docs/DECISOES.md`.
- Ao fim de cada fase, gere o "Resumo para o TC" pedido no PLANO (texto curto em português, pronto para a Seção 4.5/4.6).
- Dúvida que muda resultado do TC (protocolo, métrica, regra de decisão)? **Pare e pergunte.** Dúvida de implementação
  interna? Decida, documente em `DECISOES.md` e siga.
