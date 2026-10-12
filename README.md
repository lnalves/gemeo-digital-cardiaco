# Gêmeo digital cardíaco

Implementação do Trabalho de Conclusão "**Gêmeo digital cardíaco para detecção de arritmias em tempo real a partir
de sinais de eletrocardiograma com aprendizado profundo**" — Leonardo Nunes Alves, Ciência da Computação, UNISC.
Orientação: Profa. Dra. Rejane Frozza.

O sistema classifica cada batimento do ECG (classes AAMI N, S, V, F, Q) com uma rede híbrida CNN 1D + BiLSTM +
atenção e combina essa saída a um **perfil estatístico individual do paciente** (divergência de Kullback-Leibler
sobre RR, largura do QRS e amplitude R). Um alerta só é emitido quando classificador e desvio do perfil concordam —
a hipótese é que isso reduz falsos positivos em relação a um classificador genérico. Taquicardia ventricular
(≥ 3 V consecutivos) gera alerta crítico imediato. O especialista confirma ou descarta alertas, e os descartes
realimentam o perfil.

> Ferramenta acadêmica de apoio à decisão. Não emite diagnóstico.

## Início rápido

```bash
uv sync --extra dev                    # máquina de treino: + --extra gpu-cuda
uv run python scripts/download_data.py
uv run python scripts/build_dataset.py
uv run python scripts/train.py
uv run python scripts/evaluate_system.py --run artifacts/runs/<run>
uv run python scripts/run_server.py & (cd dashboard && npm run dev)
```

## Documentação

| Arquivo | Conteúdo |
|---|---|
| `CLAUDE.md` | regras para o Claude Code |
| `docs/PLANO.md` | fases, tarefas, critérios de aceite, cronograma até 28/10/2026 |
| `docs/ARQUITETURA.md` | módulos, modelo, perfil, regras, API/WebSocket, avaliação |
| `docs/DECISOES.md` | decisões técnicas e pontos em aberto |
| `docs/RASTREABILIDADE.md` | requisitos do TC ↔ código ↔ testes |
| `configs/default.yaml` | todos os parâmetros |

## Dados

MIT-BIH Arrhythmia Database (Moody; Mark, 2001), PhysioNet, divisão interpaciente DS1/DS2 (De Chazal; O'Dwyer;
Reilly, 2004). Os dados não são versionados.
