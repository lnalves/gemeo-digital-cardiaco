---
description: Executa uma fase do docs/PLANO.md (ex.: /fase F1)
---
Execute a fase $ARGUMENTS de `docs/PLANO.md`.

1. Releia `CLAUDE.md`, a seção da fase no PLANO e as seções de `docs/ARQUITETURA.md` citadas.
2. Liste as tarefas ainda não marcadas e os testes exigidos; comece pelos testes de funções puras.
3. Implemente tarefa a tarefa, rodando `uv run pytest -q` e `uv run ruff check .` após cada uma.
4. Marque `[x]` no PLANO ao concluir cada tarefa; registre desvios e parâmetros alterados em `docs/DECISOES.md`.
5. Ao final, escreva o "Resumo para o TC" da fase em `docs/resumos/<fase>.md` (português, números com referência ao run).
6. Se surgir dúvida que altere protocolo, métrica ou regra de decisão do TC, pare e pergunte.
