# Rastreabilidade — requisitos do TC ↔ código ↔ testes

Atualize a coluna "Teste" à medida que os testes forem escritos (F7 exige a tabela completa).

| Req. | Descrição (Tabelas 5 e 6 do TC) | Módulo | Teste |
|---|---|---|---|
| RF01 | Selecionar registro (paciente) | `server/app.py` `GET /api/records`, dashboard Seleção | `tests/server/test_api.py` |
| RF02 | Iniciar, pausar, encerrar monitoramento | `server/app.py` sessões, `engine.pipeline.state` | `tests/server/test_api.py` |
| RF03 | Receber sinal continuamente | `acquisition/simulator.py`, `/ws/ingest` | `tests/server/test_ingest.py` |
| RF04 | Construir perfil na calibração | `profile/profile.py` `build` | `tests/profile/test_profile.py` |
| RF05 | Classificar batimentos em N, S, V, F, Q | `model/inference.py` | `tests/model/test_inference.py` |
| RF06 | KL entre janela e perfil | `profile/kl.py`, `profile.score` | `tests/profile/test_kl.py` |
| RF07 | Combinar classe + desvio → nível | `decision/rules.py` | `tests/decision/test_rules.py` |
| RF08 | ECG em tempo real com classes | dashboard Monitor, WS `signal`/`beat` | teste manual + screenshot |
| RF09 | FC, indicador de desvio e limiar | WS `hr`/`kl`, dashboard | teste manual + screenshot |
| RF10 | Alertas visuais (alerta / crítico) | WS `alert`, dashboard | teste manual + screenshot |
| RF11 | Confirmar / descartar alerta | `POST /api/alerts/{id}/decision` | `tests/server/test_api.py` |
| RF12 | Atualizar perfil pelos descartes | `profile.update`, `pipeline.apply_decision` | `tests/functional/test_feedback.py` |
| RF13 | Histórico de alertas e decisões | `storage/db.py`, `GET .../history` | `tests/storage/test_db.py` |
| RNF01 | Latência de processamento < 500 ms (ver D-01) | `engine/pipeline.py`, `evaluation/replay.py` | `tests/engine/test_latency.py` |
| RNF02 | Interpaciente DS1 × DS2 | `data/splits.py` | `tests/data/test_splits.py` |
| RNF03 | Registrar probabilidade e atenção | `beats` no SQLite, WS `beat` | `tests/storage/test_db.py` |
| RNF04 | Interface via navegador | `dashboard/` | manual |
| RNF05 | Trocar fonte de sinal sem alterar módulos | `/ws/ingest` desacoplado | `tests/server/test_ingest.py` |
| RNF06 | Apenas dados públicos anonimizados | MIT-BIH (PhysioNet) | — |
| RNF07 | Apoio à decisão, sem diagnóstico | textos do dashboard | revisão manual |
