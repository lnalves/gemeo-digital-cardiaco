# scripts/

CLIs finas: só fazem parse de argumentos e chamam funções de `cardiac_twin`. Nenhuma lógica de negócio aqui.

| Script | Fase | Função |
|---|---|---|
| `download_data.py` | F0 | baixa a MIT-BIH para `data/raw/mitdb` |
| `check_env.py` | F0 | confere Python, TensorFlow, GPU (`--require-gpu`) e os 48 registros |
| `select_val_records.py` | F1 | sorteia o DS1-val (lê só DS1) e imprime o valor a fixar na config |
| `build_dataset.py` | F1 | gera `data/processed/{ds1,ds2}.npz` e `stats.json` |
| `train.py` | F2 | treina o classificador (`--loss weighted_ce|focal`) |
| `evaluate_classifier.py` | F2 | métricas por batimento em DS2 |
| `summarize_runs.py` | F2 | tabela dos runs de treino no DS1-val (média ± dp por experimento) |
| `tune_profile.py` | F3 | ajusta `k` e janela em DS1-val |
| `evaluate_system.py` | F4 | replay DS2, configuração A × B |
| `run_server.py` | F5 | sobe o FastAPI |
| `run_simulator.py` | F5 | transmite um registro via WebSocket |
