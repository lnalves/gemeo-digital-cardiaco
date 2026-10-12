# scripts/

CLIs finas: só fazem parse de argumentos e chamam funções de `cardiac_twin`. Nenhuma lógica de negócio aqui.

| Script | Fase | Função |
|---|---|---|
| `download_data.py` | F0 | baixa a MIT-BIH para `data/raw/mitdb` |
| `build_dataset.py` | F1 | gera `data/processed/{ds1,ds2}.npz` |
| `train.py` | F2 | treina o classificador (`--loss weighted_ce|focal`) |
| `evaluate_classifier.py` | F2 | métricas por batimento em DS2 |
| `tune_profile.py` | F3 | ajusta `k` e janela em DS1-val |
| `evaluate_system.py` | F4 | replay DS2, configuração A × B |
| `run_server.py` | F5 | sobe o FastAPI |
| `run_simulator.py` | F5 | transmite um registro via WebSocket |
