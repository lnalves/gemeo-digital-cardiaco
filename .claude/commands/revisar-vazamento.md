---
description: Audita o código em busca de vazamento DS1/DS2 e divergência treino/execução
---
Audite o repositório procurando:
- qualquer uso de registros de DS2 fora de `scripts/evaluate_*.py` e `evaluation/`;
- ajuste de hiperparâmetros, `k`, janela ou escolha de perda com métricas de DS2;
- `filtfilt` ou qualquer filtro não causal em código que alimenta o modelo ou o perfil;
- normalização com estatísticas globais calculadas sobre DS2;
- realimentação do perfil ligada durante a avaliação quantitativa;
- os 5 min de calibração entrando nas métricas.
Reporte arquivo:linha e a correção sugerida. Não altere código sem confirmação.
