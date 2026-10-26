# dashboard/

Front-end próprio: React 19 + Vite 8 + TypeScript + uPlot. Contrato com o servidor em `docs/ARQUITETURA.md` §9–10;
decisões em `docs/DECISOES.md` (D-29). Textos da interface em português; o aviso "ferramenta de apoio à decisão"
fica sempre visível no topo (RNF07).

## Rodar

```bash
uv run python scripts/run_server.py      # na raiz do repositório, :8000
cd dashboard && npm install && npm run dev   # http://localhost:5173
```

O Vite repassa `/api` e `/ws` para o servidor (`server.host` e `server.port` de `configs/default.yaml`; a variável
`CT_SERVER_PORT` troca a porta). A velocidade padrão, a duração da calibração e o número de V seguidos do alerta
crítico também vêm do YAML.

## Telas

| Rota | Tela |
|---|---|
| `#/` | Seleção: registro, configuração A/B, realimentação, velocidade 1× ou 5× |
| `#/sessoes/<id>` | Monitor: ECG dos últimos 10 s com classe por batimento, FC, KL × L, alertas pendentes por episódio |
| `#/sessoes/<id>/alertas/<id>` | Detalhe do alerta: trecho do sinal com a atenção, probabilidades, KL × L, Confirmar/Descartar |
| `#/historico/<id>` | Histórico: sessões, eventos, alertas, decisões e versões do perfil, com filtros |

## Verificar

```bash
npm test                  # vitest: funções puras de src/lib
npm run build             # tsc + build de produção
npm run lint              # oxlint
node scripts/screenshots.mjs   # capturas das telas em docs/figuras/f6_*.png (servidor e Vite rodando)
node scripts/soak.mjs --no-profile   # aceite: registro inteiro a 5× no navegador, relatório em JSON
```

Os dois scripts usam um navegador já instalado (`playwright-core`, `--channel msedge` por padrão; no Mac, `--channel chrome`).
