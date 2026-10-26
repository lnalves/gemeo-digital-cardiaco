// Screenshots of the 4 screens for Section 4.5 of the TC (docs/figuras/f6_*.png).
// Needs the server (scripts/run_server.py) and the dashboard (npm run dev) running.
// Usage: node scripts/screenshots.mjs [--record 213] [--url http://localhost:5173]
import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { chromium } from 'playwright-core'

const { values } = parseArgs({
  options: {
    record: { type: 'string', default: '213' },
    url: { type: 'string', default: 'http://localhost:5173' },
    channel: { type: 'string', default: 'msedge' },
    out: { type: 'string', default: fileURLToPath(new URL('../../docs/figuras/', import.meta.url)) },
    'kl-seconds': { type: 'string', default: '30' },
    'calibration-seconds': { type: 'string', default: '15' },
  },
})
mkdirSync(values.out, { recursive: true })
const out = (name) => join(values.out, name)
const MINUTES = 60_000

const browser = await chromium.launch({ channel: values.channel })
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
  colorScheme: 'light',
  locale: 'pt-BR',
})
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))

await page.goto(values.url)
await page.getByRole('radio', { name: values.record, exact: true }).check()
await page.getByLabel(/Realimentação pelo especialista/).check()
await page.getByRole('button', { name: '5×' }).click()
await page.screenshot({ path: out('f6_selecao.png') })
console.log('Seleção')

await page.getByRole('button', { name: /Iniciar monitoramento/ }).click()
await page.waitForURL(/#\/sessoes\//)
const sessionId = page.url().split('/').pop()
await page.getByText(/Calibrando o perfil · faltam/).waitFor()
await page.waitForTimeout(Number(values['calibration-seconds']) * 1000)
await page.screenshot({ path: out('f6_monitor_calibracao.png'), fullPage: true })
await page.getByText('Monitorando', { exact: true }).waitFor({ timeout: 5 * MINUTES })
await page.waitForTimeout(Number(values['kl-seconds']) * 1000)
await page.locator('.alert-list li').first().waitFor({ timeout: 5 * MINUTES })
await page.screenshot({ path: out('f6_monitor.png'), fullPage: true })
console.log(`Monitor (sessão ${sessionId})`)

await page.getByRole('link', { name: 'Revisar' }).first().click()
await page.getByRole('button', { name: 'Descartar' }).waitFor()
await page.waitForTimeout(500)
await page.screenshot({ path: out('f6_alerta.png'), fullPage: true })
await page.getByRole('button', { name: 'Descartar' }).click()
await page.getByRole('status').filter({ hasText: 'Descartado' }).waitFor()
await page.screenshot({ path: out('f6_alerta_descartado.png'), fullPage: true })
console.log('Detalhe do alerta')

await page.request.post(`${values.url}/api/sessions/${sessionId}/stop`)
await page.goto(`${values.url}/#/historico/${sessionId}`)
await page.getByRole('heading', { name: 'Versões do perfil' }).waitFor()
await page.waitForTimeout(500)
await page.screenshot({ path: out('f6_historico.png') })
console.log('Histórico')

await browser.close()
if (errors.length) {
  console.error('Erros no navegador:', errors)
  process.exit(1)
}
