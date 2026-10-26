// F6 acceptance: one whole record at 5× in the Monitor, measured in the browser.
// Needs the server and the dashboard running. Prints a JSON report; exit 1 if the page froze or lost the stream.
// Usage: node scripts/soak.mjs [--record 213] [--speed 5] [--no-profile] [--url http://localhost:5173]
import { parseArgs } from 'node:util'
import { chromium } from 'playwright-core'

const { values } = parseArgs({
  options: {
    record: { type: 'string', default: '213' },
    speed: { type: 'string', default: '5' },
    'no-profile': { type: 'boolean', default: false },
    url: { type: 'string', default: 'http://localhost:5173' },
    channel: { type: 'string', default: 'msedge' },
    'freeze-ms': { type: 'string', default: '500' },
  },
})
const FREEZE_MS = Number(values['freeze-ms'])

// Runs in the page before the app: time from each /ws/monitor message to the next animation frame.
function instrument() {
  const probe = { frames: [], delays: [], messages: 0, pending: [] }
  window.__probe = probe
  let last = performance.now()
  const frame = (now) => {
    probe.frames.push(now - last)
    last = now
    for (const t of probe.pending) probe.delays.push(now - t)
    probe.pending = []
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
  const Native = window.WebSocket
  window.WebSocket = class extends Native {
    constructor(...args) {
      super(...args)
      if (String(args[0]).includes('/ws/monitor/'))
        this.addEventListener('message', () => {
          probe.messages++
          probe.pending.push(performance.now())
        })
    }
  }
}

const percentile = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b)
  return s.length ? s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))] : null
}

const browser = await chromium.launch({ channel: values.channel })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
await page.addInitScript(instrument)

const r = await page.request.post(`${values.url}/api/sessions`, {
  data: { record_id: values.record, use_profile: !values['no-profile'], speed: Number(values.speed) },
})
const { session_id: sessionId } = await r.json()
const started = Date.now()
await page.goto(`${values.url}/#/sessoes/${sessionId}`)
await page.getByText('Encerrado', { exact: true }).waitFor({ timeout: 60 * 60_000 })
await page.waitForTimeout(1000)

const probe = await page.evaluate(() => window.__probe)
const shown = await page.evaluate(() => ({
  text: document.querySelector('main')?.innerText ?? '',
  heapMb: performance.memory ? performance.memory.usedJSHeapSize / 2 ** 20 : null,
}))
const history = await (await page.request.get(`${values.url}/api/sessions/${sessionId}/history`)).json()
await browser.close()

const frames = probe.frames.slice(1)
const reconnects = Number(/(\d+) reconexões/.exec(shown.text)?.[1] ?? 0)
const report = {
  session: sessionId,
  record: values.record,
  config: values['no-profile'] ? 'A' : 'B',
  speed: Number(values.speed),
  wall_seconds: (Date.now() - started) / 1000,
  messages: probe.messages,
  alerts: history.alerts.length,
  frames: frames.length,
  frame_ms: { p50: percentile(frames, 50), p95: percentile(frames, 95), max: Math.max(...frames) },
  frames_over_freeze: frames.filter((f) => f > FREEZE_MS).length,
  message_to_frame_ms: { p50: percentile(probe.delays, 50), p95: percentile(probe.delays, 95), max: Math.max(...probe.delays) },
  reconnects,
  heap_mb: shown.heapMb,
  errors,
}
console.log(JSON.stringify(report, null, 2))
process.exit(report.frames_over_freeze || reconnects || errors.length ? 1 : 0)
