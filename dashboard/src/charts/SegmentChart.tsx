import type uPlot from 'uplot'
import { formatNumber, formatPercent } from '../lib/format'
import { beatEdges, contextLabel } from '../lib/views'
import type { AlertMessage } from '../types'
import { levelColor, token } from '../theme'
import { FONT, axis, timeAxis, useUPlot } from './uplot'

const HEIGHT = 300
const LABEL_BAND = 0.35

/** Signal of the K context beats; band opacity follows the attention weight of each beat. */
export function SegmentChart({ alert, fs }: { alert: AlertMessage; fs: number }) {
  const t = alert.segment.map((_, i) => (alert.segment_t0 + i) / fs)
  const { ref } = useUPlot(
    (width) => ({
      width,
      height: HEIGHT,
      scales: {
        x: { time: false },
        y: {
          range: (_u, min, max) => {
            const span = max - min || 1
            return [min - 0.1 * span, max + LABEL_BAND * span]
          },
        },
      },
      series: [
        { label: 'Tempo', value: (_u, v) => (v == null ? '—' : `${formatNumber(v, 2)} s`) },
        { label: 'ECG (mV)', stroke: () => token('--ink'), width: 1.5, value: (_u, v) => formatNumber(v, 3) },
      ],
      axes: [timeAxis(true), axis({ label: 'mV', size: 50 })],
      hooks: { drawClear: [(u) => drawAttention(u, alert, fs)], draw: [(u) => drawLabels(u, alert, fs)] },
    }),
    [t, alert.segment],
    [alert, fs],
  )
  return (
    <div
      ref={ref}
      className="chart"
      role="img"
      aria-label="Trecho do ECG dos batimentos de contexto, sombreados pelo peso de atenção"
    />
  )
}

function bands(u: uPlot, alert: AlertMessage, fs: number) {
  const start = alert.segment_t0
  const edges = beatEdges(alert.context_samples, start, start + alert.segment.length - 1)
  return edges.map(([a, b]) => [u.valToPos(a / fs, 'x', true), u.valToPos(b / fs, 'x', true)])
}

function drawAttention(u: uPlot, alert: AlertMessage, fs: number) {
  const ctx = u.ctx
  const { top, height } = u.bbox
  const max = Math.max(...alert.attention)
  ctx.save()
  ctx.fillStyle = token('--attention-fill')
  bands(u, alert, fs).forEach(([x0, x1], i) => {
    ctx.globalAlpha = 0.05 + 0.3 * (alert.attention[i] / max)
    ctx.fillRect(x0, top, x1 - x0, height)
  })
  ctx.restore()
}

function drawLabels(u: uPlot, alert: AlertMessage, fs: number) {
  const ctx = u.ctx
  const { top, height } = u.bbox
  const px = devicePixelRatio
  const center = alert.context_samples.indexOf(alert.sample)
  ctx.save()
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  bands(u, alert, fs).forEach(([x0, x1], i) => {
    ctx.fillStyle = token(i === center ? '--ink' : '--ink-2')
    ctx.font = (i === center ? '600 ' : '') + FONT.replace('12px', `${12 * px}px`)
    ctx.fillText(contextLabel(i, center), (x0 + x1) / 2, top + 6 * px)
    ctx.fillText(formatPercent(alert.attention[i], 1), (x0 + x1) / 2, top + 22 * px)
  })
  const x = u.valToPos(alert.sample / fs, 'x', true)
  ctx.strokeStyle = levelColor(alert.level)
  ctx.lineWidth = 2 * px
  ctx.setLineDash([5 * px, 4 * px])
  ctx.beginPath()
  ctx.moveTo(x, top + 40 * px)
  ctx.lineTo(x, top + height)
  ctx.stroke()
  ctx.restore()
}
