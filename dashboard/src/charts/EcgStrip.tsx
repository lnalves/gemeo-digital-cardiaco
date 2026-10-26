import { useEffect } from 'react'
import type uPlot from 'uplot'
import type { MonitorStore } from '../lib/monitor'
import { classColor, levelColor, token } from '../theme'
import { FONT, axis, timeAxis, useUPlot } from './uplot'

const HEIGHT = 280
const MARKER_BAND = 0.45
const LEVEL_BAND_S = 0.12

/** Rolling ECG (filtered signal) with one marker per classified beat: class dot and letter, level band. */
export function EcgStrip({ store }: { store: MonitorStore }) {
  const { ref, plot } = useUPlot(
    (width) => ({
      width,
      height: HEIGHT,
      legend: { show: false },
      cursor: { show: false },
      scales: {
        x: { time: false },
        y: {
          range: (_u, min, max) => {
            const span = max - min || 1
            return [min - 0.1 * span, max + MARKER_BAND * span]
          },
        },
      },
      series: [{}, { stroke: () => token('--ink'), width: 1.25, spanGaps: false, points: { show: false } }],
      axes: [timeAxis(), axis({ label: 'mV', size: 50 })],
      hooks: { draw: [(u) => drawBeats(u, store)] },
    }),
    [[], []],
    [store],
  )

  useEffect(() => {
    let frame = 0
    let drawn = -1
    const tick = () => {
      const u = plot.current
      if (u && store.signal && store.version !== drawn) {
        drawn = store.version
        const { t, y } = store.signal.view()
        u.setData([t, y])
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [store, plot])

  return (
    <div
      ref={ref}
      className="chart"
      role="img"
      aria-label="ECG filtrado dos últimos 10 segundos, com a classe sugerida de cada batimento"
    />
  )
}

function drawBeats(u: uPlot, store: MonitorStore) {
  const fs = store.signal?.fs
  if (!fs) return
  const ctx = u.ctx
  const { left, top, width, height } = u.bbox
  const px = devicePixelRatio
  const bandHalf = (u.valToPos(LEVEL_BAND_S, 'x', true) - u.valToPos(0, 'x', true)) / 2
  ctx.save()
  ctx.beginPath()
  ctx.rect(left, top, width, height)
  ctx.clip()
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  for (const b of store.beats) {
    const x = u.valToPos(b.sample / fs, 'x', true)
    if (b.level !== 'normal') {
      ctx.globalAlpha = 0.22
      ctx.fillStyle = levelColor(b.level)
      ctx.fillRect(x - bandHalf, top, 2 * bandHalf, height)
      ctx.globalAlpha = 1
    }
    ctx.fillStyle = classColor(b.cls)
    ctx.beginPath()
    ctx.arc(x, top + 10 * px, 4.5 * px, 0, 2 * Math.PI)
    ctx.fill()
    ctx.fillStyle = token(b.cls === 'N' ? '--muted' : '--ink')
    ctx.font = (b.cls === 'N' ? '' : '600 ') + FONT.replace('12px', `${12 * px}px`)
    ctx.fillText(b.cls, x, top + 18 * px)
  }
  ctx.restore()
}
