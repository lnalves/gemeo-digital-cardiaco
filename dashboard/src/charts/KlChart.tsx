import { useEffect } from 'react'
import uPlot from 'uplot'
import { formatNumber } from '../lib/format'
import { klColumns } from '../lib/views'
import type { KlMessage } from '../types'
import { token } from '../theme'
import { axis, timeAxis, useUPlot } from './uplot'

const HEIGHT = 240

/** KL divergence of each window against the profile, with the threshold L (one y scale).
 *  `kl` may grow in place: the chart updates when its length changes. */
export function KlChart({ kl }: { kl: KlMessage[] }) {
  const { ref, plot } = useUPlot(
    (width) => ({
      width,
      height: HEIGHT,
      scales: { x: { time: false }, y: { range: (_u, _min, max) => [0, Math.max(max, 0.1) * 1.1] } },
      cursor: { points: { size: 8 } },
      series: [
        { label: 'Tempo', value: (_u, v) => (v == null ? '—' : `${formatNumber(v / 60, 1)} min`) },
        {
          label: 'KL da janela',
          stroke: () => token('--ink-2'),
          width: 2,
          points: { show: false },
          value: (_u, v) => formatNumber(v),
        },
        {
          label: 'Limiar L',
          stroke: () => token('--level-critical'),
          width: 1.5,
          dash: [6, 4],
          paths: uPlot.paths.stepped!({ align: 1 }),
          points: { show: false },
          value: (_u, v) => formatNumber(v),
        },
        {
          label: 'KL acima de L',
          stroke: () => token('--level-alert'),
          fill: () => token('--level-alert'),
          width: 0,
          paths: () => null,
          points: { show: true, size: 8, width: 2, stroke: () => token('--surface'), fill: () => token('--level-alert') },
          value: (_u, v) => (v == null ? '—' : 'sim'),
        },
      ],
      axes: [timeAxis(), axis({ label: 'KL', size: 50 })],
    }),
    klColumns(kl),
    [],
  )

  const n = kl.length
  useEffect(() => {
    plot.current?.setData(klColumns(kl))
  }, [kl, n, plot])

  return (
    <div
      ref={ref}
      className="chart"
      role="img"
      aria-label="Divergência KL de cada janela em relação ao perfil do paciente e o limiar L"
    />
  )
}
