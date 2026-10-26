import { type DependencyList, useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { formatClock } from '../lib/format'
import { onThemeChange, token } from '../theme'

export const FONT = '12px system-ui, -apple-system, "Segoe UI", sans-serif'

export function axis(options: Partial<uPlot.Axis> = {}): uPlot.Axis {
  return {
    stroke: () => token('--ink-2'),
    grid: { stroke: () => token('--grid'), width: 1 },
    ticks: { stroke: () => token('--axis'), width: 1, size: 4 },
    font: FONT,
    labelFont: FONT,
    ...options,
  }
}

const WHOLE_SECONDS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]

/** X axis in record time (m:ss); `decimals` adds tenths for short spans, otherwise ticks fall on whole seconds. */
export const timeAxis = (decimals = false): uPlot.Axis =>
  axis({
    ...(decimals ? {} : { incrs: WHOLE_SECONDS }),
    values: (_, splits) =>
      splits.map((s) => (decimals && s % 1 ? `${formatClock(s)},${Math.round((s % 1) * 10)}` : formatClock(s))),
  })

/** Create a uPlot in the returned ref; it follows the container width and the OS theme. */
export function useUPlot(make: (width: number) => uPlot.Options, data: uPlot.AlignedData, deps: DependencyList) {
  const ref = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const u = new uPlot(make(el.clientWidth), data, el)
    plot.current = u
    const resize = new ResizeObserver(() => {
      if (el.clientWidth && el.clientWidth !== u.width) u.setSize({ width: el.clientWidth, height: u.height })
    })
    resize.observe(el)
    const off = onThemeChange(() => u.redraw(false))
    return () => {
      resize.disconnect()
      off()
      u.destroy()
      plot.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { ref, plot }
}
