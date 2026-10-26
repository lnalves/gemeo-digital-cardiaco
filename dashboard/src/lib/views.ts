import type { AlertRow, EventRow, History, KlMessage } from '../types'

/** Edges of each context beat: midpoints between neighbouring R peaks, segment ends outside. */
export function beatEdges(peaks: number[], start: number, end: number): [number, number][] {
  return peaks.map((p, i) => [
    i === 0 ? start : (peaks[i - 1] + p) / 2,
    i === peaks.length - 1 ? end : (p + peaks[i + 1]) / 2,
  ])
}

/** Position of a context beat relative to the classified beat i: i−3 … i … i+1. */
export function contextLabel(i: number, center: number): string {
  return i === center ? 'i' : `i${i < center ? '−' : '+'}${Math.abs(i - center)}`
}

/** uPlot columns of the KL chart: time, KL, L, KL only where it is above L. */
export function klColumns(kl: KlMessage[]): [number[], number[], number[], (number | null)[]] {
  return [
    kl.map((k) => k.t),
    kl.map((k) => k.value),
    kl.map((k) => k.L),
    kl.map((k) => (k.value > k.L ? k.value : null)),
  ]
}

export interface HistoryLine {
  event: EventRow
  alert: AlertRow | undefined
}

/** One line per saved event, with the alert it raised, if any. */
export function historyLines(h: Pick<History, 'events' | 'alerts'>): HistoryLine[] {
  const byEvent = new Map(h.alerts.filter((a) => a.event_id != null).map((a) => [a.event_id, a]))
  return h.events.map((event) => ({ event, alert: byEvent.get(event.id) }))
}
