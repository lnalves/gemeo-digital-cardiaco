import { describe, expect, it } from 'vitest'
import type { AlertRow, EventRow, KlMessage } from '../types'
import { beatEdges, contextLabel, historyLines, klColumns } from './views'

describe('views', () => {
  it('splits the segment between context beats at the midpoints', () => {
    expect(beatEdges([100, 200, 400], 50, 450)).toEqual([
      [50, 150],
      [150, 300],
      [300, 450],
    ])
  })

  it('labels the context beats around i', () => {
    expect([0, 1, 2, 3, 4].map((i) => contextLabel(i, 3))).toEqual(['i−3', 'i−2', 'i−1', 'i', 'i+1'])
  })

  it('marks only the windows above L', () => {
    const kl = [
      { t: 310, value: 0.2, L: 0.5 },
      { t: 320, value: 0.7, L: 0.5 },
    ] as KlMessage[]
    expect(klColumns(kl)).toEqual([[310, 320], [0.2, 0.7], [0.5, 0.5], [null, 0.7]])
  })

  it('joins each event with its alert', () => {
    const events = [{ id: 1 }, { id: 2 }] as EventRow[]
    const alerts = [{ id: 'a', event_id: 2 }, { id: 'b', event_id: null }] as AlertRow[]
    expect(historyLines({ events, alerts }).map((l) => l.alert?.id)).toEqual([undefined, 'a'])
  })
})
