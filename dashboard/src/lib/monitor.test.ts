import { describe, expect, it } from 'vitest'
import type { AlertMessage, BeatMessage, History, Level } from '../types'
import { MonitorStore } from './monitor'

const FS = 10

const beat = (sample: number, cls: BeatMessage['cls'] = 'N'): BeatMessage => ({
  type: 'beat',
  sample,
  t: sample / FS,
  cls,
  prob: [1, 0, 0, 0, 0],
  attention: [0.2, 0.2, 0.2, 0.2, 0.2],
  level: 'normal',
  kl: null,
  L: null,
})

const alert = (sample: number, episode: number, level: Level = 'alert'): AlertMessage => ({
  type: 'alert',
  id: `s-${sample}`,
  episode_id: episode,
  level,
  sample,
  t: sample / FS,
  cls: 'V',
  prob: [0, 0, 1, 0, 0],
  kl: 1,
  L: 0.5,
  attention: [0.2, 0.2, 0.2, 0.2, 0.2],
  context_samples: [sample],
  segment_t0: sample,
  segment: [0],
})

const signal = (t0: number, n: number) => ({ type: 'signal' as const, t0, fs: FS, samples: Array(n).fill(0) })

describe('MonitorStore', () => {
  it('keeps only the beats inside the ECG window and counts every class', () => {
    const s = new MonitorStore(2)
    s.apply(beat(5))
    s.apply(beat(15, 'V'))
    s.apply(signal(0, 30))
    expect(s.beats.map((b) => b.sample)).toEqual([15])
    expect(s.counts).toMatchObject({ N: 1, V: 1 })
  })

  it('groups pending alerts by episode, most recent first, with the highest level', () => {
    const s = new MonitorStore(2)
    s.apply(alert(10, 1))
    s.apply(alert(20, 1, 'critical'))
    s.apply(alert(500, 2))
    s.apply({ type: 'decision', id: 's-500', decision: 'discard' })
    s.apply(alert(600, 3))
    const episodes = s.episodes()
    expect(episodes.map((e) => e.episodeId)).toEqual([3, 1])
    expect(episodes[1].level).toBe('critical')
    expect(episodes[1].pending.map((a) => a.id)).toEqual(['s-10', 's-20'])
    expect(s.pendingCount).toBe(3)
    expect(s.alerts.get('s-500')?.status).toBe('discarded')
  })

  it('fills from the history what a reconnection missed', () => {
    const s = new MonitorStore(2)
    s.apply(alert(10, 1))
    const history = {
      alerts: [
        { id: 's-10', status: 'confirmed', payload: alert(10, 1) },
        { id: 's-30', status: 'pending', payload: alert(30, 1) },
      ],
      profiles: [{ version: 1 }, { version: 2 }],
    } as unknown as History
    s.applyHistory(history)
    expect(s.alerts.get('s-10')?.status).toBe('confirmed')
    expect(s.pendingCount).toBe(1)
    expect(s.profileVersion).toBe(2)
  })

  it('takes the record time from the last signal chunk', () => {
    const s = new MonitorStore(2)
    s.apply({ type: 'state', state: 'calibrating', status: 'running', record_id: '1', calibration_seconds: 300, t: 0 })
    expect(s.t).toBe(0)
    s.apply(signal(0, 25))
    expect(s.t).toBe(2.5)
  })

  it('counts each change', () => {
    const s = new MonitorStore(2)
    s.apply({ type: 'hr', bpm: 72 })
    s.apply({ type: 'profile_updated', version: 2 })
    expect([s.version, s.hr, s.profileVersion]).toEqual([2, 72, 2])
  })
})
