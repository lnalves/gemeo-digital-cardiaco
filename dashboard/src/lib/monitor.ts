import type {
  AlertMessage,
  AlertStatus,
  BeatMessage,
  ClassName,
  History,
  KlMessage,
  Level,
  MonitorMessage,
  StateMessage,
} from '../types'
import { CLASSES } from './format'
import { SignalWindow } from './signal'

export interface TrackedAlert {
  alert: AlertMessage
  status: AlertStatus
}

export interface Episode {
  episodeId: number
  level: Level
  pending: AlertMessage[]
  lastT: number
}

const LEVEL_RANK: Record<Level, number> = { normal: 0, attention: 1, alert: 2, critical: 3 }

/** Dashboard state of one session, built from /ws/monitor messages. Mutable; `version` counts changes. */
export class MonitorStore {
  version = 0
  state: StateMessage | null = null
  hr: number | null = null
  signal: SignalWindow | null = null
  beats: BeatMessage[] = []
  kl: KlMessage[] = []
  counts: Record<ClassName, number> = { N: 0, S: 0, V: 0, F: 0, Q: 0 }
  alerts = new Map<string, TrackedAlert>()
  profileVersion: number | null = null
  private readonly windowSeconds: number

  constructor(windowSeconds: number) {
    this.windowSeconds = windowSeconds
  }

  apply(m: MonitorMessage): void {
    switch (m.type) {
      case 'state':
        this.state = m
        break
      case 'signal':
        this.signal ??= new SignalWindow(m.fs, this.windowSeconds)
        this.signal.push(m.t0, m.samples)
        this.trimBeats()
        break
      case 'beat':
        this.beats.push(m)
        this.counts[m.cls]++
        break
      case 'kl':
        this.kl.push(m)
        break
      case 'alert':
        if (!this.alerts.has(m.id)) this.alerts.set(m.id, { alert: m, status: 'pending' })
        break
      case 'decision': {
        const a = this.alerts.get(m.id)
        if (a) a.status = m.decision === 'confirm' ? 'confirmed' : 'discarded'
        break
      }
      case 'profile_updated':
        this.profileVersion = m.version
        break
      case 'hr':
        this.hr = m.bpm
        break
      case 'event':
        return
    }
    this.version++
  }

  /** Alerts and profile version saved on the server: fills what a (re)connection missed. */
  applyHistory(h: History): void {
    for (const row of h.alerts) this.alerts.set(row.id, { alert: row.payload, status: row.status })
    const last = h.profiles.at(-1)
    if (last) this.profileVersion = Math.max(this.profileVersion ?? 0, last.version)
    this.version++
  }

  /** Record time (s) of the last sample received; `state.t` only changes with the state. */
  get t(): number {
    return this.signal ? this.signal.endSample / this.signal.fs : (this.state?.t ?? 0)
  }

  get pendingCount(): number {
    let n = 0
    for (const a of this.alerts.values()) n += a.status === 'pending' ? 1 : 0
    return n
  }

  /** Pending alerts grouped by episode, most recent first. */
  episodes(): Episode[] {
    const byId = new Map<number, Episode>()
    for (const { alert, status } of this.alerts.values()) {
      if (status !== 'pending') continue
      let e = byId.get(alert.episode_id)
      if (!e) {
        e = { episodeId: alert.episode_id, level: alert.level, pending: [], lastT: alert.t }
        byId.set(alert.episode_id, e)
      }
      e.pending.push(alert)
      if (LEVEL_RANK[alert.level] > LEVEL_RANK[e.level]) e.level = alert.level
      e.lastT = Math.max(e.lastT, alert.t)
    }
    for (const e of byId.values()) e.pending.sort((a, b) => a.t - b.t)
    return [...byId.values()].sort((a, b) => b.lastT - a.lastT)
  }

  classCounts(): [ClassName, number][] {
    return CLASSES.map((c) => [c, this.counts[c]])
  }

  private trimBeats(): void {
    if (!this.signal) return
    const first = this.signal.endSample - this.signal.size
    let i = 0
    while (i < this.beats.length && this.beats[i].sample < first) i++
    if (i) this.beats.splice(0, i)
  }
}
