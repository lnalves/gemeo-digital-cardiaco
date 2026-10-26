// Contract of docs/ARQUITETURA.md §9 (REST and /ws/monitor).

export type ClassName = 'N' | 'S' | 'V' | 'F' | 'Q'
export type Level = 'normal' | 'attention' | 'alert' | 'critical'
export type PipelineState = 'calibrating' | 'monitoring' | 'paused' | 'finished'
export type SessionStatus = 'running' | 'paused' | 'stopped'
export type Decision = 'confirm' | 'discard'
export type AlertStatus = 'pending' | 'confirmed' | 'discarded'

export interface StateMessage {
  type: 'state'
  state: PipelineState
  status: SessionStatus
  record_id: string
  calibration_seconds: number
  t: number
}

export interface SignalMessage {
  type: 'signal'
  t0: number
  fs: number
  samples: number[]
}

export interface BeatMessage {
  type: 'beat'
  sample: number
  t: number
  cls: ClassName
  prob: number[]
  attention: number[]
  level: Level
  kl: number | null
  L: number | null
}

export interface KlMessage {
  type: 'kl'
  t: number
  value: number
  L: number
  n_beats: number
  level: Level
}

export interface EventMessage {
  type: 'event'
  id: number
  kind: 'beat' | 'window'
  level: Level
  sample: number
  t: number
  cls: ClassName | null
  kl: number | null
  L: number | null
}

export interface AlertMessage {
  type: 'alert'
  id: string
  episode_id: number
  level: Level
  sample: number
  t: number
  cls: ClassName
  prob: number[]
  kl: number | null
  L: number | null
  attention: number[]
  context_samples: number[]
  segment_t0: number
  segment: number[]
}

export interface DecisionMessage {
  type: 'decision'
  id: string
  decision: Decision
}

export interface ProfileUpdatedMessage {
  type: 'profile_updated'
  version: number
}

export interface HrMessage {
  type: 'hr'
  bpm: number
}

export type MonitorMessage =
  | StateMessage
  | SignalMessage
  | BeatMessage
  | KlMessage
  | EventMessage
  | AlertMessage
  | DecisionMessage
  | ProfileUpdatedMessage
  | HrMessage

export interface RecordRow {
  record_id: string
  split: string
  duration_s: number
  fs: number
  training: boolean
}

export interface SessionRow {
  id: string
  record_id: string
  use_profile: number
  feedback_enabled: number
  started_at: string
  ended_at: string | null
  model_run: string | null
  status: SessionStatus
  state: StateMessage | null
}

export interface EventRow {
  id: number
  session_id: string
  kind: 'beat' | 'window'
  sample: number
  level: Level
  cls: ClassName | null
  kl: number | null
  L: number | null
  recv_ts: number | null
  emit_ts: number | null
}

export interface AlertRow {
  id: string
  session_id: string
  event_id: number | null
  episode_id: number
  status: AlertStatus
  decided_at: string | null
  payload: AlertMessage
}

export interface ProfileRow {
  id: number
  version: number
  mu: number
  sigma: number
  k: number
  L: number
  created_at: string
}

export interface History {
  session: SessionRow
  events: EventRow[]
  alerts: AlertRow[]
  profiles: ProfileRow[]
}

export interface DecisionResult extends AlertRow {
  profile_version: number | null
}
