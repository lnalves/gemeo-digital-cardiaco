import type { Decision, DecisionResult, History, RecordRow, SessionRow, StateMessage } from './types'

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response
  try {
    r = await fetch(path, init)
  } catch {
    throw new ApiError(0, 'Servidor indisponível. Confira se `scripts/run_server.py` está rodando.')
  }
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new ApiError(r.status, typeof body?.detail === 'string' ? body.detail : `Erro ${r.status}`)
  }
  return r.json() as Promise<T>
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

export const api = {
  records: () => request<RecordRow[]>('/api/records'),
  sessions: () => request<SessionRow[]>('/api/sessions'),
  session: (id: string) => request<SessionRow>(`/api/sessions/${id}`),
  history: (id: string) => request<History>(`/api/sessions/${id}/history`),
  createSession: (body: { record_id: string; use_profile: boolean; feedback_enabled: boolean; speed: number }) =>
    post<{ session_id: string }>('/api/sessions', body),
  control: (id: string, action: 'pause' | 'resume' | 'stop') =>
    post<StateMessage>(`/api/sessions/${id}/${action}`),
  decide: (alertId: string, decision: Decision) =>
    post<DecisionResult>(`/api/alerts/${alertId}/decision`, { decision }),
}
