import { useMemo, useState } from 'react'
import { api } from '../api'
import { ClassChip, LevelBadge } from '../components/Badges'
import { HISTORY_PAGE_SIZE } from '../config'
import { useAsync } from '../hooks'
import { STATUS_NAMES, configLabel, formatClock, formatDateTime, formatNumber } from '../lib/format'
import { href } from '../lib/route'
import { historyLines } from '../lib/views'
import type { AlertStatus, Level } from '../types'

export function History({ sessionId }: { sessionId: string | null }) {
  const sessions = useAsync(api.sessions, [])
  return (
    <>
      <h1>Histórico</h1>
      <section className="panel">
        <header>
          <h2>Sessões gravadas</h2>
        </header>
        {sessions.error && <p className="error">{sessions.error}</p>}
        <div className="table-wrap" style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Sessão</th>
                <th>Registro</th>
                <th>Configuração</th>
                <th>Realimentação</th>
                <th>Início</th>
                <th>Fim</th>
                <th>Modelo</th>
              </tr>
            </thead>
            <tbody>
              {sessions.data?.map((s) => (
                <tr
                  key={s.id}
                  className="clickable"
                  aria-selected={s.id === sessionId}
                  onClick={() => (window.location.hash = href({ page: 'history', sessionId: s.id }))}
                >
                  <td>
                    <a href={href({ page: 'history', sessionId: s.id })}>{s.id}</a>
                  </td>
                  <td>{s.record_id}</td>
                  <td>{configLabel(s.use_profile)}</td>
                  <td>{s.feedback_enabled ? 'ligada' : 'desligada'}</td>
                  <td>{formatDateTime(s.started_at)}</td>
                  <td>{s.ended_at ? formatDateTime(s.ended_at) : 'em andamento'}</td>
                  <td className="small secondary">{s.model_run ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      {sessionId ? (
        <SessionHistory key={sessionId} sessionId={sessionId} />
      ) : (
        <p className="muted">Escolha uma sessão para ver eventos, alertas e decisões.</p>
      )}
    </>
  )
}

type Kind = 'all' | 'beat' | 'window'
type LevelFilter = 'all' | Exclude<Level, 'normal'>
type StatusFilter = 'all' | AlertStatus | 'none'

function SessionHistory({ sessionId }: { sessionId: string }) {
  const history = useAsync(() => api.history(sessionId), [sessionId])
  const records = useAsync(api.records, [])
  const [kind, setKind] = useState<Kind>('all')
  const [level, setLevel] = useState<LevelFilter>('all')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [shown, setShown] = useState(HISTORY_PAGE_SIZE)
  const lines = useMemo(() => (history.data ? historyLines(history.data) : []), [history.data])

  if (history.error) return <p className="error">{history.error}</p>
  if (!history.data || !records.data) return <p className="muted">Carregando…</p>
  const h = history.data
  const fs = records.data.find((r) => r.record_id === h.session.record_id)?.fs ?? NaN
  const filtered = lines.filter(
    ({ event, alert }) =>
      (kind === 'all' || event.kind === kind) &&
      (level === 'all' || event.level === level) &&
      (status === 'all' || (status === 'none' ? !alert : alert?.status === status)),
  )
  const count = (s: AlertStatus) => h.alerts.filter((a) => a.status === s).length

  return (
    <>
      <section className="panel row" style={{ gap: '1.5rem' }}>
        <div className="stat">
          <span className="label">Eventos</span>
          <span className="value">{h.events.length}</span>
        </div>
        <div className="stat">
          <span className="label">Alertas</span>
          <span className="value">{h.alerts.length}</span>
        </div>
        {(['pending', 'confirmed', 'discarded'] as const).map((s) => (
          <div className="stat" key={s}>
            <span className="label">{STATUS_NAMES[s]}s</span>
            <span className="value">{count(s)}</span>
          </div>
        ))}
        <div className="stat">
          <span className="label">Versões do perfil</span>
          <span className="value">{h.profiles.length}</span>
        </div>
        <div className="spacer" />
        {h.session.status !== 'stopped' && <a href={href({ page: 'monitor', sessionId })}>Abrir monitor</a>}
      </section>

      <section className="panel">
        <header>
          <h2>Eventos, alertas e decisões</h2>
          <span className="small secondary num">
            {filtered.length} de {lines.length}
          </span>
        </header>
        <div className="row" style={{ marginBottom: '0.75rem' }}>
          <label className="row small">
            Tipo
            <select value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
              <option value="all">todos</option>
              <option value="beat">batimento</option>
              <option value="window">janela</option>
            </select>
          </label>
          <label className="row small">
            Nível
            <select value={level} onChange={(e) => setLevel(e.target.value as LevelFilter)}>
              <option value="all">todos</option>
              <option value="attention">atenção</option>
              <option value="alert">alerta</option>
              <option value="critical">crítico</option>
            </select>
          </label>
          <label className="row small">
            Decisão
            <select value={status} onChange={(e) => setStatus(e.target.value as StatusFilter)}>
              <option value="all">todas</option>
              <option value="pending">pendente</option>
              <option value="confirmed">confirmado</option>
              <option value="discarded">descartado</option>
              <option value="none">sem alerta</option>
            </select>
          </label>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Tempo</th>
                <th>Tipo</th>
                <th>Nível</th>
                <th>Classe</th>
                <th>KL</th>
                <th>L</th>
                <th>Episódio</th>
                <th>Decisão</th>
                <th>Decidido em</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, shown).map(({ event: e, alert: a }) => (
                <tr
                  key={e.id}
                  className={a ? 'clickable' : undefined}
                  onClick={a ? () => (window.location.hash = href({ page: 'alert', sessionId, alertId: a.id })) : undefined}
                >
                  <td>{formatClock(e.sample / fs)}</td>
                  <td>{e.kind === 'beat' ? 'batimento' : 'janela'}</td>
                  <td>
                    <LevelBadge level={e.level} />
                  </td>
                  <td>{e.cls ? <ClassChip cls={e.cls} /> : '—'}</td>
                  <td>{formatNumber(e.kl)}</td>
                  <td>{formatNumber(e.L)}</td>
                  <td>{a?.episode_id ?? '—'}</td>
                  <td>
                    {a ? <a href={href({ page: 'alert', sessionId, alertId: a.id })}>{STATUS_NAMES[a.status]}</a> : '—'}
                  </td>
                  <td>{formatDateTime(a?.decided_at ?? null)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length > shown && (
          <button style={{ marginTop: '0.75rem' }} onClick={() => setShown((n) => n + HISTORY_PAGE_SIZE)}>
            Mostrar mais {Math.min(HISTORY_PAGE_SIZE, filtered.length - shown)}
          </button>
        )}
      </section>

      <section className="panel">
        <header>
          <h2>Versões do perfil</h2>
        </header>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Versão</th>
                <th>μ</th>
                <th>σ</th>
                <th>k</th>
                <th>L = μ + kσ</th>
                <th>Criada em</th>
              </tr>
            </thead>
            <tbody>
              {h.profiles.map((p) => (
                <tr key={p.id}>
                  <td>v{p.version}</td>
                  <td>{formatNumber(p.mu, 3)}</td>
                  <td>{formatNumber(p.sigma, 3)}</td>
                  <td>{formatNumber(p.k, 1)}</td>
                  <td>{formatNumber(p.L, 3)}</td>
                  <td>{formatDateTime(p.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}
