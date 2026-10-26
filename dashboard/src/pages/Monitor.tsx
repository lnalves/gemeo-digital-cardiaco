import { useState } from 'react'
import { api } from '../api'
import { ClassChip, LevelBadge } from '../components/Badges'
import { EcgStrip } from '../charts/EcgStrip'
import { KlChart } from '../charts/KlChart'
import { ALERT_LIST_LIMIT, ECG_WINDOW_SECONDS, VT_CONSECUTIVE_V } from '../config'
import { useAsync, useMonitor } from '../hooks'
import { CLASS_NAMES, alertSummary, configLabel, formatClock, formatNumber, plural, stateLabel } from '../lib/format'
import type { MonitorStore } from '../lib/monitor'
import { href } from '../lib/route'
import type { SocketStatus } from '../lib/socket'

const SOCKET_TEXT: Record<SocketStatus, string> = {
  connecting: 'Conectando…',
  open: 'Conectado',
  closed: 'Conexão perdida; tentando de novo…',
  gone: 'Sessão não está ativa no servidor',
}

export function Monitor({ sessionId }: { sessionId: string }) {
  const { store, socket, reconnects } = useMonitor(sessionId)
  const session = useAsync(() => api.session(sessionId), [sessionId])
  const state = store.state
  const finished = state?.state === 'finished' || state?.status === 'stopped'
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function control(action: 'pause' | 'resume' | 'stop') {
    setBusy(true)
    setError(null)
    try {
      store.apply(await api.control(sessionId, action))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (socket === 'gone')
    return (
      <section className="panel">
        <h1>Sessão {sessionId}</h1>
        <p>Esta sessão não está mais ativa no servidor. Os dados gravados estão no histórico.</p>
        <a href={href({ page: 'history', sessionId })}>Abrir no histórico</a>
      </section>
    )

  return (
    <>
      <section className="panel row" style={{ gap: '1.5rem' }}>
        <div className="stat">
          <span className="label">Registro</span>
          <span className="value">{state?.record_id ?? session.data?.record_id ?? '—'}</span>
        </div>
        <div className="stat">
          <span className="label">Configuração</span>
          <span>{session.data ? configLabel(session.data.use_profile) : '—'}</span>
          <span className="small secondary">
            Realimentação {session.data?.feedback_enabled ? 'ligada' : 'desligada'}
            {store.profileVersion != null && ` · perfil v${store.profileVersion}`}
          </span>
        </div>
        <div className="stat">
          <span className="label">Estado</span>
          <span className="badge" role="status">
            {state ? stateLabel(state, store.t) : 'Aguardando o sinal…'}
          </span>
        </div>
        <div className="stat">
          <span className="label">Tempo do registro</span>
          <span className="value">{formatClock(store.t)}</span>
        </div>
        <div className="stat">
          <span className="label">Frequência cardíaca</span>
          <span className="value">
            {formatNumber(store.hr, 0)} <small>bpm</small>
          </span>
        </div>
        <div className="spacer" />
        <div className="row">
          {state?.status === 'paused' ? (
            <button disabled={busy || finished} onClick={() => control('resume')}>
              Retomar
            </button>
          ) : (
            <button disabled={busy || finished} onClick={() => control('pause')}>
              Pausar
            </button>
          )}
          <button disabled={busy || finished} onClick={() => control('stop')}>
            Encerrar
          </button>
        </div>
        {error && <p className="error small">{error}</p>}
      </section>

      <section className="panel">
        <header>
          <h2>ECG · últimos {ECG_WINDOW_SECONDS} s</h2>
          <ClassLegend store={store} />
        </header>
        <EcgStrip store={store} />
        <p className="small muted" style={{ margin: '0.5rem 0 0' }}>
          Cada batimento aparece com a classe sugerida cerca de dois intervalos RR depois do pico R: o modelo usa o
          batimento seguinte como contexto. Faixa colorida: nível atenção, alerta ou crítico.{' '}
          <span aria-live="polite">{SOCKET_TEXT[socket]}</span>
          {reconnects > 0 && ` · ${reconnects} reconexões`}
        </p>
      </section>

      <div className="grid-2">
        <section className="panel">
          <header>
            <h2>Desvio do perfil · KL × limiar L</h2>
            <span className="small muted">janela deslizante</span>
          </header>
          {store.kl.length > 0 ? (
            <KlChart kl={store.kl} />
          ) : (
            <div className="empty">
              O gráfico aparece quando o perfil do paciente estiver pronto, ao fim da calibração.
            </div>
          )}
        </section>
        <AlertList store={store} sessionId={sessionId} />
      </div>
    </>
  )
}

function ClassLegend({ store }: { store: MonitorStore }) {
  return (
    <div className="legend" aria-label="Batimentos por classe sugerida">
      {store.classCounts().map(([cls, n]) => (
        <span key={cls} className="chip" title={CLASS_NAMES[cls]}>
          <ClassChip cls={cls} />
          <span className="num">{n}</span>
        </span>
      ))}
    </div>
  )
}

function AlertList({ store, sessionId }: { store: MonitorStore; sessionId: string }) {
  const episodes = store.episodes()
  return (
    <section className="panel">
      <header>
        <h2>Alertas pendentes</h2>
        <span className="small secondary num">
          {plural(store.pendingCount, 'alerta', 'alertas')} · {plural(episodes.length, 'episódio', 'episódios')}
        </span>
      </header>
      {episodes.length === 0 ? (
        <div className="empty">Nenhum alerta pendente.</div>
      ) : (
        <ul className="alert-list">
          {episodes.slice(0, ALERT_LIST_LIMIT).map((e) => {
            const first = e.pending[0]
            return (
              <li key={e.episodeId} style={{ ['--edge' as string]: `var(--level-${e.level})` }}>
                <div className="row">
                  <LevelBadge level={e.level} />
                  <span className="small secondary num">
                    {formatClock(first.t)}
                    {e.pending.length > 1 && ` · ${e.pending.length} alertas no episódio`}
                  </span>
                </div>
                <span className="small">
                  {alertSummary(first.level, first.cls, first.kl, first.L, VT_CONSECUTIVE_V)}
                </span>
                <a href={href({ page: 'alert', sessionId, alertId: first.id })}>Revisar</a>
              </li>
            )
          })}
        </ul>
      )}
      {episodes.length > ALERT_LIST_LIMIT && (
        <p className="small muted">
          Mostrando os {ALERT_LIST_LIMIT} episódios mais recentes. Os demais estão no{' '}
          <a href={href({ page: 'history', sessionId })}>histórico</a>.
        </p>
      )}
    </section>
  )
}
