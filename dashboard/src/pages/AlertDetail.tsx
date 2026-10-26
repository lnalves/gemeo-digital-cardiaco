import { useState } from 'react'
import { api } from '../api'
import { ClassChip, LevelBadge } from '../components/Badges'
import { SegmentChart } from '../charts/SegmentChart'
import { VT_CONSECUTIVE_V } from '../config'
import { useAsync } from '../hooks'
import {
  CLASSES,
  STATUS_NAMES,
  alertSummary,
  configLabel,
  formatClock,
  formatDateTime,
  formatNumber,
  formatPercent,
} from '../lib/format'
import { href } from '../lib/route'
import { contextLabel } from '../lib/views'
import type { AlertMessage, AlertRow, Decision, DecisionResult } from '../types'

export function AlertDetail({ sessionId, alertId }: { sessionId: string; alertId: string }) {
  const history = useAsync(() => api.history(sessionId), [sessionId, alertId])
  const records = useAsync(api.records, [])
  const [result, setResult] = useState<DecisionResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (history.error) return <p className="error">{history.error}</p>
  if (!history.data || !records.data) return <p className="muted">Carregando…</p>
  const row = history.data.alerts.find((a) => a.id === alertId)
  if (!row) return <p className="error">Alerta {alertId} não encontrado.</p>
  const session = history.data.session
  const fs = records.data.find((r) => r.record_id === session.record_id)?.fs
  const a = row.payload
  const status = result?.status ?? row.status
  const next = nextPending(history.data.alerts, row)
  const live = session.status !== 'stopped'

  async function decide(decision: Decision) {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.decide(alertId, decision))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="row">
        <a href={live ? href({ page: 'monitor', sessionId }) : href({ page: 'history', sessionId })}>
          ← {live ? 'Voltar ao monitor' : 'Voltar ao histórico'}
        </a>
      </div>
      <section className="panel row" style={{ gap: '1.5rem' }}>
        <h1>Detalhe do alerta</h1>
        <LevelBadge level={a.level} />
        <span className="secondary num">
          Registro {session.record_id} · {formatClock(a.t)} · episódio {a.episode_id} ·{' '}
          {configLabel(session.use_profile)}
        </span>
        <div className="spacer" />
        <span className="badge">{STATUS_NAMES[status]}</span>
      </section>

      <section className="panel">
        <header>
          <h2>Trecho do sinal e atenção do modelo</h2>
          <span className="small muted">{a.context_samples.length} batimentos de contexto</span>
        </header>
        <p style={{ marginTop: 0 }}>{alertSummary(a.level, a.cls, a.kl, a.L, VT_CONSECUTIVE_V)}</p>
        {fs ? <SegmentChart alert={a} fs={fs} /> : <p className="error">Frequência de amostragem desconhecida.</p>}
        <p className="small muted" style={{ marginBottom: 0 }}>
          Faixas: um batimento cada; quanto mais escura, maior o peso de atenção que o modelo deu a ele. Linha
          tracejada: batimento do alerta (i).
        </p>
        <AttentionTable alert={a} />
      </section>

      <div className="grid-2">
        <section className="panel">
          <header>
            <h2>Classe sugerida</h2>
            <ClassChip cls={a.cls} long />
          </header>
          <Probabilities alert={a} />
          <h3 style={{ margin: '1.25rem 0 0.5rem' }}>Desvio do perfil</h3>
          <KlVersusL kl={a.kl} L={a.L} />
        </section>

        <section className="panel" style={{ display: 'grid', gap: '0.75rem', alignContent: 'start' }}>
          <h2>Decisão do especialista</h2>
          <p className="small secondary" style={{ margin: 0 }}>
            Confirmar: o alerta procede. Descartar: o alerta não procede.
            {session.feedback_enabled
              ? ' Com a realimentação ligada, o descarte ajusta o perfil do paciente e cria uma nova versão.'
              : ' A realimentação está desligada: o descarte não muda o perfil.'}
          </p>
          <div className="row">
            <button className="primary" disabled={busy || status !== 'pending'} onClick={() => decide('confirm')}>
              Confirmar
            </button>
            <button disabled={busy || status !== 'pending'} onClick={() => decide('discard')}>
              Descartar
            </button>
          </div>
          {error && <p className="error small">{error}</p>}
          {status !== 'pending' && (
            <p className="small" role="status" style={{ margin: 0 }}>
              {STATUS_NAMES[status]} em {formatDateTime(result?.decided_at ?? row.decided_at)}.
              {result?.profile_version != null && ` Perfil atualizado para a versão ${result.profile_version}.`}
            </p>
          )}
          {next && (
            <a href={href({ page: 'alert', sessionId, alertId: next.id })}>
              Próximo alerta pendente do episódio ({formatClock(next.payload.t)}) →
            </a>
          )}
        </section>
      </div>
    </>
  )
}

function nextPending(alerts: AlertRow[], current: AlertRow): AlertRow | undefined {
  return alerts
    .filter((a) => a.episode_id === current.episode_id && a.status === 'pending' && a.id !== current.id)
    .sort((x, y) => x.payload.t - y.payload.t)[0]
}

function AttentionTable({ alert }: { alert: AlertMessage }) {
  const center = alert.context_samples.indexOf(alert.sample)
  return (
    <div className="table-wrap" style={{ marginTop: '0.75rem' }}>
      <table aria-label="Peso de atenção por batimento de contexto">
        <thead>
          <tr>
            <th>Batimento</th>
            {alert.context_samples.map((s, i) => (
              <th key={s}>{contextLabel(i, center)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Atenção</td>
            {alert.attention.map((w, i) => (
              <td key={i}>{formatPercent(w, 1)}</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function Probabilities({ alert }: { alert: AlertMessage }) {
  return (
    <div className="bars" role="table" aria-label="Probabilidade de cada classe">
      {CLASSES.map((c, i) => (
        <div key={c} role="row" style={{ display: 'contents' }}>
          <span role="cell">
            <ClassChip cls={c} />
          </span>
          <div className="track" role="cell">
            <div className={`fill${c === alert.cls ? ' strong' : ''}`} style={{ width: `${alert.prob[i] * 100}%` }} />
          </div>
          <span role="cell" className="num">
            {formatPercent(alert.prob[i])}
          </span>
        </div>
      ))}
    </div>
  )
}

function KlVersusL({ kl, L }: { kl: number | null; L: number | null }) {
  if (kl == null || L == null)
    return <p className="small secondary">Sem KL: o alerta ocorreu antes do fim da calibração do perfil.</p>
  const max = Math.max(kl, L) * 1.25
  return (
    <div style={{ display: 'grid', gap: '0.4rem' }}>
      <div className="bullet" role="img" aria-label={`KL ${formatNumber(kl)}, limiar ${formatNumber(L)}`}>
        <div className="fill" style={{ width: `${(kl / max) * 100}%` }} />
        <div className="mark" style={{ left: `${(L / max) * 100}%` }} />
      </div>
      <span className="small num">
        KL da janela <strong>{formatNumber(kl)}</strong> · limiar L <strong>{formatNumber(L)}</strong> ·{' '}
        {kl > L ? 'acima do limiar: o ritmo recente difere do perfil' : 'dentro do perfil'}
      </span>
    </div>
  )
}
