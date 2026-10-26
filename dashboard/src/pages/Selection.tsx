import { useState } from 'react'
import { api } from '../api'
import { CALIBRATION_SECONDS, DEFAULT_SPEED, SPEEDS, VT_CONSECUTIVE_V } from '../config'
import { useAsync } from '../hooks'
import { configLabel, formatClock, formatDateTime } from '../lib/format'
import { href } from '../lib/route'

export function Selection() {
  const records = useAsync(api.records, [])
  const sessions = useAsync(api.sessions, [])
  const [recordId, setRecordId] = useState<string | null>(null)
  const [useProfile, setUseProfile] = useState(true)
  const [feedback, setFeedback] = useState(false)
  const [speed, setSpeed] = useState<number>(DEFAULT_SPEED)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const selected = recordId ?? records.data?.find((r) => !r.training)?.record_id ?? null
  const record = records.data?.find((r) => r.record_id === selected)
  const active = sessions.data?.filter((s) => s.status !== 'stopped') ?? []

  async function start() {
    if (!selected) return
    setStarting(true)
    setError(null)
    try {
      const { session_id } = await api.createSession({
        record_id: selected,
        use_profile: useProfile,
        feedback_enabled: feedback,
        speed,
      })
      window.location.hash = href({ page: 'monitor', sessionId: session_id })
    } catch (e) {
      setError((e as Error).message)
      setStarting(false)
    }
  }

  return (
    <>
      <h1>Nova sessão de monitoramento</h1>
      {active.length > 0 && (
        <section className="panel">
          <header>
            <h2>Sessões em andamento</h2>
          </header>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Registro</th>
                  <th>Configuração</th>
                  <th>Início</th>
                  <th>Situação</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {active.map((s) => (
                  <tr key={s.id}>
                    <td>{s.record_id}</td>
                    <td>{configLabel(s.use_profile)}</td>
                    <td>{formatDateTime(s.started_at)}</td>
                    <td>{s.status === 'paused' ? 'Pausada' : 'Em andamento'}</td>
                    <td>
                      <a href={href({ page: 'monitor', sessionId: s.id })}>Abrir monitor</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="grid-2">
        <section className="panel">
          <header>
            <h2>Registro (paciente)</h2>
            <span className="small muted">MIT-BIH, derivação MLII, 360 Hz</span>
          </header>
          {records.error && <p className="error">{records.error}</p>}
          <div className="table-wrap" style={{ maxHeight: 460, overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Registro</th>
                  <th>Conjunto</th>
                  <th>Duração</th>
                </tr>
              </thead>
              <tbody>
                {records.data?.map((r) => (
                  <tr
                    key={r.record_id}
                    className="clickable"
                    aria-selected={r.record_id === selected}
                    onClick={() => setRecordId(r.record_id)}
                  >
                    <td>
                      <label className="chip">
                        <input
                          type="radio"
                          name="record"
                          checked={r.record_id === selected}
                          onChange={() => setRecordId(r.record_id)}
                        />
                        {r.record_id}
                      </label>
                    </td>
                    <td>{r.training ? `${r.split} · usado no treino` : `${r.split} · teste`}</td>
                    <td>{formatClock(r.duration_s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel" style={{ display: 'grid', gap: '1rem', alignContent: 'start' }}>
          <h2>Configuração</h2>
          <label className="toggle">
            <input type="checkbox" checked={useProfile} onChange={(e) => setUseProfile(e.target.checked)} />
            <span>
              Usar perfil do paciente
              <br />
              <span className="small secondary">
                Ligado: configuração B (alerta quando a classe e o desvio KL concordam). Desligado: configuração A
                (todo batimento diferente de N gera alerta).
              </span>
            </span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={feedback} onChange={(e) => setFeedback(e.target.checked)} />
            <span>
              Realimentação pelo especialista
              <br />
              <span className="small secondary">
                Cada alerta descartado ajusta a distribuição de referência do perfil e cria uma nova versão.
              </span>
            </span>
          </label>
          <div className="row">
            <span>Velocidade</span>
            <div className="segmented" role="group" aria-label="Velocidade">
              {SPEEDS.map((s) => (
                <button key={s} aria-pressed={speed === s} onClick={() => setSpeed(s)}>
                  {s}×
                </button>
              ))}
            </div>
          </div>
          {record?.training && (
            <p className="small secondary" role="status">
              O registro {record.record_id} fez parte do treino do classificador: os resultados dele são otimistas.
              Para demonstrar o sistema, prefira um registro de DS2.
            </p>
          )}
          <p className="small secondary">
            Os primeiros {formatClock(CALIBRATION_SECONDS)} min de sinal calibram o perfil. Nesse período só o
            alerta crítico ({VT_CONSECUTIVE_V} ou mais V seguidos) é emitido.
          </p>
          {error && <p className="error">{error}</p>}
          <button className="primary" disabled={!selected || starting} onClick={start}>
            {starting ? 'Iniciando…' : `Iniciar monitoramento${selected ? ` do registro ${selected}` : ''}`}
          </button>
          {sessions.data && sessions.data.length > 0 && (
            <a href={href({ page: 'history', sessionId: null })} className="small">
              Ver {sessions.data.length} sessões gravadas no histórico
            </a>
          )}
        </section>
      </div>
    </>
  )
}
