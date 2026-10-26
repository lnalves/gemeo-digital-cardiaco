import type { AlertStatus, ClassName, Level, PipelineState, StateMessage } from '../types'

export const CLASSES: ClassName[] = ['N', 'S', 'V', 'F', 'Q']

export const CLASS_NAMES: Record<ClassName, string> = {
  N: 'normal',
  S: 'ectópico supraventricular',
  V: 'ectópico ventricular',
  F: 'fusão',
  Q: 'não classificável',
}

export const LEVEL_NAMES: Record<Level, string> = {
  normal: 'Normal',
  attention: 'Atenção',
  alert: 'Alerta',
  critical: 'Alerta crítico',
}

export const STATUS_NAMES: Record<AlertStatus, string> = {
  pending: 'Pendente',
  confirmed: 'Confirmado',
  discarded: 'Descartado',
}

const STATE_NAMES: Record<PipelineState, string> = {
  calibrating: 'Calibrando o perfil',
  monitoring: 'Monitorando',
  paused: 'Pausado',
  finished: 'Encerrado',
}

const number = (digits: number) =>
  new Intl.NumberFormat('pt-BR', { minimumFractionDigits: digits, maximumFractionDigits: digits })

export function formatNumber(x: number | null | undefined, digits = 2): string {
  return x == null || Number.isNaN(x) ? '—' : number(digits).format(x)
}

export function formatPercent(x: number, digits = 0): string {
  return `${number(digits).format(x * 100)} %`
}

export function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`
}

/** Record time: m:ss below one hour, h:mm:ss above. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = String(s % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'medium' })
}

/** `t`: record time now; the state message keeps the time of the last state change. */
export function stateLabel(state: StateMessage, t: number): string {
  if (state.state === 'calibrating') {
    const left = state.calibration_seconds - t
    return `${STATE_NAMES.calibrating} · faltam ${formatClock(Math.ceil(Math.max(0, left)))}`
  }
  return STATE_NAMES[state.state]
}

export function configLabel(useProfile: boolean | number): string {
  return useProfile ? 'B · classificador + perfil' : 'A · classificador isolado'
}

/** Short text of an alert. Describes what the system saw; never states a diagnosis (RNF07). */
export function alertSummary(
  level: Level,
  cls: ClassName,
  kl: number | null,
  L: number | null,
  vtRun: number,
): string {
  const beat = `Batimento sugerido como ${cls} (${CLASS_NAMES[cls]})`
  if (level === 'critical') return `${beat}: ${vtRun} ou mais V seguidos. Revise o traçado com prioridade.`
  if (kl != null && L != null && kl > L) return `${beat}, com desvio do perfil acima do limiar. Revise o traçado.`
  return `${beat}. Revise o traçado.`
}
