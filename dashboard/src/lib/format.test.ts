import { describe, expect, it } from 'vitest'
import type { StateMessage } from '../types'
import { alertSummary, formatClock, formatNumber, formatPercent, plural, stateLabel } from './format'

const state = (s: Partial<StateMessage>): StateMessage => ({
  type: 'state',
  state: 'calibrating',
  status: 'running',
  record_id: '100',
  calibration_seconds: 300,
  t: 0,
  ...s,
})

describe('format', () => {
  it('formats record time', () => {
    expect(formatClock(0)).toBe('0:00')
    expect(formatClock(65.9)).toBe('1:05')
    expect(formatClock(3725)).toBe('1:02:05')
  })

  it('uses the Brazilian decimal comma', () => {
    expect(formatNumber(0.8412)).toBe('0,84')
    expect(formatNumber(null)).toBe('—')
    expect(formatPercent(0.873)).toBe('87 %')
    expect(formatPercent(0.0042, 1)).toBe('0,4 %')
    expect([plural(1, 'episódio', 'episódios'), plural(2, 'episódio', 'episódios')]).toEqual([
      '1 episódio',
      '2 episódios',
    ])
  })

  it('counts down the calibration', () => {
    expect(stateLabel(state({}), 59.5)).toBe('Calibrando o perfil · faltam 4:01')
    expect(stateLabel(state({}), 400)).toBe('Calibrando o perfil · faltam 0:00')
    expect(stateLabel(state({ state: 'monitoring' }), 10)).toBe('Monitorando')
  })

  it('never states a diagnosis', () => {
    const texts = [
      alertSummary('critical', 'V', null, null, 3),
      alertSummary('alert', 'S', 0.9, 0.5, 3),
      alertSummary('alert', 'F', null, null, 3),
    ]
    expect(texts[0]).toContain('3 ou mais V seguidos')
    expect(texts[1]).toContain('acima do limiar')
    for (const t of texts) {
      expect(t).toMatch(/sugerido/)
      expect(t).not.toMatch(/diagn[oó]stic|arritmia detectada|paciente tem/i)
    }
  })
})
