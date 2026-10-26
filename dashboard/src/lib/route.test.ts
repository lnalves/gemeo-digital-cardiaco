import { describe, expect, it } from 'vitest'
import { href, parseRoute, type Route } from './route'

describe('route', () => {
  it.each<Route>([
    { page: 'selection' },
    { page: 'monitor', sessionId: 'abc123' },
    { page: 'alert', sessionId: 'abc123', alertId: 'abc123-90210' },
    { page: 'history', sessionId: null },
    { page: 'history', sessionId: 'abc123' },
  ])('round-trips %o', (route) => {
    expect(parseRoute(href(route))).toEqual(route)
  })

  it('falls back to the selection', () => {
    expect(parseRoute('')).toEqual({ page: 'selection' })
    expect(parseRoute('#/nada')).toEqual({ page: 'selection' })
    expect(parseRoute('#/sessoes')).toEqual({ page: 'selection' })
  })
})
