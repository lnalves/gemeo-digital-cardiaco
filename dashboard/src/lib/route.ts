export type Route =
  | { page: 'selection' }
  | { page: 'monitor'; sessionId: string }
  | { page: 'alert'; sessionId: string; alertId: string }
  | { page: 'history'; sessionId: string | null }

export function parseRoute(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  if (parts[0] === 'sessoes' && parts[1]) {
    if (parts[2] === 'alertas' && parts[3]) return { page: 'alert', sessionId: parts[1], alertId: parts[3] }
    return { page: 'monitor', sessionId: parts[1] }
  }
  if (parts[0] === 'historico') return { page: 'history', sessionId: parts[1] ?? null }
  return { page: 'selection' }
}

export function href(route: Route): string {
  const e = encodeURIComponent
  switch (route.page) {
    case 'selection':
      return '#/'
    case 'monitor':
      return `#/sessoes/${e(route.sessionId)}`
    case 'alert':
      return `#/sessoes/${e(route.sessionId)}/alertas/${e(route.alertId)}`
    case 'history':
      return route.sessionId ? `#/historico/${e(route.sessionId)}` : '#/historico'
  }
}
