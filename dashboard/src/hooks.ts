import { type DependencyList, useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { RECONNECT_BASE_MS, RECONNECT_MAX_MS, UI_REFRESH_MS, ECG_WINDOW_SECONDS } from './config'
import { MonitorStore } from './lib/monitor'
import { parseRoute, type Route } from './lib/route'
import { ReconnectingSocket, type SocketStatus, wsUrl } from './lib/socket'
import type { MonitorMessage } from './types'

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash))
  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  return route
}

export interface Async<T> {
  data: T | null
  error: string | null
  reload: () => void
}

export function useAsync<T>(load: () => Promise<T>, deps: DependencyList): Async<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [n, setN] = useState(0)
  useEffect(() => {
    let live = true
    load().then(
      (d) => live && (setData(d), setError(null)),
      (e: Error) => live && setError(e.message),
    )
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n])
  return { data, error, reload: useCallback(() => setN((x) => x + 1), []) }
}

export interface Monitor {
  store: MonitorStore
  socket: SocketStatus
  reconnects: number
}

/** Session state from /ws/monitor; mount it with `key={sessionId}`. On each (re)connection the history fills the alerts missed meanwhile.
 *  React re-renders at most every UI_REFRESH_MS; the ECG strip draws from the store on its own. */
export function useMonitor(sessionId: string): Monitor {
  const [store] = useState(() => new MonitorStore(ECG_WINDOW_SECONDS))
  const [socket, setSocket] = useState<SocketStatus>('connecting')
  const [reconnects, setReconnects] = useState(0)
  const [, setVersion] = useState(0)

  useEffect(() => {
    const s = store
    const ws = new ReconnectingSocket(wsUrl(`/ws/monitor/${sessionId}`), {
      baseMs: RECONNECT_BASE_MS,
      maxMs: RECONNECT_MAX_MS,
      onMessage: (m) => s.apply(m as MonitorMessage),
      onStatus: (status, n) => {
        setSocket(status)
        setReconnects(n)
      },
      onOpen: () => {
        api.history(sessionId).then((h) => s.applyHistory(h), () => undefined)
      },
    })
    let shown = -1
    const timer = setInterval(() => {
      if (s.version !== shown) {
        shown = s.version
        setVersion(shown)
      }
    }, UI_REFRESH_MS)
    return () => {
      clearInterval(timer)
      ws.close()
    }
  }, [sessionId, store])

  return { store, socket, reconnects }
}
