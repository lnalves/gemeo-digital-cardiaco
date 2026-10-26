export type SocketStatus = 'connecting' | 'open' | 'closed' | 'gone'

/** Exponential backoff with a ceiling. */
export function reconnectDelay(attempt: number, baseMs: number, maxMs: number): number {
  return Math.min(maxMs, baseMs * 2 ** attempt)
}

/** Close codes after which a reconnection cannot help (the session is not active on the server). */
export const FINAL_CLOSE_CODES = new Set([4404])

interface Options {
  baseMs: number
  maxMs: number
  onMessage: (data: unknown) => void
  onStatus: (status: SocketStatus, reconnects: number) => void
  onOpen?: () => void
  create?: (url: string) => WebSocket
}

/** WebSocket that reconnects after a drop (network, server restart, slow-client close 1013). */
export class ReconnectingSocket {
  private ws: WebSocket | null = null
  private timer: ReturnType<typeof setTimeout> | null = null
  private attempt = 0
  private reconnects = 0
  private closed = false
  private readonly url: string
  private readonly options: Options

  constructor(url: string, options: Options) {
    this.url = url
    this.options = options
    this.connect()
  }

  close(): void {
    this.closed = true
    if (this.timer) clearTimeout(this.timer)
    this.ws?.close()
  }

  private connect(): void {
    const { onMessage, onStatus, onOpen, create } = this.options
    onStatus('connecting', this.reconnects)
    const ws = (create ?? ((u) => new WebSocket(u)))(this.url)
    this.ws = ws
    ws.onopen = () => {
      this.attempt = 0
      onStatus('open', this.reconnects)
      onOpen?.()
    }
    ws.onmessage = (e) => onMessage(JSON.parse(e.data as string))
    ws.onclose = (e) => {
      if (this.closed || this.ws !== ws) return
      if (FINAL_CLOSE_CODES.has(e.code)) {
        onStatus('gone', this.reconnects)
        return
      }
      onStatus('closed', this.reconnects)
      const delay = reconnectDelay(this.attempt++, this.options.baseMs, this.options.maxMs)
      this.timer = setTimeout(() => {
        this.reconnects++
        this.connect()
      }, delay)
    }
  }
}

export function wsUrl(path: string): string {
  const { protocol, host } = window.location
  return `${protocol === 'https:' ? 'wss' : 'ws'}://${host}${path}`
}
