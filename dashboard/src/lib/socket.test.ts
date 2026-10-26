import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ReconnectingSocket, reconnectDelay, type SocketStatus } from './socket'

class FakeWebSocket {
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: ((e: { code: number }) => void) | null = null
  closed = false
  close() {
    this.closed = true
  }
}

describe('reconnectDelay', () => {
  it('doubles up to the ceiling', () => {
    expect([0, 1, 2, 3, 4, 5].map((a) => reconnectDelay(a, 500, 8000))).toEqual([
      500, 1000, 2000, 4000, 8000, 8000,
    ])
  })
})

describe('ReconnectingSocket', () => {
  let sockets: FakeWebSocket[]
  let statuses: [SocketStatus, number][]
  let messages: unknown[]
  const open = () =>
    new ReconnectingSocket('ws://x', {
      baseMs: 100,
      maxMs: 400,
      onMessage: (m) => messages.push(m),
      onStatus: (s, n) => statuses.push([s, n]),
      create: () => {
        const ws = new FakeWebSocket()
        sockets.push(ws)
        return ws as unknown as WebSocket
      },
    })

  beforeEach(() => {
    vi.useFakeTimers()
    sockets = []
    statuses = []
    messages = []
  })
  afterEach(() => vi.useRealTimers())

  it('reconnects after a drop and resets the backoff when open', () => {
    open()
    sockets[0].onopen?.()
    sockets[0].onmessage?.({ data: '{"type":"hr","bpm":60}' })
    sockets[0].onclose?.({ code: 1013 })
    vi.advanceTimersByTime(99)
    expect(sockets).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(sockets).toHaveLength(2)
    sockets[1].onclose?.({ code: 1006 })
    vi.advanceTimersByTime(200)
    expect(sockets).toHaveLength(3)
    sockets[2].onopen?.()
    expect(messages).toEqual([{ type: 'hr', bpm: 60 }])
    expect(statuses.at(-1)).toEqual(['open', 2])
  })

  it('stops on a final close code and on close()', () => {
    const s = open()
    sockets[0].onclose?.({ code: 4404 })
    vi.advanceTimersByTime(10_000)
    expect(sockets).toHaveLength(1)
    expect(statuses.at(-1)).toEqual(['gone', 0])

    const t = open()
    t.close()
    sockets[1].onclose?.({ code: 1000 })
    vi.advanceTimersByTime(10_000)
    expect(sockets).toHaveLength(2)
    expect(sockets[1].closed).toBe(true)
    s.close()
  })
})
