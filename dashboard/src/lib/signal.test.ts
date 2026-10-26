import { describe, expect, it } from 'vitest'
import { SignalWindow } from './signal'

const ramp = (from: number, n: number) => Array.from({ length: n }, (_, i) => from + i)

describe('SignalWindow', () => {
  it('keeps the last seconds in order', () => {
    const w = new SignalWindow(4, 2)
    w.push(0, ramp(0, 4))
    w.push(4, ramp(4, 4))
    w.push(8, ramp(8, 4))
    expect(w.endSample).toBe(12)
    expect(w.view().y).toEqual(ramp(4, 8))
    expect(w.view().t).toEqual(ramp(4, 8).map((s) => s / 4))
  })

  it('pads the start with null', () => {
    const w = new SignalWindow(4, 2)
    w.push(0, [1, 2, 3])
    expect(w.view().y).toEqual([null, null, null, null, null, 1, 2, 3])
    expect(w.view().t[0]).toBe(-5 / 4)
  })

  it('leaves a gap empty and skips samples already in the window', () => {
    const w = new SignalWindow(4, 2)
    w.push(0, [0, 1, 2, 3])
    w.push(6, [6, 7])
    expect(w.view().y).toEqual([0, 1, 2, 3, null, null, 6, 7])
    w.push(6, [60, 70, 8])
    expect(w.view().y).toEqual([1, 2, 3, null, null, 6, 7, 8])
  })

  it('takes only the tail of a chunk longer than the window', () => {
    const w = new SignalWindow(2, 2)
    w.push(0, ramp(0, 10))
    expect(w.view().y).toEqual([6, 7, 8, 9])
  })

  it('clears old values after a gap longer than the window', () => {
    const w = new SignalWindow(2, 2)
    w.push(0, [0, 1, 2, 3])
    w.push(20, [20])
    expect(w.view().y).toEqual([null, null, null, 20])
  })
})
