/** The last `seconds` of a signal, keyed by absolute sample. A gap between chunks stays empty (null). */
export class SignalWindow {
  readonly fs: number
  readonly size: number
  private readonly values: Float64Array
  private end = 0

  constructor(fs: number, seconds: number) {
    this.fs = fs
    this.size = Math.round(fs * seconds)
    this.values = new Float64Array(this.size).fill(NaN)
  }

  /** Next sample expected. */
  get endSample(): number {
    return this.end
  }

  push(t0: number, samples: ArrayLike<number>): void {
    const stop = t0 + samples.length
    if (stop <= this.end) return
    const from = Math.max(t0, this.end, stop - this.size)
    for (let s = Math.max(this.end, stop - this.size); s < from; s++) this.values[s % this.size] = NaN
    for (let s = from; s < stop; s++) this.values[s % this.size] = samples[s - t0]
    this.end = stop
  }

  /** Time (s) and values of the window, oldest first; empty samples are null. */
  view(): { t: number[]; y: (number | null)[] } {
    const start = this.end - this.size
    const t = new Array<number>(this.size)
    const y = new Array<number | null>(this.size)
    for (let i = 0; i < this.size; i++) {
      const s = start + i
      t[i] = s / this.fs
      const v = s < 0 ? NaN : this.values[s % this.size]
      y[i] = Number.isNaN(v) ? null : v
    }
    return { t, y }
  }
}
