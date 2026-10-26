import type { ClassName, Level } from './types'

/** Value of a color token of index.css for the current theme (canvas drawing cannot use var()). */
export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export const classColor = (cls: ClassName) => token(`--cls-${cls}`)
export const levelColor = (level: Level) => token(`--level-${level}`)

/** Re-run `redraw` when the OS switches between light and dark. */
export function onThemeChange(redraw: () => void): () => void {
  const media = window.matchMedia('(prefers-color-scheme: dark)')
  media.addEventListener('change', redraw)
  return () => media.removeEventListener('change', redraw)
}
