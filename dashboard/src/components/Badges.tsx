import { CLASS_NAMES, LEVEL_NAMES } from '../lib/format'
import type { ClassName, Level } from '../types'

function LevelIcon({ level }: { level: Level }) {
  const color = `var(--level-${level})`
  if (level === 'critical')
    return (
      <svg className="icon" viewBox="0 0 16 16" aria-hidden>
        <path d="M8 1 15 14H1Z" fill={color} />
        <path d="M8 5.5v4M8 11.5v.5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    )
  if (level === 'alert')
    return (
      <svg className="icon" viewBox="0 0 16 16" aria-hidden>
        <circle cx="8" cy="8" r="7" fill={color} />
        <path d="M8 4.5v4.5M8 11v.5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    )
  if (level === 'attention')
    return (
      <svg className="icon" viewBox="0 0 16 16" aria-hidden>
        <rect x="2" y="2" width="12" height="12" rx="2" transform="rotate(45 8 8)" fill={color} />
      </svg>
    )
  return (
    <svg className="icon" viewBox="0 0 16 16" aria-hidden>
      <circle cx="8" cy="8" r="6" fill={color} />
    </svg>
  )
}

export function LevelBadge({ level }: { level: Level }) {
  return (
    <span className="badge">
      <LevelIcon level={level} />
      {LEVEL_NAMES[level]}
    </span>
  )
}

export function ClassChip({ cls, long = false }: { cls: ClassName; long?: boolean }) {
  return (
    <span className="chip">
      <span className="dot" style={{ background: `var(--cls-${cls})` }} aria-hidden />
      <strong>{cls}</strong>
      {long && <span className="secondary">{CLASS_NAMES[cls]}</span>}
    </span>
  )
}

export function Disclaimer() {
  return (
    <div className="disclaimer" role="note">
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8 7v4.5M8 4.5v.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
      <span>
        <strong>Ferramenta de apoio à decisão.</strong> Classes e alertas são sugestões do modelo e não
        constituem diagnóstico. A decisão é sempre do profissional de saúde.
      </span>
    </div>
  )
}
