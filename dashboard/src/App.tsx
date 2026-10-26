import { Disclaimer } from './components/Badges'
import { useRoute } from './hooks'
import { href } from './lib/route'
import { AlertDetail } from './pages/AlertDetail'
import { History } from './pages/History'
import { Monitor } from './pages/Monitor'
import { Selection } from './pages/Selection'

export function App() {
  const route = useRoute()
  return (
    <>
      <header className="app-header">
        <div className="bar">
          <span className="brand">
            <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden>
              <path
                d="M2 12h4l2-5 4 10 2-5h8"
                fill="none"
                stroke="var(--level-critical)"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Gêmeo digital cardíaco
          </span>
          <nav>
            <a href={href({ page: 'selection' })} aria-current={route.page === 'selection' ? 'page' : undefined}>
              Nova sessão
            </a>
            <a href={href({ page: 'history', sessionId: null })} aria-current={route.page === 'history' ? 'page' : undefined}>
              Histórico
            </a>
          </nav>
        </div>
        <Disclaimer />
      </header>
      <main>
        {route.page === 'selection' && <Selection />}
        {route.page === 'monitor' && <Monitor key={route.sessionId} sessionId={route.sessionId} />}
        {route.page === 'alert' && (
          <AlertDetail key={route.alertId} sessionId={route.sessionId} alertId={route.alertId} />
        )}
        {route.page === 'history' && <History sessionId={route.sessionId} />}
      </main>
    </>
  )
}
