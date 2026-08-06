import { useEffect, useState, type SubmitEvent } from 'react'
import Lenis from 'lenis'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

interface SimulationResult {
  revenue: number
  monthly_costs: number
  burn_rate: number
  cash_on_hand: number
  runway_months: number
  growth_rate: number
  customer_count: number
  employee_count: number
  marketing_spend: number
  customers_churned: number
  customers_acquired: number
  market_condition: string
  market_multiplier: number
  narration: string
}

interface Snapshot {
  id: number
  month_number: number
  revenue: number
  cash_on_hand: number
  customer_count: number
  customers_churned: number
  customers_acquired: number
}

interface ChurnModelResult {
  accuracy: number
  precision: number
  recall: number
  feature_importance: Record<string, number>
}

interface ModelMetrics {
  churn_model_comparison: Record<string, ChurnModelResult>
  growth_model: { mae: number; r2: number }
}

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(
    (localStorage.getItem('theme') as 'dark' | 'light') || 'dark'
  )

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [authError, setAuthError] = useState('')
  const [authLoading, setAuthLoading] = useState<'login' | 'register' | 'guest' | null>(null)

  const [name, setName] = useState('')
  const [businessType, setBusinessType] = useState('')
  const [initialPrice, setInitialPrice] = useState('')
  const [founderCount, setFounderCount] = useState('')
  const [initialFunding, setInitialFunding] = useState('')
  const [initialCustomerCount, setInitialCustomerCount] = useState('')
  const [createdStartupId, setCreatedStartupId] = useState<number | null>(null)
  const [creatingStartup, setCreatingStartup] = useState(false)

  const [simMarketingSpend, setSimMarketingSpend] = useState('')
  const [simEmployeeCount, setSimEmployeeCount] = useState('')
  const [simResult, setSimResult] = useState<SimulationResult | null>(null)
  const [simulating, setSimulating] = useState(false)

  const [history, setHistory] = useState<Snapshot[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)

  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null)

  useEffect(() => {
    fetch('http://127.0.0.1:8000/model-metrics')
      .then((res) => res.json())
      .then(setModelMetrics)
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    const lenis = new Lenis({ duration: 1.1, easing: (t) => 1 - Math.pow(1 - t, 3) })
    function raf(time: number) {
      lenis.raf(time)
      requestAnimationFrame(raf)
    }
    requestAnimationFrame(raf)
    return () => lenis.destroy()
  }, [])

  async function loginWithCredentials() {
    const response = await fetch('http://127.0.0.1:8000/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })

    if (!response.ok) {
      setAuthError('Login failed')
      return
    }

    const data = await response.json()
    localStorage.setItem('token', data.access_token)
    setToken(data.access_token)
  }

  async function handleLogin(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault()
    setAuthError('')
    setAuthLoading('login')
    await loginWithCredentials()
    setAuthLoading(null)
  }

  async function handleRegister() {
    setAuthError('')
    setAuthLoading('register')

    const response = await fetch('http://127.0.0.1:8000/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })

    if (!response.ok) {
      setAuthError('Registration failed')
      setAuthLoading(null)
      return
    }

    await loginWithCredentials()
    setAuthLoading(null)
  }

  async function handleGuestLogin() {
    setAuthError('')
    setAuthLoading('guest')

    const response = await fetch('http://127.0.0.1:8000/guest-login', { method: 'POST' })
    const data = await response.json()
    localStorage.setItem('token', data.access_token)
    setToken(data.access_token)
    setAuthLoading(null)
  }

  function handleLogout() {
    localStorage.removeItem('token')
    setToken(null)
  }

  async function handleSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault()
    setCreatingStartup(true)

    const response = await fetch('http://127.0.0.1:8000/startups', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        name,
        business_type: businessType,
        initial_price: Number(initialPrice),
        founder_count: Number(founderCount),
        initial_funding: Number(initialFunding),
        initial_customer_count: Number(initialCustomerCount),
      }),
    })

    const data = await response.json()
    setCreatedStartupId(data.startup_id)
    setCreatingStartup(false)
  }

  async function handleSimulate(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault()
    setSimulating(true)

    const response = await fetch(
      `http://127.0.0.1:8000/startups/${createdStartupId}/simulate-next-month`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          marketing_spend: Number(simMarketingSpend),
          employee_count: Number(simEmployeeCount),
        }),
      }
    )

    const data = await response.json()
    setSimResult(data)
    setSimulating(false)
  }

  async function handleViewHistory() {
    setLoadingHistory(true)
    const response = await fetch(`http://127.0.0.1:8000/startups/${createdStartupId}/snapshots`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    const data = await response.json()
    setHistory(data)
    setLoadingHistory(false)
  }

  function buildFeatureImportanceData(comparison: Record<string, ChurnModelResult>) {
    const featureNames = Object.keys(Object.values(comparison)[0]?.feature_importance ?? {})
    return featureNames.map((feature) => {
      const row: Record<string, string | number> = { feature }
      for (const [modelName, result] of Object.entries(comparison)) {
        const total = Object.values(result.feature_importance).reduce((sum, v) => sum + Math.abs(v), 0)
        row[modelName] = total > 0 ? Math.abs(result.feature_importance[feature]) / total : 0
      }
      return row
    })
  }

  const themeToggle = (
    <button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
      {theme === 'dark' ? '☀ Light' : '☾ Dark'}
    </button>
  )

  if (!token) {
    return (
      <div className="auth-screen">
        {themeToggle}
        <h1>AI Startup Intelligence Platform</h1>
        <div className="card auth-card">
          <form onSubmit={handleLogin}>
            <div className="field-stack">
              <input className="field" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
              <input className="field" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
            </div>
            <div className="btn-row">
              <button className="btn btn-primary" type="submit" disabled={authLoading !== null}>
                {authLoading === 'login' && <span className="spinner" />}
                Log In
              </button>
              <button className="btn btn-secondary" type="button" onClick={handleRegister} disabled={authLoading !== null}>
                {authLoading === 'register' && <span className="spinner" />}
                Register
              </button>
            </div>
          </form>
          <button className="btn-ghost" type="button" onClick={handleGuestLogin} disabled={authLoading !== null}>
            {authLoading === 'guest' ? 'Setting up...' : 'Try as Guest →'}
          </button>
          {authError && <p className="error-text">{authError}</p>}
        </div>

        {modelMetrics && (
          <div className="card metrics-card">
            <h2>🧠 Live Model Benchmarks</h2>
            <p className="card-sub">Real churn models, trained and evaluated on every server start</p>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(modelMetrics.churn_model_comparison).map(([name, result]) => (
                  <tr key={name}>
                    <td>{name.replace(/_/g, ' ')}</td>
                    <td>{(result.accuracy * 100).toFixed(1)}%</td>
                    <td>{(result.precision * 100).toFixed(1)}%</td>
                    <td>{(result.recall * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="card-sub metrics-footnote">
              Growth model (linear regression): R² {modelMetrics.growth_model.r2.toFixed(2)}, MAE{' '}
              {modelMetrics.growth_model.mae.toFixed(1)} customers
            </p>

            <p className="chart-title">Feature importance by model</p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={buildFeatureImportanceData(modelMetrics.churn_model_comparison)}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
                <XAxis dataKey="feature" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="logistic_regression" fill="#6e7bff" radius={[4, 4, 0, 0]} />
                <Bar dataKey="random_forest" fill="#4ade80" radius={[4, 4, 0, 0]} />
                <Bar dataKey="xgboost" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="app">
      <div className="topbar">
        <h1>AI Startup Intelligence Platform</h1>
        <div className="topbar-actions">
          {themeToggle}
          <button className="theme-toggle" onClick={handleLogout}>Log Out</button>
        </div>
      </div>

      <div className="card">
        <h2>Create a Startup</h2>
        <form onSubmit={handleSubmit}>
          <div className="field-grid">
            <input className="field" type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Startup name" />
            <input className="field" type="text" value={businessType} onChange={(e) => setBusinessType(e.target.value)} placeholder="Business type" />
            <input className="field" type="number" value={initialPrice} onChange={(e) => setInitialPrice(e.target.value)} placeholder="Initial price" />
            <input className="field" type="number" value={founderCount} onChange={(e) => setFounderCount(e.target.value)} placeholder="Founder count" />
            <input className="field" type="number" value={initialFunding} onChange={(e) => setInitialFunding(e.target.value)} placeholder="Initial funding" />
            <input className="field" type="number" value={initialCustomerCount} onChange={(e) => setInitialCustomerCount(e.target.value)} placeholder="Initial customers" />
          </div>
          <button className="btn btn-primary" type="submit" disabled={creatingStartup}>
            {creatingStartup && <span className="spinner" />}
            Create Startup
          </button>
        </form>
        {createdStartupId && <div className="badge">✓ Created startup #{createdStartupId}</div>}
      </div>

      {createdStartupId && (
        <div className="card">
          <h2>Simulate Next Month</h2>
          <form onSubmit={handleSimulate}>
            <div className="field-grid">
              <input className="field" type="number" value={simMarketingSpend} onChange={(e) => setSimMarketingSpend(e.target.value)} placeholder="Marketing spend" />
              <input className="field" type="number" value={simEmployeeCount} onChange={(e) => setSimEmployeeCount(e.target.value)} placeholder="Employee count" />
            </div>
            <button className="btn btn-primary" type="submit" disabled={simulating}>
              {simulating && <span className="spinner" />}
              Simulate Next Month
            </button>
          </form>

          {simResult && (
            <div className="result-box">
              <div className={`market-badge market-${simResult.market_condition}`}>
                {simResult.market_condition === 'booming' && '📈'}
                {simResult.market_condition === 'favorable' && '🙂'}
                {simResult.market_condition === 'cooling' && '🌥️'}
                {simResult.market_condition === 'recessionary' && '📉'}
                {' '}Market: {simResult.market_condition} ({simResult.market_multiplier}x growth)
              </div>
              <div className="result-stats">
                <div>
                  <div className="stat-label">Revenue</div>
                  <div className="stat-value">${simResult.revenue.toLocaleString()}</div>
                </div>
                <div>
                  <div className="stat-label">Cash on Hand</div>
                  <div className={`stat-value ${simResult.cash_on_hand < 0 ? 'stat-negative' : ''}`}>
                    ${simResult.cash_on_hand.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="stat-label">Customers</div>
                  <div className="stat-value">{simResult.customer_count}</div>
                  <div className="stat-delta">
                    <span className="stat-positive">+{simResult.customers_acquired}</span>
                    {' / '}
                    <span className="stat-negative">-{simResult.customers_churned}</span>
                  </div>
                </div>
              </div>
              <p className="narration">{simResult.narration}</p>
            </div>
          )}
        </div>
      )}

      {createdStartupId && (
        <div className="card">
          <h2>History</h2>
          <button className="btn btn-secondary btn-history" onClick={handleViewHistory} disabled={loadingHistory}>
            {loadingHistory && <span className="spinner" />}
            Refresh History
          </button>
          {history.length > 0 && (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
                <XAxis dataKey="month_number" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} label={{ value: 'Month', position: 'insideBottom', offset: -5, fontSize: 11, fill: 'var(--text-muted)' }} />
                <YAxis yAxisId="money" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <YAxis yAxisId="customers" orientation="right" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line yAxisId="money" type="monotone" dataKey="revenue" stroke="#6e7bff" strokeWidth={2} dot={false} />
                <Line yAxisId="money" type="monotone" dataKey="cash_on_hand" stroke="#f59e0b" strokeWidth={2} dot={false} />
                <Line yAxisId="customers" type="monotone" dataKey="customer_count" stroke="#4ade80" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
          <ul className="history-list">
            {history.map((snapshot) => (
              <li className="history-row" key={snapshot.id}>
                <span className="history-month">Month {snapshot.month_number}</span>
                <span>${snapshot.revenue.toLocaleString()} revenue</span>
                <span>${snapshot.cash_on_hand.toLocaleString()} cash</span>
                <span>{snapshot.customer_count} customers (+{snapshot.customers_acquired}/-{snapshot.customers_churned})</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
