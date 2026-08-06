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

const API_URL = 'http://127.0.0.1:8001'

interface FundraisingResult {
  attempted: boolean
  success_probability: number
  raised: boolean
  amount_raised: number
}

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
  investor_count: number
  funding_raised_to_date: number
  fundraising_result: FundraisingResult | null
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
  fundraising_model: { accuracy: number; precision: number; recall: number }
}

interface Strategy {
  rank: number
  price: number
  monthly_marketing: number
  employee_count: number
  survival_probability: number
  ending_cash_p10: number
  ending_cash_median: number
  ending_cash_p90: number
  ending_revenue_median: number
  ending_customers_median: number
}

interface StrategyLabResult {
  horizon_months: number
  simulations_per_strategy: number
  strategies_evaluated: number
  recommendation: string
  best_strategy: Strategy
  top_strategies: Strategy[]
}

interface DigitalTwinResult {
  forecast_horizon_months: number
  observed_months: number
  predictions: {
    future_revenue: number
    future_customer_count: number
    future_cash_on_hand: number
    revenue_growth: number
    cash_exhaustion_probability: number
  }
  model: {
    algorithm: string
    feature_count: number
    training_rows: number
    data_source: string
    metrics: Record<string, { mae: number; r2: number }>
  }
  data_coverage: {
    observed_signals: number
    total_signals: number
    coverage_percent: number
    warning: string
  }
}

interface AICeoResult {
  recommendation: { rank: number; action: string; action_label: string; long_term_value: number; explanation: string }
  alternatives: Array<{ rank: number; action: string; action_label: string; long_term_value: number; explanation: string }>
  projected_trajectory: Array<{ month: number; action: string; revenue: number; cash: number; customers: number; ownership: number; company_value: number }>
  policy: {
    algorithm: string
    training_episodes: number
    training_transitions: number
    policy: { survival_rate: number; median_company_value: number }
    random_baseline: { survival_rate: number; median_company_value: number }
  }
  limitations: string
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
  const [attemptFundraising, setAttemptFundraising] = useState(false)
  const [simResult, setSimResult] = useState<SimulationResult | null>(null)
  const [simulating, setSimulating] = useState(false)
  const [simulationError, setSimulationError] = useState('')

  const [history, setHistory] = useState<Snapshot[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)

  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null)
  const [strategyResult, setStrategyResult] = useState<StrategyLabResult | null>(null)
  const [analyzingStrategies, setAnalyzingStrategies] = useState(false)
  const [strategyError, setStrategyError] = useState('')
  const [digitalTwin, setDigitalTwin] = useState<DigitalTwinResult | null>(null)
  const [loadingTwin, setLoadingTwin] = useState(false)
  const [twinError, setTwinError] = useState('')
  const [aiCeo, setAiCeo] = useState<AICeoResult | null>(null)
  const [loadingCeo, setLoadingCeo] = useState(false)
  const [ceoError, setCeoError] = useState('')

  useEffect(() => {
    fetch(`${API_URL}/model-metrics`)
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
    const response = await fetch(`${API_URL}/login`, {
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

    const response = await fetch(`${API_URL}/register`, {
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

    const response = await fetch(`${API_URL}/guest-login`, { method: 'POST' })
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

    const response = await fetch(`${API_URL}/startups`, {
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
    setSimulationError('')
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 15000)
    try {
      const response = await fetch(
        `${API_URL}/startups/${createdStartupId}/simulate-next-month`,
        {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            marketing_spend: Number(simMarketingSpend),
            employee_count: Number(simEmployeeCount),
            attempt_fundraising: attemptFundraising,
          }),
        }
      )
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `Simulation failed (${response.status})`)
      }
      setSimResult(await response.json())
    } catch (error) {
      setSimulationError(
        error instanceof DOMException && error.name === 'AbortError'
          ? 'Simulation timed out. Please try again.'
          : error instanceof Error ? error.message : 'Simulation failed'
      )
    } finally {
      window.clearTimeout(timeout)
      setSimulating(false)
    }
  }

  async function handleViewHistory() {
    setLoadingHistory(true)
    const response = await fetch(`${API_URL}/startups/${createdStartupId}/snapshots`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    const data = await response.json()
    setHistory(data)
    setLoadingHistory(false)
  }

  async function handleAnalyzeStrategies() {
    setAnalyzingStrategies(true)
    setStrategyError('')
    try {
      const response = await fetch(`${API_URL}/startups/${createdStartupId}/strategy-lab`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ horizon_months: 12, simulations: 250 }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `Strategy analysis failed (${response.status})`)
      }
      setStrategyResult(await response.json())
    } catch (error) {
      setStrategyError(error instanceof Error ? error.message : 'Could not reach the strategy engine')
    } finally {
      setAnalyzingStrategies(false)
    }
  }

  async function handleRunDigitalTwin() {
    setLoadingTwin(true)
    setTwinError('')
    try {
      const response = await fetch(`${API_URL}/startups/${createdStartupId}/digital-twin`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `Digital Twin failed (${response.status})`)
      }
      setDigitalTwin(await response.json())
    } catch (error) {
      setTwinError(error instanceof Error ? error.message : 'Digital Twin failed')
    } finally {
      setLoadingTwin(false)
    }
  }

  async function handleAskAiCeo() {
    setLoadingCeo(true)
    setCeoError('')
    try {
      const response = await fetch(`${API_URL}/startups/${createdStartupId}/ai-ceo`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) throw new Error(`AI CEO failed (${response.status})`)
      setAiCeo(await response.json())
    } catch (error) {
      setCeoError(error instanceof Error ? error.message : 'AI CEO failed')
    } finally {
      setLoadingCeo(false)
    }
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
                <tr>
                  <td>fundraising (logistic)</td>
                  <td>{(modelMetrics.fundraising_model.accuracy * 100).toFixed(1)}%</td>
                  <td>{(modelMetrics.fundraising_model.precision * 100).toFixed(1)}%</td>
                  <td>{(modelMetrics.fundraising_model.recall * 100).toFixed(1)}%</td>
                </tr>
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
                  formatter={(v) => `${(Number(v ?? 0) * 100).toFixed(1)}%`}
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
        <div className="card ceo-card">
          <div className="section-heading">
            <div>
              <h2>AI CEO</h2>
              <p className="card-sub">A reinforcement-learning policy chooses what the company should do next.</p>
            </div>
            <button className="btn btn-secondary strategy-button" onClick={handleAskAiCeo} disabled={loadingCeo}>
              {loadingCeo && <span className="spinner" />}
              Ask AI CEO
            </button>
          </div>
          {ceoError && <p className="error-text">{ceoError}</p>}
          {aiCeo && (
            <div className="ceo-results">
              <div className="ceo-decision">
                <span className="recommendation-label">Next decision</span>
                <strong>{aiCeo.recommendation.action_label}</strong>
                <p>{aiCeo.recommendation.explanation}</p>
              </div>
              <div className="ceo-benchmark">
                <div><strong>{(aiCeo.policy.policy.survival_rate * 100).toFixed(1)}%</strong><span>AI survival</span></div>
                <div><strong>{(aiCeo.policy.random_baseline.survival_rate * 100).toFixed(1)}%</strong><span>Random baseline</span></div>
                <div><strong>{aiCeo.policy.training_transitions.toLocaleString()}</strong><span>Training decisions</span></div>
              </div>
              <p className="chart-title">Projected AI-controlled trajectory</p>
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={aiCeo.projected_trajectory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }} />
                  <Legend />
                  <Line type="monotone" dataKey="cash" stroke="#f59e0b" strokeWidth={2} />
                  <Line type="monotone" dataKey="company_value" stroke="#6e7bff" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
              <div className="ceo-actions">
                {aiCeo.alternatives.map((action) => <span key={action.action}>#{action.rank} {action.action_label}</span>)}
              </div>
              <p className="uncertainty-note">{aiCeo.limitations}</p>
            </div>
          )}
        </div>
      )}

      {createdStartupId && (
        <div className="card twin-card">
          <div className="section-heading">
            <div>
              <h2>Startup Digital Twin</h2>
              <p className="card-sub">One model forecasts the connected financial and customer state three months ahead.</p>
            </div>
            <button className="btn btn-secondary strategy-button" onClick={handleRunDigitalTwin} disabled={loadingTwin}>
              {loadingTwin && <span className="spinner" />}
              Run Digital Twin
            </button>
          </div>
          {twinError && <p className="error-text">{twinError}</p>}
          {digitalTwin && (
            <div className="twin-results">
              <div className="twin-stats">
                <div><span>Revenue in 3 months</span><strong>${digitalTwin.predictions.future_revenue.toLocaleString()}</strong></div>
                <div><span>Cash in 3 months</span><strong>${digitalTwin.predictions.future_cash_on_hand.toLocaleString()}</strong></div>
                <div><span>Customers in 3 months</span><strong>{digitalTwin.predictions.future_customer_count.toLocaleString()}</strong></div>
                <div><span>Cash-exhaustion risk</span><strong>{(digitalTwin.predictions.cash_exhaustion_probability * 100).toFixed(1)}%</strong></div>
              </div>
              <div className="model-provenance">
                <span>{digitalTwin.model.algorithm}</span>
                <span>{digitalTwin.model.feature_count.toLocaleString()} features</span>
                <span>{digitalTwin.model.training_rows.toLocaleString()} training examples</span>
                <span>{digitalTwin.observed_months} observed month(s)</span>
              </div>
              <div className="coverage-row">
                <div><i style={{ width: `${digitalTwin.data_coverage.coverage_percent}%` }} /></div>
                <span>Live data coverage: {digitalTwin.data_coverage.observed_signals}/{digitalTwin.data_coverage.total_signals} signals ({digitalTwin.data_coverage.coverage_percent}%)</span>
              </div>
              <p className="uncertainty-note">{digitalTwin.data_coverage.warning} Training source: {digitalTwin.model.data_source.replace(/_/g, ' ')}.</p>
            </div>
          )}
        </div>
      )}

      {createdStartupId && (
        <div className="card strategy-card">
          <div className="section-heading">
            <div>
              <h2>AI Strategy Lab</h2>
              <p className="card-sub">Stress-test pricing, marketing, and hiring across thousands of possible futures.</p>
            </div>
            <button className="btn btn-secondary strategy-button" onClick={handleAnalyzeStrategies} disabled={analyzingStrategies}>
              {analyzingStrategies && <span className="spinner" />}
              Analyze 12-month strategy
            </button>
          </div>
          {strategyError && <p className="error-text">{strategyError}. Restart the backend and try again.</p>}

          {strategyResult && (
            <div className="strategy-results">
              <div className="recommendation">
                <span className="recommendation-label">Recommended decision</span>
                <strong>{strategyResult.recommendation}</strong>
                <span>
                  Tested {strategyResult.strategies_evaluated} strategies with{' '}
                  {strategyResult.simulations_per_strategy.toLocaleString()} futures each.
                </span>
              </div>
              <div className="strategy-grid">
                {strategyResult.top_strategies.map((strategy) => (
                  <div className={`strategy-option ${strategy.rank === 1 ? 'strategy-winner' : ''}`} key={strategy.rank}>
                    <span className="strategy-rank">#{strategy.rank}</span>
                    <strong>${strategy.price}/customer</strong>
                    <span>${strategy.monthly_marketing.toLocaleString()} marketing · {strategy.employee_count} staff</span>
                    <div className="survival-meter">
                      <i style={{ width: `${strategy.survival_probability * 100}%` }} />
                    </div>
                    <span>{(strategy.survival_probability * 100).toFixed(0)}% survival · ${strategy.ending_cash_median.toLocaleString()} median cash</span>
                  </div>
                ))}
              </div>
              <p className="uncertainty-note">
                Best-case / expected / stress-case ending cash: ${strategyResult.best_strategy.ending_cash_p90.toLocaleString()} / ${strategyResult.best_strategy.ending_cash_median.toLocaleString()} / ${strategyResult.best_strategy.ending_cash_p10.toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}

      {createdStartupId && (
        <div className="card">
          <h2>Simulate Next Month</h2>
          <form onSubmit={handleSimulate}>
            <div className="field-grid">
              <input className="field" type="number" value={simMarketingSpend} onChange={(e) => setSimMarketingSpend(e.target.value)} placeholder="Marketing spend" />
              <input className="field" type="number" value={simEmployeeCount} onChange={(e) => setSimEmployeeCount(e.target.value)} placeholder="Employee count" />
            </div>
            <label className="checkbox-row">
              <input type="checkbox" checked={attemptFundraising} onChange={(e) => setAttemptFundraising(e.target.checked)} />
              Attempt to raise a funding round this month
            </label>
            <button className="btn btn-primary" type="submit" disabled={simulating}>
              {simulating && <span className="spinner" />}
              Simulate Next Month
            </button>
            {simulationError && <p className="error-text">{simulationError}</p>}
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

              {simResult.fundraising_result && (
                <div className={`fundraising-badge ${simResult.fundraising_result.raised ? 'fundraising-success' : 'fundraising-fail'}`}>
                  {simResult.fundraising_result.raised
                    ? `💰 Raised $${simResult.fundraising_result.amount_raised.toLocaleString()} (${(simResult.fundraising_result.success_probability * 100).toFixed(0)}% model confidence)`
                    : `❌ Fundraising attempt failed (${(simResult.fundraising_result.success_probability * 100).toFixed(0)}% model confidence)`}
                </div>
              )}

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
