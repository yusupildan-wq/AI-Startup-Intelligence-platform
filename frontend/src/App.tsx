import { useEffect, useState, type SubmitEvent } from 'react'
import Lenis from 'lenis'
import {
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
  digital_twin: {
    feature_count: number; training_rows: number; held_out_companies: number; data_source: string
    future_revenue: { r2: number; mae: number }
    future_customer_count: { r2: number; mae: number }
    future_cash_on_hand: { r2: number; mae: number }
  }
  ai_ceo: {
    algorithm: string; training_transitions: number; data_source: string
    policy: { survival_rate: number; median_company_value: number }
    random_baseline: { survival_rate: number; median_company_value: number }
  }
  population_models: {
    data_source: string; generated_rows_per_model: number
    customer_choice: { roc_auc: number; balanced_accuracy: number; majority_baseline_accuracy: number }
    employee_attrition: { roc_auc: number; balanced_accuracy: number; majority_baseline_accuracy: number }
    product_adoption: { roc_auc: number; balanced_accuracy: number; majority_baseline_accuracy: number }
  }
  economy_agents: {
    rows_per_system: number; data_source: string
    investor: { roc_auc: number; amount_r2: number }
    competitor_policy: { accuracy: number; balanced_accuracy: number; majority_baseline: number }
    macro_regime: { accuracy: number; balanced_accuracy: number; majority_baseline: number }
  }
  world_generator: {
    algorithm: string; training_worlds: number; held_out_worlds: number; feature_count: number
    real_vs_generated_auc: number; generated_validity_rate_before_constraints: number; standardized_diversity: number
  }
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
  current_state: { month: number; cash: number; customers: number; price: number; marketing: number; employees: number }
}

interface WorldCompany {
  id: string; name: string; cash: number; customers: number; price: number; marketing: number
  engineers: number; salespeople: number; support: number; product_quality: number
  technical_debt: number; reputation: number; founder_ownership: number; revenue: number; alive: boolean
  customers_acquired: number; customers_churned: number; employees_departed: number; product_adoption_rate: number
  last_action: string; last_funding_probability: number; last_funding_raised: number
}

interface CivilizationWorld {
  id: string; name: string; seed: number; month: number; branch_id: string; parent_branch_id: string | null
  companies: Record<string, WorldCompany>
  segments: Record<string, { id: string; name: string; population: number; budget: number }>
  investors: { available_capital: number; risk_appetite: number; valuation_multiple: number }
  macro: { regime: string; demand_multiplier: number; interest_rate: number; unemployment_rate: number; venture_sentiment: number }
}

interface WorldEvent { id: string; month: number; type: string; actor_id: string; payload: Record<string, unknown> }
interface WorldBranch { id: string; parent_branch_id: string | null; fork_month: number; name: string; current_month: number }

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
  const [world, setWorld] = useState<CivilizationWorld | null>(null)
  const [worldEvents, setWorldEvents] = useState<WorldEvent[]>([])
  const [worldBranches, setWorldBranches] = useState<WorldBranch[]>([])
  const [worldAction, setWorldAction] = useState('hold')
  const [worldShock, setWorldShock] = useState('')
  const [worldLoading, setWorldLoading] = useState(false)
  const [worldError, setWorldError] = useState('')
  const [generationScenario, setGenerationScenario] = useState('balanced')
  const [strategyResult, setStrategyResult] = useState<StrategyLabResult | null>(null)
  const [analyzingStrategies, setAnalyzingStrategies] = useState(false)
  const [strategyError, setStrategyError] = useState('')
  const [digitalTwin, setDigitalTwin] = useState<DigitalTwinResult | null>(null)
  const [loadingTwin, setLoadingTwin] = useState(false)
  const [twinError, setTwinError] = useState('')
  const [aiCeo, setAiCeo] = useState<AICeoResult | null>(null)
  const [loadingCeo, setLoadingCeo] = useState(false)
  const [ceoError, setCeoError] = useState('')
  const [executingCeo, setExecutingCeo] = useState(false)
  const [executedDecision, setExecutedDecision] = useState('')

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

  async function handleExecuteAiCeo() {
    setExecutingCeo(true)
    setCeoError('')
    setExecutedDecision('')
    try {
      const response = await fetch(`${API_URL}/startups/${createdStartupId}/ai-ceo/execute`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `AI CEO execution failed (${response.status})`)
      }
      const data = await response.json()
      setExecutedDecision(data.executed_decision.action_label)
      setSimResult(data.simulation)
      await handleViewHistory()
      await handleAskAiCeo()
    } catch (error) {
      setCeoError(error instanceof Error ? error.message : 'AI CEO execution failed')
    } finally {
      setExecutingCeo(false)
    }
  }

  async function refreshWorldNavigation(worldId: string, branchId: string) {
    const headers = { Authorization: `Bearer ${token}` }
    const [eventsResponse, branchesResponse] = await Promise.all([
      fetch(`${API_URL}/worlds/${worldId}/branches/${branchId}/events`, { headers }),
      fetch(`${API_URL}/worlds/${worldId}/branches`, { headers }),
    ])
    if (eventsResponse.ok) setWorldEvents(await eventsResponse.json())
    if (branchesResponse.ok) setWorldBranches(await branchesResponse.json())
  }

  async function handleCreateWorld() {
    setWorldLoading(true); setWorldError('')
    try {
      const response = await fetch(`${API_URL}/worlds`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ name: name || 'Startup Civilization', seed: Date.now() % 2147483647, startup_id: createdStartupId, generator: 'learned', scenario: generationScenario }),
      })
      if (!response.ok) throw new Error(`World creation failed (${response.status})`)
      const created = await response.json()
      setWorld(created); setWorldEvents([])
      await refreshWorldNavigation(created.id, created.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'World creation failed') }
    finally { setWorldLoading(false) }
  }

  async function handleAdvanceWorld() {
    if (!world) return
    setWorldLoading(true); setWorldError('')
    try {
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/advance`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ action: worldAction, shock: worldShock || null }),
      })
      if (!response.ok) throw new Error(`World advance failed (${response.status})`)
      const result = await response.json()
      setWorld(result.state); setWorldShock('')
      await refreshWorldNavigation(result.state.id, result.state.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'World advance failed') }
    finally { setWorldLoading(false) }
  }

  async function handleBranchWorld() {
    if (!world) return
    setWorldLoading(true); setWorldError('')
    const branchName = `branch-month-${world.month}-${worldBranches.length + 1}`
    try {
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/branch`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ from_month: world.month, name: branchName }),
      })
      if (!response.ok) throw new Error(`Timeline fork failed (${response.status})`)
      const created = await response.json()
      setWorld(created); setWorldEvents([])
      await refreshWorldNavigation(created.id, created.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Timeline fork failed') }
    finally { setWorldLoading(false) }
  }

  async function handleSwitchBranch(branchId: string) {
    if (!world) return
    const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${branchId}`, { headers: { Authorization: `Bearer ${token}` } })
    if (response.ok) {
      const selected = await response.json(); setWorld(selected)
      await refreshWorldNavigation(selected.id, selected.branch_id)
    }
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
            <h2>Machine Learning Systems</h2>
            <p className="card-sub">The trained systems currently powering the startup simulator.</p>
            <div className="ml-system-list">
              <div className="ml-system">
                <div className="ml-system-head"><strong>Startup Digital Twin</strong><span>ACTIVE · SYNTHETIC V1</span></div>
                <p>Forecasts revenue, customers, cash, growth, and cash-exhaustion risk together.</p>
                <div className="ml-facts">
                  <span>{modelMetrics.digital_twin.feature_count.toLocaleString()} features</span>
                  <span>{modelMetrics.digital_twin.training_rows.toLocaleString()} training states</span>
                  <span>{modelMetrics.digital_twin.held_out_companies} held-out companies</span>
                </div>
                <div className="metric-strip">
                  <span>Revenue R² <b>{modelMetrics.digital_twin.future_revenue.r2.toFixed(2)}</b></span>
                  <span>Customers R² <b>{modelMetrics.digital_twin.future_customer_count.r2.toFixed(2)}</b></span>
                  <span>Cash R² <b>{modelMetrics.digital_twin.future_cash_on_hand.r2.toFixed(2)}</b></span>
                </div>
              </div>
              <div className="ml-system">
                <div className="ml-system-head"><strong>Generative World Model</strong><span>ACTIVE · PROBABILISTIC GENERATOR</span></div>
                <p>Samples complete civilizations with correlated companies, markets, investors, customer segments, and economies.</p>
                <div className="ml-facts">
                  <span>{modelMetrics.world_generator.training_worlds.toLocaleString()} training worlds</span>
                  <span>{modelMetrics.world_generator.feature_count} generated dimensions</span>
                  <span>{(modelMetrics.world_generator.generated_validity_rate_before_constraints * 100).toFixed(0)}% raw validity</span>
                  <span>Discriminator AUC {modelMetrics.world_generator.real_vs_generated_auc.toFixed(2)}</span>
                </div>
              </div>
              <div className="ml-system">
                <div className="ml-system-head"><strong>Economic Agent Models</strong><span>ACTIVE · INVESTORS + RIVALS + MACRO</span></div>
                <p>Investors price funding offers, competitors select strategies, and the economy changes regimes.</p>
                <div className="ml-facts">
                  <span>Investor AUC {modelMetrics.economy_agents.investor.roc_auc.toFixed(2)}</span>
                  <span>Valuation R² {modelMetrics.economy_agents.investor.amount_r2.toFixed(2)}</span>
                  <span>Competitor balanced accuracy {modelMetrics.economy_agents.competitor_policy.balanced_accuracy.toFixed(2)}</span>
                  <span>Macro balanced accuracy {modelMetrics.economy_agents.macro_regime.balanced_accuracy.toFixed(2)}</span>
                </div>
              </div>
              <div className="ml-system">
                <div className="ml-system-head"><strong>Behavioral Population Models</strong><span>ACTIVE · 3 LEARNED SYSTEMS</span></div>
                <p>Customers choose products, employees leave or stay, and segments adopt new product value.</p>
                <div className="ml-facts">
                  <span>{modelMetrics.population_models.generated_rows_per_model.toLocaleString()} examples / model</span>
                  <span>Choice AUC {modelMetrics.population_models.customer_choice.roc_auc.toFixed(2)}</span>
                  <span>Attrition AUC {modelMetrics.population_models.employee_attrition.roc_auc.toFixed(2)}</span>
                  <span>Adoption AUC {modelMetrics.population_models.product_adoption.roc_auc.toFixed(2)}</span>
                </div>
              </div>
              <div className="ml-system">
                <div className="ml-system-head"><strong>AI CEO Policy</strong><span>ACTIVE · REINFORCEMENT LEARNING</span></div>
                <p>Chooses and executes pricing, marketing, hiring, product, market, and funding actions.</p>
                <div className="ml-facts">
                  <span>{modelMetrics.ai_ceo.training_transitions.toLocaleString()} learned transitions</span>
                  <span>{(modelMetrics.ai_ceo.policy.survival_rate * 100).toFixed(1)}% policy survival</span>
                  <span>{(modelMetrics.ai_ceo.random_baseline.survival_rate * 100).toFixed(1)}% random baseline</span>
                </div>
              </div>
            </div>
            <div className="data-honesty">
              <strong>Data status</strong>
              <span>Both primary systems are trained in synthetic environments. These metrics measure performance inside those environments—not proven real-world accuracy.</span>
            </div>
            <details className="legacy-models">
              <summary>Legacy baseline models</summary>
              <p>Small synthetic churn, growth, and fundraising models remain in the monthly engine while their responsibilities are migrated into the Digital Twin.</p>
            </details>
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

      <div className="card civilization-card">
        <div className="section-heading">
          <div>
            <h2>Startup Civilization</h2>
            <p className="card-sub">Operate one company inside a persistent world of competitors, customers, investors, and economic regimes.</p>
          </div>
          {!world && <div className="world-generator-controls">
            <select className="field" value={generationScenario} onChange={(e) => setGenerationScenario(e.target.value)}>
              <option value="balanced">Generated balanced world</option><option value="recession">Generated recession</option>
              <option value="funding_boom">Generated funding boom</option><option value="technology_shift">Generated technology shift</option>
            </select>
            <button className="btn btn-primary world-launch" onClick={handleCreateWorld} disabled={worldLoading}>{worldLoading && <span className="spinner" />}Generate World</button>
          </div>}
        </div>
        {worldError && <p className="error-text">{worldError}</p>}
        {world && (
          <div className="world-control-room">
            <div className="world-status">
              <span>Month <b>{world.month}</b></span><span>Branch <b>{world.branch_id}</b></span>
              <span>Economy <b>{world.macro.regime}</b></span><span>Demand <b>{world.macro.demand_multiplier.toFixed(2)}×</b></span>
              <span>Investor capital <b>${(world.investors.available_capital / 1e6).toFixed(1)}M</b></span>
            </div>
            <div className="world-companies">
              {Object.values(world.companies).map((company) => (
                <div className={`world-company ${company.id === 'player' ? 'world-player' : ''}`} key={company.id}>
                  <div><strong>{company.name}</strong><span>{company.alive ? 'ACTIVE' : 'FAILED'}</span></div>
                  <p>${company.revenue.toLocaleString()} revenue · ${company.cash.toLocaleString()} cash</p>
                  <p>{company.customers} customers · ${company.price.toFixed(0)} price · {(company.product_quality * 100).toFixed(0)}% quality</p>
                  <p className="population-flow">+{company.customers_acquired} acquired · −{company.customers_churned} churned · {company.employees_departed} staff left · {(company.product_adoption_rate * 100).toFixed(0)}% adoption</p>
                  <p>Last action: <b>{company.last_action.replace(/_/g, ' ')}</b>{company.last_funding_raised > 0 ? ` · raised $${company.last_funding_raised.toLocaleString()}` : ''}</p>
                </div>
              ))}
            </div>
            <div className="world-controls">
              <select className="field" value={worldAction} onChange={(e) => setWorldAction(e.target.value)}>
                {['hold','raise_price','lower_price','increase_marketing','decrease_marketing','hire_engineer','hire_sales','hire_support','reduce_headcount','fundraise','invest_in_product','enter_new_market'].map((action) => <option key={action} value={action}>{action.replace(/_/g, ' ')}</option>)}
              </select>
              <select className="field" value={worldShock} onChange={(e) => setWorldShock(e.target.value)}>
                <option value="">No forced shock</option><option value="recession">Recession</option><option value="funding_boom">Funding boom</option><option value="demand_surge">Demand surge</option><option value="technology_shift">Technology shift</option>
              </select>
              <button className="btn btn-primary" onClick={handleAdvanceWorld} disabled={worldLoading}>{worldLoading && <span className="spinner" />}Advance Month</button>
              <button className="btn btn-secondary" onClick={handleBranchWorld} disabled={worldLoading}>Fork Timeline</button>
            </div>
            <div className="branch-tabs">
              {worldBranches.map((branch) => <button className={branch.id === world.branch_id ? 'active' : ''} onClick={() => handleSwitchBranch(branch.id)} key={branch.id}>{branch.name} · M{branch.current_month}</button>)}
            </div>
            <details className="world-event-log">
              <summary>Event log ({worldEvents.length})</summary>
              {worldEvents.slice(-12).reverse().map((event) => <div key={event.id}><b>M{event.month}</b> {event.actor_id}: {event.type.replace(/_/g, ' ')}</div>)}
            </details>
          </div>
        )}
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
                <button className="btn btn-primary execute-ceo" onClick={handleExecuteAiCeo} disabled={executingCeo}>
                  {executingCeo && <span className="spinner" />}
                  Authorize AI CEO and advance one month
                </button>
              </div>
              {executedDecision && <div className="execution-success">Executed {executedDecision}. The startup state and history have been updated.</div>}
              <div className="ceo-benchmark">
                <div><strong>Month {aiCeo.current_state.month}</strong><span>Current month</span></div>
                <div><strong>${aiCeo.current_state.cash.toLocaleString()}</strong><span>Current cash</span></div>
                <div><strong>{aiCeo.current_state.customers.toLocaleString()}</strong><span>Current customers</span></div>
                <div><strong>${aiCeo.current_state.price.toLocaleString()}</strong><span>Current price</span></div>
                <div><strong>${aiCeo.current_state.marketing.toLocaleString()}</strong><span>Marketing / month</span></div>
                <div><strong>{aiCeo.current_state.employees}</strong><span>Current employees</span></div>
              </div>
              <p className="model-benchmark-note">
                Fixed model benchmark: {(aiCeo.policy.policy.survival_rate * 100).toFixed(1)}% policy survival vs{' '}
                {(aiCeo.policy.random_baseline.survival_rate * 100).toFixed(1)}% random across{' '}
                {aiCeo.policy.training_transitions.toLocaleString()} training decisions.
              </p>
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
