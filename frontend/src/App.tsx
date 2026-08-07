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
  trajectory_model: {
    algorithm: string; training_worlds: number; held_out_worlds: number; training_transitions: number
    state_dimensions: number; action_count: number; interval_80_coverage: number
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
interface GeneratedFutures {
  action: string; horizon: number; paths: number; limitations: string
  timeline: Array<{ month: number; cash_p10: number; cash_median: number; cash_p90: number; customers_median: number; revenue_median: number; survival_probability: number }>
}

interface DatasetImport {
  id: number; source: string; dataset_name: string; source_url: string | null
  content_sha256: string; row_count: number; imported_at: string
}

interface RegistryModel {
  model_id: string; version: string; status: string; artifact_bytes: number
  artifact_sha256: string; training_code: string; training_code_sha256: string
  reproduce_command: string; data_lineage: { source: string; kind: string }
  limitations: string
}

interface CausalEffects {
  effects: Array<{ action: string; effects_vs_hold: { cash: number; customers: number; revenue: number; product_quality: number } }>
  model: { training_states: number; held_out_states: number; training_counterfactuals: number; outcomes: Record<string, { treatment_effect_mae_vs_hold: number }> }
  limitations: string
}

interface ModelBasedPlan {
  recommendation: PlannedAction
  action_comparison: PlannedAction[]
  search: { horizon: number; actions: number; beam_width_per_first_action: number; stochastic_paths_per_plan: number; risk_aversion: number }
  limitations: string
}

interface PlannedAction {
  rank: number; first_action: string; planned_sequence: string[]; risk_adjusted_score: number
  survival_probability: number; cash_p10: number; cash_median: number; revenue_median: number; customers_median: number
}

interface HumanAiComparison {
  fork_month: number; human_action: string; ai_action: string
  human_branch: CivilizationWorld; ai_branch: CivilizationWorld; ai_plan: ModelBasedPlan
}

type HelpTopic = {
  title: string
  category: string
  summary: string
  uses: string
  how: string[]
  output: string
}

const SECTION_HELP: Record<string, HelpTopic> = {
  models: { title: 'Machine Learning Systems', category: 'Model overview', summary: 'An inventory of the trained systems currently available to the backend.', uses: 'Saved ML artifacts and held-out evaluation metrics.', how: ['Shows which learned systems are active.', 'Displays training scale and evaluation results.', 'Does not run or retrain a model by itself.'], output: 'A technical overview for understanding what powers the application.' },
  civilization: { title: 'Startup Civilization', category: 'Primary simulation', summary: 'A persistent multi-agent world containing your startup, rivals, customers, employees, investors, products, and the economy.', uses: 'Event-driven simulation plus learned behavioral, economic, generative, causal, and planning models.', how: ['Choose an action and optional external shock.', 'Advance one month to resolve the entire world.', 'Fork timelines, generate futures, compare causal effects, or ask the model-based CEO.'], output: 'A saved world state, event history, branch tree, and month-by-month company outcomes.' },
  data: { title: 'Real Data Lab', category: 'Data infrastructure', summary: 'Imports official economic/company evidence or your own historical time-series data.', uses: 'FRED, US Census BDS, SEC Company Facts, PostgreSQL, and immutable SHA-256 manifests.', how: ['Select an official connector or paste a longitudinal CSV.', 'The backend normalizes every observation.', 'A content hash and provenance manifest are saved.'], output: 'Versioned datasets that can later support calibration, validation, and retraining.' },
  registry: { title: 'Experiment & Model Registry', category: 'Model governance', summary: 'The audit trail for every saved trained model—not a simulation control.', uses: 'Artifact hashes, training-code hashes, metrics files, lineage metadata, and reproduction commands.', how: ['Open any model card.', 'Inspect exactly what data and code produced it.', 'Use the command to reproduce its training run.'], output: 'Evidence that each displayed model is a real versioned artifact and has not silently changed.' },
  startup: { title: 'Create a Startup', category: 'Company setup', summary: 'Creates the single-company record used by the original monthly simulator, Digital Twin, Strategy Lab, and RL AI CEO.', uses: 'PostgreSQL application data; this step is ordinary software rather than ML.', how: ['Enter the company’s starting conditions.', 'Create the record to receive a startup ID.', 'Run monthly simulations to build its history.'], output: 'A persistent startup that can also be inserted as the player inside a civilization.' },
  ceo: { title: 'AI CEO', category: 'Decision intelligence', summary: 'The original reinforcement-learning policy that ranks the next startup action from the current company state.', uses: 'Offline fitted Q-iteration trained on synthetic startup transitions.', how: ['Ask the policy for a recommendation.', 'Review its explanation and alternatives.', 'Authorize the action to execute one month.'], output: 'A recommended action, policy values, alternatives, and projected AI-controlled trajectory.' },
  twin: { title: 'Startup Digital Twin', category: 'Predictive intelligence', summary: 'Forecasts several connected startup outcomes three months ahead from one shared feature representation.', uses: 'A trained 2,064-feature multi-output Extra Trees model.', how: ['Collect startup snapshots by simulating months.', 'Run the twin using all currently observed history.', 'Check the data-coverage warning before trusting the result.'], output: 'Forecast revenue, cash, customers, growth, and cash-exhaustion risk.' },
  strategy: { title: 'AI Strategy Lab', category: 'Decision intelligence', summary: 'Searches pricing, marketing, and staffing configurations across uncertain future simulations.', uses: 'Monte Carlo simulation and trained startup response models.', how: ['Run a 12-month strategy analysis.', 'The engine evaluates many configurations repeatedly.', 'Compare survival and downside—not only the best median result.'], output: 'Ranked strategies with cash ranges, revenue, customers, and survival probability.' },
  month: { title: 'Simulate Next Month', category: 'Monthly operations', summary: 'Runs one month of the original single-company simulator.', uses: 'Deterministic business logic, trained response models, randomness, and OpenAI for the final written narration.', how: ['Choose marketing, staffing, and whether to fundraise.', 'The backend calculates operating outcomes.', 'OpenAI explains the completed numerical result in plain language.'], output: 'Revenue, costs, burn, runway, acquisition, churn, funding results, and executive narration.' },
  history: { title: 'History', category: 'Saved evidence', summary: 'Displays the startup snapshots that were actually saved after monthly simulations.', uses: 'PostgreSQL snapshot records and frontend charting; it is not another predictive model.', how: ['Run at least one monthly simulation.', 'Load history.', 'Compare the recorded trajectory over time.'], output: 'A historical chart and auditable month-by-month operating record.' },
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
  const [world, setWorld] = useState<CivilizationWorld | null>(null)
  const [worldEvents, setWorldEvents] = useState<WorldEvent[]>([])
  const [worldBranches, setWorldBranches] = useState<WorldBranch[]>([])
  const [worldAction, setWorldAction] = useState('hold')
  const [worldShock, setWorldShock] = useState('')
  const [worldLoading, setWorldLoading] = useState(false)
  const [worldError, setWorldError] = useState('')
  const [generationScenario, setGenerationScenario] = useState('balanced')
  const [generatedFutures, setGeneratedFutures] = useState<GeneratedFutures | null>(null)
  const [generatingFutures, setGeneratingFutures] = useState(false)
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
  const [datasets, setDatasets] = useState<DatasetImport[]>([])
  const [dataLoading, setDataLoading] = useState('')
  const [dataError, setDataError] = useState('')
  const [secCik, setSecCik] = useState('0000320193')
  const [csvName, setCsvName] = useState('My startup observations')
  const [csvText, setCsvText] = useState('date,series,value,unit,entity\n2026-01,MRR,12000,USD,My Startup')
  const [registry, setRegistry] = useState<RegistryModel[]>([])
  const [causalEffects, setCausalEffects] = useState<CausalEffects | null>(null)
  const [modelPlan, setModelPlan] = useState<ModelBasedPlan | null>(null)
  const [decisionLoading, setDecisionLoading] = useState('')
  const [riskAversion, setRiskAversion] = useState(0.65)
  const [humanAi, setHumanAi] = useState<HumanAiComparison | null>(null)
  const [replayMonth, setReplayMonth] = useState(0)
  const [replayState, setReplayState] = useState<CivilizationWorld | null>(null)
  const [helpTopic, setHelpTopic] = useState<string | null>(null)
  const [activeWorkspace, setActiveWorkspace] = useState<'world' | 'intelligence' | 'data' | 'operations'>('world')

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
    if (!token) return
    const headers = { Authorization: `Bearer ${token}` }
    Promise.all([
      fetch(`${API_URL}/datasets`, { headers }).then((res) => res.ok ? res.json() : []),
      fetch(`${API_URL}/ml/registry`, { headers }).then((res) => res.ok ? res.json() : []),
    ]).then(([loadedDatasets, loadedRegistry]) => {
      setDatasets(loadedDatasets); setRegistry(loadedRegistry)
    })
  }, [token])

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
      setWorld(created); setWorldEvents([]); setGeneratedFutures(null); setCausalEffects(null); setModelPlan(null); setHumanAi(null)
      await refreshWorldNavigation(created.id, created.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'World creation failed') }
    finally { setWorldLoading(false) }
  }

  function handleNewCivilization() {
    setWorld(null)
    setWorldEvents([])
    setWorldBranches([])
    setGeneratedFutures(null)
    setCausalEffects(null)
    setModelPlan(null)
    setHumanAi(null)
    setReplayState(null)
    setWorldError('')
    setWorldAction('hold')
    setWorldShock('')
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
      setWorld(result.state); setWorldShock(''); setGeneratedFutures(null); setCausalEffects(null); setModelPlan(null); setHumanAi(null); setReplayState(null)
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
      setWorld(created); setWorldEvents([]); setGeneratedFutures(null); setCausalEffects(null); setModelPlan(null); setHumanAi(null); setReplayState(null)
      await refreshWorldNavigation(created.id, created.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Timeline fork failed') }
    finally { setWorldLoading(false) }
  }

  async function handleSwitchBranch(branchId: string) {
    if (!world) return
    const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${branchId}`, { headers: { Authorization: `Bearer ${token}` } })
    if (response.ok) {
      const selected = await response.json(); setWorld(selected); setReplayState(null); setHumanAi(null); setGeneratedFutures(null); setCausalEffects(null); setModelPlan(null)
      await refreshWorldNavigation(selected.id, selected.branch_id)
    }
  }

  async function handleGenerateFutures() {
    if (!world) return
    setGeneratingFutures(true); setWorldError('')
    try {
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/generate-trajectories`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ action: worldAction, horizon: 12, paths: 150, seed: Date.now() % 2147483647 }),
      })
      if (!response.ok) throw new Error(`Future generation failed (${response.status})`)
      setGeneratedFutures(await response.json())
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Future generation failed') }
    finally { setGeneratingFutures(false) }
  }

  async function refreshDatasets() {
    const response = await fetch(`${API_URL}/datasets`, { headers: { Authorization: `Bearer ${token}` } })
    if (response.ok) setDatasets(await response.json())
  }

  async function importOfficialData(source: 'fred' | 'census_bds' | 'sec_companyfacts') {
    setDataLoading(source); setDataError('')
    try {
      const response = await fetch(`${API_URL}/datasets/import/official`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ source, cik: source === 'sec_companyfacts' ? secCik : null, start_year: 2015 }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `Import failed (${response.status})`)
      }
      await refreshDatasets()
    } catch (error) { setDataError(error instanceof Error ? error.message : 'Import failed') }
    finally { setDataLoading('') }
  }

  async function importCsvData() {
    setDataLoading('csv'); setDataError('')
    try {
      const response = await fetch(`${API_URL}/datasets/import/csv`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ dataset_name: csvName, csv_text: csvText }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail || `CSV import failed (${response.status})`)
      }
      await refreshDatasets()
    } catch (error) { setDataError(error instanceof Error ? error.message : 'CSV import failed') }
    finally { setDataLoading('') }
  }

  async function handleCausalAnalysis() {
    if (!world) return
    setDecisionLoading('causal'); setWorldError('')
    try {
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/causal-effects`, { headers: { Authorization: `Bearer ${token}` } })
      if (!response.ok) throw new Error(`Causal analysis failed (${response.status})`)
      setCausalEffects(await response.json())
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Causal analysis failed') }
    finally { setDecisionLoading('') }
  }

  async function handleModelBasedPlan() {
    if (!world) return
    setDecisionLoading('planner'); setWorldError('')
    try {
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/model-based-ceo`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ horizon: 12, beam_width: 10, paths: 60, risk_aversion: riskAversion, seed: Date.now() % 2147483647 }),
      })
      if (!response.ok) throw new Error(`Model-based planning failed (${response.status})`)
      setModelPlan(await response.json())
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Model-based planning failed') }
    finally { setDecisionLoading('') }
  }

  async function executePlannedAction() {
    if (!world || !modelPlan) return
    setWorldLoading(true); setWorldError('')
    try {
      const action = modelPlan.recommendation.first_action
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/advance`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ action, shock: null }),
      })
      if (!response.ok) throw new Error(`Plan execution failed (${response.status})`)
      const result = await response.json(); setWorld(result.state); setWorldAction(action)
      setModelPlan(null); setCausalEffects(null); setGeneratedFutures(null); setHumanAi(null); setReplayState(null)
      await refreshWorldNavigation(result.state.id, result.state.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Plan execution failed') }
    finally { setWorldLoading(false) }
  }

  async function compareHumanWithAi() {
    if (!world) return
    setDecisionLoading('compare'); setWorldError('')
    try {
      const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/compare-human-ai`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ human_action: worldAction, risk_aversion: riskAversion, seed: Date.now() % 2147483647 }),
      })
      if (!response.ok) throw new Error(`Human vs AI experiment failed (${response.status})`)
      const result = await response.json(); setHumanAi(result)
      await refreshWorldNavigation(world.id, world.branch_id)
    } catch (error) { setWorldError(error instanceof Error ? error.message : 'Human vs AI experiment failed') }
    finally { setDecisionLoading('') }
  }

  async function inspectReplayMonth(month: number) {
    if (!world) return
    setReplayMonth(month)
    const response = await fetch(`${API_URL}/worlds/${world.id}/branches/${world.branch_id}/replay/${month}`, { headers: { Authorization: `Bearer ${token}` } })
    if (response.ok) setReplayState(await response.json())
  }

  function helpButton(topic: string) {
    return <button className="section-help-button" type="button" onClick={() => setHelpTopic(topic)}><span>?</span> What is this?</button>
  }

  const helpOverlay = helpTopic && SECTION_HELP[helpTopic] ? (
    <div className="help-overlay" role="presentation" onMouseDown={() => setHelpTopic(null)}>
      <section className="help-panel" role="dialog" aria-modal="true" aria-labelledby="help-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="help-panel-top"><span>{SECTION_HELP[helpTopic].category}</span><button type="button" onClick={() => setHelpTopic(null)} aria-label="Close explanation">×</button></div>
        <h2 id="help-title">{SECTION_HELP[helpTopic].title}</h2>
        <p className="help-summary">{SECTION_HELP[helpTopic].summary}</p>
        <div className="help-engine"><span>What powers it</span><p>{SECTION_HELP[helpTopic].uses}</p></div>
        <div className="help-steps"><span>What happens</span><ol>{SECTION_HELP[helpTopic].how.map((step) => <li key={step}>{step}</li>)}</ol></div>
        <div className="help-output"><span>What you get</span><p>{SECTION_HELP[helpTopic].output}</p></div>
      </section>
    </div>
  ) : null

  const themeToggle = (
    <button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
      <span className="theme-indicator" />{theme === 'dark' ? 'Light mode' : 'Dark mode'}
    </button>
  )

  if (!token) {
    return (
      <div className="auth-screen">
        {themeToggle}
        <div className="auth-brand"><span>S/01</span><h1>Startup<br />Civilization<br />Lab</h1></div>
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
          <div className="card metrics-card section-tone-governance">
            <div className="section-title-row"><h2>Machine Learning Systems</h2>{helpButton('models')}</div>
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
                <div className="ml-system-head"><strong>Generative Trajectory Model</strong><span>ACTIVE · ACTION-CONDITIONED</span></div>
                <p>Recursively samples correlated multi-month futures from the current world and selected decision.</p>
                <div className="ml-facts">
                  <span>{modelMetrics.trajectory_model.training_transitions.toLocaleString()} transitions</span>
                  <span>{modelMetrics.trajectory_model.state_dimensions} state dimensions</span>
                  <span>{modelMetrics.trajectory_model.action_count} actions</span>
                  <span>{(modelMetrics.trajectory_model.interval_80_coverage * 100).toFixed(1)}% interval coverage</span>
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
        {helpOverlay}
      </div>
    )
  }

  return (
    <div className="app" data-workspace={activeWorkspace}>
      <div className="topbar">
        <div className="brand-lockup"><span className="brand-mark">S/01</span><h1>Startup<br />Civilization Lab</h1><em>Experimental operating system</em></div>
        <div className="topbar-actions">
          {themeToggle}
          <button className="theme-toggle" onClick={handleLogout}>Log Out</button>
        </div>
      </div>

      <div className="section-legend" aria-label="Section color guide">
        <span className="legend-simulation">Simulation</span><span className="legend-decision">Decisions</span>
        <span className="legend-predictive">Prediction</span><span className="legend-data">Real data</span>
        <span className="legend-governance">Model governance</span><span className="legend-operations">Operations</span>
        <span className="legend-history">History</span>
      </div>

      <nav className="workspace-nav" aria-label="Application workspaces">
        <button className={`world-nav-button ${activeWorkspace === 'world' ? 'active' : ''}`} onClick={() => setActiveWorkspace('world')}>
          <span>A</span><strong>World Simulation</strong><small>Standalone civilization environment</small>
        </button>
        <div className="company-system-nav">
          <div className="company-system-label"><span>Connected workflow</span><strong>Company System</strong><small>These three workspaces share one startup record</small></div>
          <div className="company-system-steps">
            <button className={activeWorkspace === 'operations' ? 'active' : ''} onClick={() => setActiveWorkspace('operations')}>
              <span>01</span><strong>Operations</strong><small>Create and simulate</small>
            </button>
            <i>→</i>
            <button className={activeWorkspace === 'intelligence' ? 'active' : ''} onClick={() => setActiveWorkspace('intelligence')}>
              <span>02</span><strong>AI Intelligence</strong><small>{createdStartupId ? 'Ready to analyze' : 'Requires a startup'}</small>
            </button>
            <i>→</i>
            <button className={activeWorkspace === 'data' ? 'active' : ''} onClick={() => setActiveWorkspace('data')}>
              <span>03</span><strong>Data & Models</strong><small>Evidence and lineage</small>
            </button>
          </div>
        </div>
      </nav>

      <section className="workspace-intro">
        {activeWorkspace === 'world' && <><span>Standalone system</span><h2>World Simulation</h2><p>Create a civilization, choose one company action, advance time, and compare alternate timelines. This environment is separate from the connected company workflow.</p><div><b>Flow</b> Generate world → read conditions → choose action → analyze → advance or fork</div></>}
        {activeWorkspace === 'operations' && <><span>Company step 1 of 3</span><h2>Company Operations</h2><p>Create the shared company record, run monthly operations, read OpenAI narration, and build the history used by AI Intelligence.</p><div><b>Next</b> Create startup → simulate months → refresh history → open AI Intelligence</div></>}
        {activeWorkspace === 'intelligence' && <><span>Company step 2 of 3</span><h2>AI Intelligence</h2><p>Use the reinforcement-learning CEO, Digital Twin, and Strategy Lab on the startup created in Company Operations.</p><div><b>Input</b> Uses the same company record and saved monthly snapshots from step 1.</div></>}
        {activeWorkspace === 'data' && <><span>Company step 3 of 3</span><h2>Data & Models</h2><p>Import supporting evidence, inspect dataset hashes, and audit the models used throughout the platform.</p><div><b>Important</b> Imported data is versioned but does not automatically retrain the synthetic models.</div></>}
      </section>
      {activeWorkspace !== 'world' && (
        <div className="company-context-bar">
          <div><span>Active company</span><strong>{createdStartupId ? `${name || 'Startup'} · ID ${createdStartupId}` : 'No company created yet'}</strong></div>
          <div className="company-progress">
            <button className={activeWorkspace === 'operations' ? 'active' : ''} onClick={() => setActiveWorkspace('operations')}><b>1</b> Operations</button>
            <span>→</span>
            <button className={activeWorkspace === 'intelligence' ? 'active' : ''} disabled={!createdStartupId} onClick={() => setActiveWorkspace('intelligence')}><b>2</b> Intelligence</button>
            <span>→</span>
            <button className={activeWorkspace === 'data' ? 'active' : ''} onClick={() => setActiveWorkspace('data')}><b>3</b> Data & Models</button>
          </div>
        </div>
      )}
      {activeWorkspace === 'intelligence' && !createdStartupId && (
        <div className="workspace-empty">
          <span>Setup required</span><h3>Create a startup before opening its intelligence tools.</h3>
          <p>The AI CEO, Digital Twin, and Strategy Lab need one persistent company record to analyze.</p>
          <button className="btn btn-primary" onClick={() => setActiveWorkspace('operations')}>Go to Company Operations</button>
        </div>
      )}

      <div className="card civilization-card section-tone-simulation workspace-world" id="world-simulation">
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
          <div className="section-heading-actions">{helpButton('civilization')}{world && <button className="btn btn-secondary new-world-button" onClick={handleNewCivilization}>New Civilization</button>}</div>
        </div>
        {worldError && <p className="error-text">{worldError}</p>}
        {world && (
          <div className="world-control-room">
            <div className="workflow-step"><b>01</b><span>Read the current world</span><small>Global conditions and company performance</small></div>
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
            <div className="workflow-step"><b>02</b><span>Choose the next move</span><small>One action and an optional external shock</small></div>
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
            <div className="workflow-step"><b>03</b><span>Analyze before committing</span><small>Generate uncertainty, causal comparisons, or an AI plan</small></div>
            <button className="btn btn-secondary future-button" onClick={handleGenerateFutures} disabled={generatingFutures}>
              {generatingFutures && <span className="spinner" />}Generate 150 Possible Futures for “{worldAction.replace(/_/g, ' ')}”
            </button>
            {generatedFutures && (
              <div className="generated-futures">
                <p className="chart-title">Generated 12-month uncertainty · {generatedFutures.paths} paths</p>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={generatedFutures.timeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
                    <XAxis dataKey="month" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                    <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }} />
                    <Legend />
                    <Line type="monotone" dataKey="cash_p10" name="Stress cash" stroke="#ff6b6b" dot={false} />
                    <Line type="monotone" dataKey="cash_median" name="Median cash" stroke="#6e7bff" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="cash_p90" name="Upside cash" stroke="#4ade80" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <p className="uncertainty-note">Month {generatedFutures.timeline.at(-1)?.month}: {((generatedFutures.timeline.at(-1)?.survival_probability ?? 0) * 100).toFixed(1)}% generated survival. {generatedFutures.limitations}</p>
              </div>
            )}
            <div className="decision-lab">
              <div className="decision-lab-head">
                <div><strong>Causal Decision Lab</strong><span>Compare interventions, then search 12-month action sequences.</span></div>
                <div className="decision-buttons">
                  <button className="btn btn-secondary" onClick={handleCausalAnalysis} disabled={!!decisionLoading}>{decisionLoading === 'causal' && <span className="spinner" />}Estimate Action Effects</button>
                  <button className="btn btn-primary" onClick={handleModelBasedPlan} disabled={!!decisionLoading}>{decisionLoading === 'planner' && <span className="spinner" />}Ask Model-Based CEO</button>
                  <button className="btn btn-secondary" onClick={compareHumanWithAi} disabled={!!decisionLoading}>{decisionLoading === 'compare' && <span className="spinner" />}Fork My Choice vs AI</button>
                </div>
              </div>
              <label className="risk-control"><span>CEO downside protection</span><input type="range" min="0" max="1" step="0.05" value={riskAversion} onChange={(e) => setRiskAversion(Number(e.target.value))} /><b>{Math.round(riskAversion * 100)}%</b></label>
              {causalEffects && (
                <div className="causal-results">
                  <div className="causal-table causal-header"><span>Intervention vs hold</span><span>Cash</span><span>Customers</span><span>Revenue</span><span>Quality</span></div>
                  {causalEffects.effects.map((effect) => <div className="causal-table" key={effect.action}>
                    <strong>{effect.action.replace(/_/g, ' ')}</strong>
                    <span>{effect.effects_vs_hold.cash >= 0 ? '+' : ''}${effect.effects_vs_hold.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                    <span>{effect.effects_vs_hold.customers >= 0 ? '+' : ''}{effect.effects_vs_hold.customers.toFixed(1)}</span>
                    <span>{effect.effects_vs_hold.revenue >= 0 ? '+' : ''}${effect.effects_vs_hold.revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                    <span>{effect.effects_vs_hold.product_quality >= 0 ? '+' : ''}{(effect.effects_vs_hold.product_quality * 100).toFixed(2)}%</span>
                  </div>)}
                  <p className="uncertainty-note">Trained on {causalEffects.model.training_counterfactuals.toLocaleString()} paired interventions across {causalEffects.model.training_states} training and {causalEffects.model.held_out_states} held-out states. {causalEffects.limitations}</p>
                </div>
              )}
              {modelPlan && (
                <div className="plan-results">
                  <div className="plan-winner">
                    <span className="recommendation-label">Recommended first action</span>
                    <strong>{modelPlan.recommendation.first_action.replace(/_/g, ' ')}</strong>
                    <p>{(modelPlan.recommendation.survival_probability * 100).toFixed(1)}% modeled survival · ${modelPlan.recommendation.cash_p10.toLocaleString()} stress cash · ${modelPlan.recommendation.cash_median.toLocaleString()} median cash</p>
                    <div className="plan-sequence">{modelPlan.recommendation.planned_sequence.map((action, index) => <span key={`${action}-${index}`}>M{index + 1} {action.replace(/_/g, ' ')}</span>)}</div>
                    <button className="btn btn-primary" onClick={executePlannedAction} disabled={worldLoading}>Authorize first action and advance</button>
                  </div>
                  <div className="plan-comparison">
                    {modelPlan.action_comparison.slice(0, 5).map((plan) => <div key={plan.first_action}><b>#{plan.rank} {plan.first_action.replace(/_/g, ' ')}</b><span>{(plan.survival_probability * 100).toFixed(0)}% survive</span><span>${plan.cash_p10.toLocaleString()} stress</span></div>)}
                  </div>
                  <p className="uncertainty-note">Searched {modelPlan.search.actions} actions over {modelPlan.search.horizon} months with beam width {modelPlan.search.beam_width_per_first_action}, then stress-tested each plan through {modelPlan.search.stochastic_paths_per_plan} generated futures. {modelPlan.limitations}</p>
                </div>
              )}
              {humanAi && (
                <div className="human-ai-results">
                  <div className="human-ai-column">
                    <span>YOUR BRANCH</span><strong>{humanAi.human_action.replace(/_/g, ' ')}</strong>
                    <b>${humanAi.human_branch.companies.player.cash.toLocaleString()} cash</b>
                    <p>{humanAi.human_branch.companies.player.customers.toLocaleString()} customers · ${humanAi.human_branch.companies.player.revenue.toLocaleString()} revenue</p>
                    <button className="btn btn-secondary" onClick={() => handleSwitchBranch(humanAi.human_branch.branch_id)}>Open your timeline</button>
                  </div>
                  <div className="comparison-vs">VS</div>
                  <div className="human-ai-column ai-column">
                    <span>MODEL-BASED AI BRANCH</span><strong>{humanAi.ai_action.replace(/_/g, ' ')}</strong>
                    <b>${humanAi.ai_branch.companies.player.cash.toLocaleString()} cash</b>
                    <p>{humanAi.ai_branch.companies.player.customers.toLocaleString()} customers · ${humanAi.ai_branch.companies.player.revenue.toLocaleString()} revenue</p>
                    <button className="btn btn-primary" onClick={() => handleSwitchBranch(humanAi.ai_branch.branch_id)}>Open AI timeline</button>
                  </div>
                </div>
              )}
            </div>
            <div className="workflow-step"><b>04</b><span>Inspect timelines and evidence</span><small>Switch branches, replay snapshots, and audit events</small></div>
            <div className="branch-tabs">
              {worldBranches.map((branch) => <button className={branch.id === world.branch_id ? 'active' : ''} onClick={() => handleSwitchBranch(branch.id)} key={branch.id}>{branch.name} · M{branch.current_month}</button>)}
            </div>
            <div className="replay-control">
              <div><strong>Timeline replay</strong><span>Inspect a saved month without changing history.</span></div>
              <input type="range" min="0" max={world.month} value={replayState ? replayMonth : world.month} onChange={(e) => inspectReplayMonth(Number(e.target.value))} />
              <b>M{replayState ? replayMonth : world.month}</b>
              {replayState && <button className="btn btn-secondary" onClick={() => setReplayState(null)}>Return live</button>}
            </div>
            {replayState && <div className="replay-snapshot">
              <span>REPLAY · READ ONLY</span><strong>{replayState.companies.player.name} at month {replayState.month}</strong>
              <p>${replayState.companies.player.cash.toLocaleString()} cash · ${replayState.companies.player.revenue.toLocaleString()} revenue · {replayState.companies.player.customers.toLocaleString()} customers · {replayState.macro.regime} economy</p>
            </div>}
            <details className="world-event-log">
              <summary>Event log ({worldEvents.length})</summary>
              {worldEvents.slice(-12).reverse().map((event) => <div key={event.id}><b>M{event.month}</b> {event.actor_id}: {event.type.replace(/_/g, ' ')}</div>)}
            </details>
          </div>
        )}
      </div>

      <div className="card data-lab-card section-tone-data workspace-data" id="real-data-lab">
        <div className="section-heading">
          <div>
            <h2>Real Data Lab</h2>
            <p className="card-sub">Import versioned official evidence and your own time-series data with hashes and provenance.</p>
          </div>
          <div className="official-data-actions">
            {helpButton('data')}
            <button className="btn btn-secondary" onClick={() => importOfficialData('fred')} disabled={!!dataLoading}>{dataLoading === 'fred' && <span className="spinner" />}Sync FRED</button>
            <button className="btn btn-secondary" onClick={() => importOfficialData('census_bds')} disabled={!!dataLoading}>{dataLoading === 'census_bds' && <span className="spinner" />}Sync Census BDS</button>
          </div>
        </div>
        <div className="data-import-grid">
          <div className="data-import-panel">
            <strong>SEC Company Facts</strong>
            <p>Pull audited 10-K and 10-Q facts directly from the SEC. Apple is the example CIK.</p>
            <div className="inline-import"><input className="field" value={secCik} onChange={(e) => setSecCik(e.target.value)} placeholder="10-digit CIK" /><button className="btn btn-primary" onClick={() => importOfficialData('sec_companyfacts')} disabled={!!dataLoading}>Import SEC</button></div>
          </div>
          <div className="data-import-panel">
            <strong>Your longitudinal CSV</strong>
            <p>Required columns: date, series, value. Unit, entity, and any extra dimensions are preserved.</p>
            <input className="field" value={csvName} onChange={(e) => setCsvName(e.target.value)} placeholder="Dataset name" />
            <textarea className="field data-textarea" value={csvText} onChange={(e) => setCsvText(e.target.value)} />
            <button className="btn btn-primary" onClick={importCsvData} disabled={!!dataLoading}>{dataLoading === 'csv' && <span className="spinner" />}Import CSV</button>
          </div>
        </div>
        {dataError && <p className="error-text">{dataError}</p>}
        <div className="dataset-list">
          {datasets.length === 0 ? <p className="uncertainty-note">No datasets imported for this account yet.</p> : datasets.map((dataset) => (
            <div className="dataset-row" key={dataset.id}>
              <div><strong>{dataset.dataset_name}</strong><span>{dataset.source.replace(/_/g, ' ')} · {dataset.row_count.toLocaleString()} observations</span></div>
              <code title={dataset.content_sha256}>SHA-256 {dataset.content_sha256.slice(0, 12)}…</code>
            </div>
          ))}
        </div>
      </div>

      <div className="card registry-card section-tone-governance workspace-data" id="model-registry">
        <div className="section-heading">
          <div>
            <h2>Experiment & Model Registry</h2>
            <p className="card-sub">Cryptographic lineage for every active learned system—not a hand-written benchmark panel.</p>
          </div>
          <div className="section-heading-actions">{helpButton('registry')}<span className="registry-count">{registry.length} versioned artifacts</span></div>
        </div>
        <div className="registry-grid">
          {registry.map((model) => (
            <details className="registry-model" key={model.model_id}>
              <summary><span><strong>{model.model_id.replace(/_/g, ' ')}</strong><small>{model.data_lineage.kind} · {(model.artifact_bytes / 1e6).toFixed(1)} MB</small></span><b>{model.status}</b></summary>
              <div className="registry-lineage">
                <span>Training data</span><code>{model.data_lineage.source}</code>
                <span>Artifact SHA-256</span><code>{model.artifact_sha256}</code>
                <span>Training-code SHA-256</span><code>{model.training_code_sha256}</code>
                <span>Reproduce</span><code>{model.reproduce_command}</code>
              </div>
              <p>{model.limitations}</p>
            </details>
          ))}
        </div>
      </div>

      <div className="card section-tone-operations workspace-operations" id="create-startup">
        <div className="section-title-row"><h2>Create a Startup</h2>{helpButton('startup')}</div>
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
        <div className="card ceo-card section-tone-decision workspace-intelligence" id="ai-ceo">
          <div className="section-heading">
            <div>
              <h2>AI CEO</h2>
              <p className="card-sub">A reinforcement-learning policy chooses what the company should do next.</p>
            </div>
            <button className="btn btn-secondary strategy-button" onClick={handleAskAiCeo} disabled={loadingCeo}>
              {loadingCeo && <span className="spinner" />}
              Ask AI CEO
            </button>
            {helpButton('ceo')}
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
        <div className="card twin-card section-tone-predictive workspace-intelligence" id="digital-twin">
          <div className="section-heading">
            <div>
              <h2>Startup Digital Twin</h2>
              <p className="card-sub">One model forecasts the connected financial and customer state three months ahead.</p>
            </div>
            <button className="btn btn-secondary strategy-button" onClick={handleRunDigitalTwin} disabled={loadingTwin}>
              {loadingTwin && <span className="spinner" />}
              Run Digital Twin
            </button>
            {helpButton('twin')}
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
        <div className="card strategy-card section-tone-decision workspace-intelligence" id="strategy-lab">
          <div className="section-heading">
            <div>
              <h2>AI Strategy Lab</h2>
              <p className="card-sub">Stress-test pricing, marketing, and hiring across thousands of possible futures.</p>
            </div>
            <button className="btn btn-secondary strategy-button" onClick={handleAnalyzeStrategies} disabled={analyzingStrategies}>
              {analyzingStrategies && <span className="spinner" />}
              Analyze 12-month strategy
            </button>
            {helpButton('strategy')}
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
        <div className="card section-tone-operations workspace-operations" id="monthly-simulation">
          <div className="section-title-row"><h2>Simulate Next Month</h2>{helpButton('month')}</div>
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
        <div className="card section-tone-history workspace-operations" id="startup-history">
          <div className="section-title-row"><h2>History</h2>{helpButton('history')}</div>
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
      {helpOverlay}
    </div>
  )
}

export default App
