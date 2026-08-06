import { useState, type SubmitEvent } from 'react'

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
  narration: string
}

interface Snapshot {
  id: number
  month_number: number
  revenue: number
  cash_on_hand: number
  customer_count: number
  customers_churned: number
}

function App() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [authError, setAuthError] = useState('')

  const [name, setName] = useState('')
  const [businessType, setBusinessType] = useState('')
  const [initialPrice, setInitialPrice] = useState('')
  const [founderCount, setFounderCount] = useState('')
  const [initialFunding, setInitialFunding] = useState('')
  const [initialCustomerCount, setInitialCustomerCount] = useState('')
  const [createdStartupId, setCreatedStartupId] = useState<number | null>(null)

  const [simMarketingSpend, setSimMarketingSpend] = useState('')
  const [simEmployeeCount, setSimEmployeeCount] = useState('')
  const [simResult, setSimResult] = useState<SimulationResult | null>(null)
  const [history, setHistory] = useState<Snapshot[]>([])

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
    await loginWithCredentials()
  }

  async function handleRegister() {
    setAuthError('')

    const response = await fetch('http://127.0.0.1:8000/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })

    if (!response.ok) {
      setAuthError('Registration failed')
      return
    }

    await loginWithCredentials()
  }

  async function handleGuestLogin() {
    setAuthError('')

    const response = await fetch('http://127.0.0.1:8000/guest-login', { method: 'POST' })
    const data = await response.json()
    localStorage.setItem('token', data.access_token)
    setToken(data.access_token)
  }

  function handleLogout() {
    localStorage.removeItem('token')
    setToken(null)
  }

  async function handleSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault()

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
  }

  async function handleSimulate(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault()

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
  }

  async function handleViewHistory() {
    const response = await fetch(`http://127.0.0.1:8000/startups/${createdStartupId}/snapshots`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    const data = await response.json()
    setHistory(data)
  }

  if (!token) {
    return (
      <div>
        <h1>AI Startup Intelligence Platform</h1>
        <form onSubmit={handleLogin}>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
          <button type="submit">Log In</button>
          <button type="button" onClick={handleRegister}>Register</button>
        </form>
        <button type="button" onClick={handleGuestLogin}>Try as Guest</button>
        {authError && <p>{authError}</p>}
      </div>
    )
  }

  return (
    <div>
      <h1>AI Startup Intelligence Platform</h1>
      <button onClick={handleLogout}>Log Out</button>
      <form onSubmit={handleSubmit}>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Startup name" />
        <input type="text" value={businessType} onChange={(e) => setBusinessType(e.target.value)} placeholder="Business type" />
        <input type="number" value={initialPrice} onChange={(e) => setInitialPrice(e.target.value)} placeholder="Initial price" />
        <input type="number" value={founderCount} onChange={(e) => setFounderCount(e.target.value)} placeholder="Founder count" />
        <input type="number" value={initialFunding} onChange={(e) => setInitialFunding(e.target.value)} placeholder="Initial funding" />
        <input type="number" value={initialCustomerCount} onChange={(e) => setInitialCustomerCount(e.target.value)} placeholder="Initial customer count" />
        <button type="submit">Create Startup</button>
      </form>
      {createdStartupId && <p>Created startup with id {createdStartupId}</p>}

      {createdStartupId && (
        <div>
          <h2>Simulate Next Month</h2>
          <form onSubmit={handleSimulate}>
            <input type="number" value={simMarketingSpend} onChange={(e) => setSimMarketingSpend(e.target.value)} placeholder="Marketing spend" />
            <input type="number" value={simEmployeeCount} onChange={(e) => setSimEmployeeCount(e.target.value)} placeholder="Employee count" />
            <button type="submit">Simulate Next Month</button>
          </form>

          {simResult && (
            <div>
              <p>Revenue: {simResult.revenue}</p>
              <p>Cash on hand: {simResult.cash_on_hand}</p>
              <p>Customers: {simResult.customer_count}</p>
              <p>{simResult.narration}</p>
            </div>
          )}

          <button onClick={handleViewHistory}>View History</button>
          <ul>
            {history.map((snapshot) => (
              <li key={snapshot.id}>
                Month {snapshot.month_number}: Revenue ${snapshot.revenue}, Cash ${snapshot.cash_on_hand}, Customers {snapshot.customer_count}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
