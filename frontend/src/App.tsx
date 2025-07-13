import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [apiData, setApiData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchFromAPI = async (endpoint: string) => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api${endpoint}`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setApiData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div>
        <h1>ALEC Frontend</h1>
        <p>Connected to FastAPI Backend</p>
      </div>
      
      <div className="card">
        <h2>Test API Connection</h2>
        <button onClick={() => fetchFromAPI('/')}>
          Test Root Endpoint
        </button>
        <button onClick={() => fetchFromAPI('/items/42?q=test')}>
          Test Items Endpoint
        </button>
        
        {loading && <p>Loading...</p>}
        {error && <p style={{color: 'red'}}>Error: {error}</p>}
        {apiData && (
          <div>
            <h3>API Response:</h3>
            <pre>{JSON.stringify(apiData, null, 2)}</pre>
          </div>
        )}
      </div>
      
      <p className="read-the-docs">
        Make sure your FastAPI backend is running on localhost:8000
      </p>
    </>
  )
}

export default App
