import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000'
axios.defaults.withCredentials = true

const QUALITY_COLORS = {
  Excellent: '#10b981',
  Good: '#6366f1',
  Average: '#f59e0b',
  Poor: '#ef4444'
}

function LoginScreen({ onLoggedIn }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const endpoint = mode === 'login' ? '/api/login' : '/api/register'
      const response = await axios.post(`${API_BASE}${endpoint}`, { username, password })
      onLoggedIn(response.data.username)
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="login-wrapper">
      <div className="login-card">
        <h1 className="login-title">Signal Quality Forecast</h1>
        <p className="login-subtitle">{mode === 'login' ? 'Log in to continue' : 'Create an account'}</p>

        <form onSubmit={submit} className="login-form">
          <input
            type="text" placeholder="Username" value={username}
            onChange={(e) => setUsername(e.target.value)} className="login-input" required
          />
          <input
            type="password" placeholder="Password" value={password}
            onChange={(e) => setPassword(e.target.value)} className="login-input" required
          />
          {error && <p className="error">{error}</p>}
          <button type="submit" disabled={submitting} className="btn btn-primary login-submit">
            {submitting ? 'Please wait…' : mode === 'login' ? 'Log In' : 'Register'}
          </button>
        </form>

        <p className="login-switch">
          {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
          <button className="link-btn" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null) }}>
            {mode === 'login' ? 'Register' : 'Log in'}
          </button>
        </p>
      </div>
    </div>
  )
}

function App() {
  const [authChecked, setAuthChecked] = useState(false)
  const [username, setUsername] = useState(null)
  const [currentPage, setCurrentPage] = useState('setup') // 'setup' | 'results'

  const [forecast, setForecast] = useState([])
  const [historical, setHistorical] = useState([])
  const [summary, setSummary] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadedFilename, setUploadedFilename] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [model, setModel] = useState('random_forest')
  const [trainDays, setTrainDays] = useState(7)
  const [predictDays, setPredictDays] = useState(7)
  const [activeTab, setActiveTab] = useState('RSRP')
  const [theme, setTheme] = useState('light')
  const [linkCopied, setLinkCopied] = useState(false)

  const [validation, setValidation] = useState(null)
  const [validating, setValidating] = useState(false)
  const [validationError, setValidationError] = useState(null)

  const [dateStart, setDateStart] = useState('')
  const [dateEnd, setDateEnd] = useState('')
  const [compareMode, setCompareMode] = useState(false)
  const [compareForecast, setCompareForecast] = useState([])
  const [comparing, setComparing] = useState(false)
  const [showAnomalies, setShowAnomalies] = useState(true)
  const [showHistorical, setShowHistorical] = useState(true)
  const [locations, setLocations] = useState([])
  const [loadingLocations, setLoadingLocations] = useState(false)
  const [viewMode, setViewMode] = useState('charts')

  const [compareFilesMode, setCompareFilesMode] = useState(false)
  const [secondFilename, setSecondFilename] = useState(null)
  const [uploadingSecond, setUploadingSecond] = useState(false)
  const [secondForecast, setSecondForecast] = useState([])

  const [userLocation, setUserLocation] = useState(null)
  const [locationError, setLocationError] = useState(null)
  const [nearestSignal, setNearestSignal] = useState(null)
  const [findingSignal, setFindingSignal] = useState(false)

  // SHAP classification state
  const [classification, setClassification] = useState(null)
  const [classificationLoading, setClassificationLoading] = useState(false)
  const [classificationError, setClassificationError] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  useEffect(() => {
    axios.get(`${API_BASE}/api/me`).then((res) => {
      setUsername(res.data.username)
      setAuthChecked(true)
    }).catch(() => setAuthChecked(true))
  }, [])

  useEffect(() => {
    if (!username) return
    const params = new URLSearchParams(window.location.search)
    const m = params.get('model')
    const td = params.get('train_days')
    const pd = params.get('predict_days')
    const fn = params.get('filename')
    if (m) setModel(m)
    if (td) setTrainDays(clamp(td))
    if (pd) setPredictDays(clamp(pd))
    if (fn) setUploadedFilename(fn)
  }, [username])

  const handleLogout = async () => {
    await axios.post(`${API_BASE}/api/logout`)
    setUsername(null)
    resetAll()
  }

  const toggleTheme = () => setTheme(theme === 'light' ? 'dark' : 'light')

  const clamp = (val) => {
    const n = parseInt(val, 10)
    if (isNaN(n)) return 1
    return Math.max(1, Math.min(10, n))
  }

  const goToSetup = () => setCurrentPage('setup')

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setUploading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setUploadedFilename(response.data.filename)
    } catch (err) {
      setError('Upload failed. Make sure it is a valid .xlsx file.')
    } finally {
      setUploading(false)
    }
  }

  const handleSecondFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setUploadingSecond(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setSecondFilename(response.data.filename)
    } catch (err) {
      setError('Second file upload failed. Make sure it is a valid .xlsx file.')
    } finally {
      setUploadingSecond(false)
    }
  }

  const fetchForecastFor = async (modelName, filename) => {
    const params = { model: modelName, train_days: trainDays, predict_days: predictDays }
    if (filename) params.filename = filename
    const response = await axios.get(`${API_BASE}/api/forecast`, { params })
    return response.data
  }

  const runForecast = async () => {
    setLoading(true)
    setError(null)
    setCompareForecast([])
    setSecondForecast([])
    try {
      const data = await fetchForecastFor(model, uploadedFilename)
      setForecast(data.forecast)
      setHistorical(data.historical || [])
      setSummary(data.summary)
      setCurrentPage('results')

      if (compareMode) {
        setComparing(true)
        const otherModel = model === 'prophet' ? 'random_forest' : 'prophet'
        const otherData = await fetchForecastFor(otherModel, uploadedFilename)
        setCompareForecast(otherData.forecast)
        setComparing(false)
      }

      if (compareFilesMode && secondFilename) {
        setComparing(true)
        const secondData = await fetchForecastFor(model, secondFilename)
        setSecondForecast(secondData.forecast)
        setComparing(false)
      }
    } catch (err) {
      if (err.response?.status === 401) {
        setUsername(null)
      } else {
        setError('Failed to fetch forecast. Is the Flask backend running?')
      }
    } finally {
      setLoading(false)
    }
  }

  const runValidation = async () => {
    setValidating(true)
    setValidationError(null)
    try {
      const params = {}
      if (uploadedFilename) params.filename = uploadedFilename
      const response = await axios.get(`${API_BASE}/api/validate`, { params })
      setValidation(response.data)
      setCurrentPage('results')
    } catch (err) {
      setValidationError('Validation failed. Check the backend terminal for details.')
    } finally {
      setValidating(false)
    }
  }

  const loadLocations = async () => {
    setLoadingLocations(true)
    try {
      const params = {}
      if (uploadedFilename) params.filename = uploadedFilename
      const response = await axios.get(`${API_BASE}/api/locations`, { params })
      setLocations(response.data.locations)
    } catch (err) {
      setError('Failed to load location data.')
    } finally {
      setLoadingLocations(false)
    }
  }

  const findNearestGoodSignal = () => {
    setLocationError(null)
    setFindingSignal(true)

    if (!navigator.geolocation) {
      setLocationError('Your browser does not support location access.')
      setFindingSignal(false)
      return
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude
        const lon = position.coords.longitude
        setUserLocation({ lat, lon })
        try {
          const params = { lat, lon }
          if (uploadedFilename) params.filename = uploadedFilename
          const response = await axios.get(`${API_BASE}/api/nearest-good-signal`, { params })
          setNearestSignal(response.data)
        } catch (err) {
          setLocationError('Could not fetch nearest signal data.')
        } finally {
          setFindingSignal(false)
        }
      },
      () => {
        setLocationError('Location permission denied or unavailable. You can still check the reference guide below.')
        setFindingSignal(false)
      }
    )
  }

  const loadClassification = async () => {
    setClassificationLoading(true)
    setClassificationError(null)
    try {
      const response = await axios.get(`${API_BASE}/api/classification`)
      setClassification(response.data)
    } catch (err) {
      if (err.response?.status === 401) {
        setUsername(null)
      } else {
        setClassificationError(err.response?.data?.error || 'Could not load classification and SHAP explanations. Check that the Flask backend is running and the model is available.')
      }
    } finally {
      setClassificationLoading(false)
    }
  }

  const exportCSV = () => {
    if (forecast.length === 0) return
    const headers = Object.keys(forecast[0])
    const rows = forecast.map((row) => headers.map((h) => row[h]).join(','))
    const csvContent = [headers.join(','), ...rows].join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `forecast_${model}_${predictDays}day.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const copyShareLink = async () => {
    const params = new URLSearchParams()
    params.set('model', model)
    params.set('train_days', trainDays)
    params.set('predict_days', predictDays)
    if (uploadedFilename) params.set('filename', uploadedFilename)
    const shareUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`
    try {
      await navigator.clipboard.writeText(shareUrl)
      setLinkCopied(true)
      setTimeout(() => setLinkCopied(false), 2000)
    } catch (err) {
      setError('Could not copy link — please copy it manually.')
    }
  }

  const resetAll = () => {
    setForecast([])
    setHistorical([])
    setCompareForecast([])
    setSecondForecast([])
    setSummary([])
    setValidation(null)
    setError(null)
    setValidationError(null)
    setUploadedFilename(null)
    setSecondFilename(null)
    setModel('random_forest')
    setTrainDays(7)
    setPredictDays(7)
    setActiveTab('RSRP')
    setDateStart('')
    setDateEnd('')
    setCompareMode(false)
    setCompareFilesMode(false)
    setLocations([])
    setViewMode('charts')
    setNearestSignal(null)
    setUserLocation(null)
    setClassification(null)
    setClassificationError(null)
    setClassificationLoading(false)
    setCurrentPage('setup')
    window.history.replaceState({}, '', window.location.pathname)
  }

  const buildChartData = () => {
    let combined = []
    if (showHistorical && historical.length > 0) {
      combined = combined.concat(historical.map((r) => ({ ...r, __phase: 'historical' })))
    }
    combined = combined.concat(forecast.map((r) => ({ ...r, __phase: 'forecast' })))

    if (compareMode && compareForecast.length > 0) {
      const compareMap = {}
      compareForecast.forEach((r) => { compareMap[r.Timestamp] = r })
      combined = combined.map((r) => {
        const alt = compareMap[r.Timestamp]
        if (!alt) return r
        return { ...r, RSRP_alt: alt.RSRP, RSRQ_alt: alt.RSRQ, MOS_alt: alt.MOS }
      })
    }

    if (compareFilesMode && secondForecast.length > 0) {
      combined = combined.map((r, idx) => {
        const alt = secondForecast[idx]
        if (!alt) return r
        return { ...r, RSRP_file2: alt.RSRP, RSRQ_file2: alt.RSRQ, MOS_file2: alt.MOS }
      })
    }

    if (dateStart) combined = combined.filter((r) => r.Timestamp >= dateStart)
    if (dateEnd) combined = combined.filter((r) => r.Timestamp <= dateEnd + ' 23:59:59')

    return combined
  }

  const chartData = buildChartData()
  const forecastStartTs = forecast.length > 0 ? forecast[0].Timestamp : null

  const anomalyDot = (metric) => (props) => {
    const { cx, cy, payload } = props
    if (!showAnomalies) return null
    if (payload.Quality === 'Poor' || payload.Quality === 'Average') {
      return <circle cx={cx} cy={cy} r={4} fill={payload.Quality === 'Poor' ? '#ef4444' : '#f59e0b'} stroke="white" strokeWidth={1} />
    }
    return null
  }

  const altModelLabel = model === 'prophet' ? 'Random Forest' : 'Prophet'

  const qualityColors = {
    Excellent: 'var(--pill-excellent)',
    Good: 'var(--pill-good)',
    Average: 'var(--pill-average)',
    Poor: 'var(--pill-poor)'
  }

  const validationChartData = validation
    ? Object.entries(validation.results).map(([metric, vals]) => ({
        metric, Prophet: vals.prophet.mae, 'Random Forest': vals.random_forest.mae ?? 0
      }))
    : []

  const renderChart = (metric, color, domain) => (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="Timestamp" tick={false} />
        <YAxis domain={domain} tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
        <Tooltip />
        <Legend />
        {showHistorical && forecastStartTs && !compareFilesMode && (
          <ReferenceLine x={forecastStartTs} stroke="var(--text-muted)" strokeDasharray="4 4" label={{ value: 'Forecast starts', position: 'top', fill: 'var(--text-muted)', fontSize: 11 }} />
        )}
        <Line type="monotone" dataKey={metric} name={`${metric} — ${uploadedFilename || 'default file'}`} stroke={color} strokeWidth={2} dot={anomalyDot(metric)} />
        {compareMode && (
          <Line type="monotone" dataKey={`${metric}_alt`} name={`${metric} (${altModelLabel})`} stroke="var(--chart-prophet)" strokeWidth={2} strokeDasharray="5 3" dot={false} />
        )}
        {compareFilesMode && secondForecast.length > 0 && (
          <Line type="monotone" dataKey={`${metric}_file2`} name={`${metric} — ${secondFilename}`} stroke="var(--chart-prophet)" strokeWidth={2} strokeDasharray="5 3" dot={false} />
        )}
      </LineChart>
    </ResponsiveContainer>
  )

  if (!authChecked) {
    return <div className="page"><p className="muted">Loading…</p></div>
  }

  if (!username) {
    return <LoginScreen onLoggedIn={setUsername} />
  }

  return (
    <div className="page" data-theme={theme}>
      <header className="hero">
        <div className="hero-top-row">
          <span className="user-badge">👤 {username}</span>
          <div className="hero-buttons">
            <button className="theme-toggle" onClick={toggleTheme}>{theme === 'light' ? '🌙 Dark' : '☀️ Light'}</button>
            <button className="theme-toggle" onClick={handleLogout}>Log out</button>
          </div>
        </div>
        <h1>Signal Quality Forecast</h1>
        <p className="subtitle">RSRP, RSRQ &amp; MOS prediction — Random Forest vs Prophet</p>
      </header>

      {currentPage === 'setup' && (
        <>
          <section className="card controls-card">
            <div className="control-group">
              <label>Data file</label>
              <input type="file" accept=".xlsx" onChange={handleFileUpload} />
              {uploading && <span className="hint">Uploading...</span>}
              {uploadedFilename && !uploading && <span className="hint success">✓ {uploadedFilename}</span>}
              {!uploadedFilename && !uploading && <span className="hint">No file uploaded — using default sample</span>}
            </div>

            <div className="control-group">
              <label>Model</label>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="random_forest">Random Forest</option>
                <option value="prophet">Prophet</option>
              </select>
            </div>

            <div className="control-group">
              <label>Days of data</label>
              <input type="number" min="1" max="10" value={trainDays} onChange={(e) => setTrainDays(clamp(e.target.value))} className="day-input" />
              <span className="hint">1–10 days</span>
            </div>

            <div className="control-group">
              <label>Days to predict</label>
              <input type="number" min="1" max="10" value={predictDays} onChange={(e) => setPredictDays(clamp(e.target.value))} className="day-input" />
              <span className="hint">1–10 days</span>
            </div>

            <div className="control-group">
              <label>Compare models</label>
              <label className="switch">
                <input type="checkbox" checked={compareMode} onChange={(e) => setCompareMode(e.target.checked)} />
                <span className="switch-slider"></span>
              </label>
              <span className="hint">Overlay both models on one chart</span>
            </div>

            <div className="control-group">
              <label>Compare files</label>
              <label className="switch">
                <input type="checkbox" checked={compareFilesMode} onChange={(e) => setCompareFilesMode(e.target.checked)} />
                <span className="switch-slider"></span>
              </label>
              <span className="hint">Upload a second file to compare side by side</span>
            </div>

            {compareFilesMode && (
              <div className="control-group">
                <label>Second file</label>
                <input type="file" accept=".xlsx" onChange={handleSecondFileUpload} />
                {uploadingSecond && <span className="hint">Uploading...</span>}
                {secondFilename && !uploadingSecond && <span className="hint success">✓ {secondFilename}</span>}
              </div>
            )}

            <div className="button-row">
              <button onClick={runForecast} disabled={loading} className="btn btn-primary">
                {loading ? 'Running forecast…' : comparing ? 'Fetching comparison…' : 'Run Forecast'}
              </button>
              <button onClick={runValidation} disabled={validating} className="btn btn-secondary">
                {validating ? 'Validating…' : 'Validate Models'}
              </button>
              <button onClick={copyShareLink} className="btn btn-ghost">
                {linkCopied ? '✓ Link copied!' : '🔗 Copy Share Link'}
              </button>
              <button onClick={resetAll} className="btn btn-reset">✕ Reset</button>
            </div>

            {error && <p className="error">{error}</p>}
            {validationError && <p className="error">{validationError}</p>}
          </section>

          <section className="card">
            <h2>📶 Signal Quality Reference</h2>
            <p className="muted">What the numbers actually mean, based on standard MOS-to-signal-quality mapping.</p>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Quality Grade</th>
                  <th>MOS Score</th>
                  <th>Typical RSRP</th>
                  <th>Typical RSRQ</th>
                  <th>What it means</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="row-label" style={{ color: 'var(--pill-excellent)' }}>Excellent</td>
                  <td>4.0 – 5.0</td><td>-80 dBm or stronger</td><td>-8 dB or better</td>
                  <td>Strong, reliable connection — smooth calls/streaming</td>
                </tr>
                <tr>
                  <td className="row-label" style={{ color: 'var(--pill-good)' }}>Good</td>
                  <td>3.5 – 4.0</td><td>-80 to -95 dBm</td><td>-8 to -12 dB</td>
                  <td>Solid everyday use, minor slowdowns possible</td>
                </tr>
                <tr>
                  <td className="row-label" style={{ color: 'var(--pill-average)' }}>Average</td>
                  <td>3.0 – 3.5</td><td>-95 to -105 dBm</td><td>-12 to -15 dB</td>
                  <td>Noticeable lag, occasional call quality issues</td>
                </tr>
                <tr>
                  <td className="row-label" style={{ color: 'var(--pill-poor)' }}>Poor</td>
                  <td>Below 3.0</td><td>Weaker than -105 dBm</td><td>Weaker than -15 dB</td>
                  <td>Dropped calls, buffering, unreliable data</td>
                </tr>
              </tbody>
            </table>
            <p className="warning" style={{ marginTop: '1rem' }}>
              These are widely-used general reference ranges — actual thresholds can vary slightly by operator and device.
            </p>
          </section>

          <section className="card">
            <h2>📍 How Far to Better Signal?</h2>
            <p className="muted">
              Uses your current location and compares it against the real GPS-tagged readings in the uploaded dataset
              to estimate how far away a stronger signal was actually recorded.
            </p>
            <button onClick={findNearestGoodSignal} disabled={findingSignal} className="btn btn-primary">
              {findingSignal ? 'Finding nearest good signal…' : '📍 Use My Location'}
            </button>

            {locationError && <p className="error">{locationError}</p>}

            {nearestSignal && (
              <div className="signal-estimate">
                {nearestSignal.nearest_good_or_better ? (
                  <p>
                    The nearest recorded <strong>{nearestSignal.nearest_good_or_better.quality}</strong> signal
                    (MOS {nearestSignal.nearest_good_or_better.mos}) in this dataset is approximately{' '}
                    <strong>{nearestSignal.nearest_good_or_better.distance_km} km</strong> away from your current location.
                  </p>
                ) : (
                  <p>No Good or Excellent readings were found in this dataset.</p>
                )}
                {nearestSignal.nearest_excellent && (
                  <p className="muted">
                    The nearest <strong>Excellent</strong> reading specifically (MOS {nearestSignal.nearest_excellent.mos})
                    was recorded about {nearestSignal.nearest_excellent.distance_km} km away.
                  </p>
                )}
                <p className="warning">
                  This estimate is based only on where the uploaded drive-test data happened to be recorded — it reflects
                  historical measurements, not a live coverage map, and may not reflect current network conditions at that location.
                </p>
              </div>
            )}
          </section>

          <section className="card shap-card">
            <div className="shap-heading-row">
              <div>
                <h2>🧠 Signal Quality Classification &amp; SHAP Explanations</h2>
                <p className="muted">
                  Predict signal quality for each site using the trained Random Forest classifier and inspect the
                  five strongest feature contributions behind each prediction.
                </p>
              </div>
              <button onClick={loadClassification} disabled={classificationLoading} className="btn btn-primary">
                {classificationLoading ? 'Analysing sites…' : classification ? 'Refresh Analysis' : 'Run Classification'}
              </button>
            </div>

            {classificationError && <p className="error">{classificationError}</p>}
            {classificationLoading && <p className="muted">Generating predictions and SHAP explanations. This may take a moment…</p>}

            {classification?.success && (
              <>
                <div className="shap-summary-row">
                  <div className="shap-summary-item">
                    <span className="shap-summary-value">{classification.total_sites}</span>
                    <span className="shap-summary-label">Sites analysed</span>
                  </div>
                  <div className="shap-summary-item">
                    <span className="shap-summary-value">{classification.predictions?.length || 0}</span>
                    <span className="shap-summary-label">Predictions returned</span>
                  </div>
                  <div className="shap-summary-item">
                    <span className={`shap-status-dot ${classification.explanations_available ? 'available' : 'unavailable'}`}></span>
                    <span className="shap-summary-label">{classification.explanations_available ? 'SHAP explanations available' : 'SHAP explanations unavailable'}</span>
                  </div>
                </div>

                {classification.explanation_error && (
                  <p className="warning">Predictions were generated, but SHAP explanations could not be calculated: {classification.explanation_error}</p>
                )}

                <div className="shap-site-grid">
                  {(classification.predictions || []).map((site) => (
                    <article className="shap-site-card" key={site.site_id}>
                      <div className="shap-site-header">
                        <div>
                          <h3>{site.site_id}</h3>
                          <p className="muted">Latest reading: {site.timestamp ? new Date(site.timestamp).toLocaleString() : 'Not provided'}</p>
                        </div>
                        <span className="shap-quality-badge" style={{ backgroundColor: QUALITY_COLORS[site.predicted_quality] || '#64748b' }}>
                          {site.predicted_quality}
                        </span>
                      </div>

                      <h4>Class probabilities</h4>
                      <div className="shap-probabilities">
                        {Object.entries(site.probabilities || {}).map(([quality, probability]) => {
                          const percentage = Math.max(0, Math.min(100, Number(probability) * 100))
                          return (
                            <div className="shap-probability-row" key={quality}>
                              <span>{quality}</span>
                              <div className="shap-probability-track">
                                <div className="shap-probability-fill" style={{ width: `${percentage}%`, backgroundColor: QUALITY_COLORS[quality] || '#64748b' }} />
                              </div>
                              <strong>{percentage.toFixed(1)}%</strong>
                            </div>
                          )
                        })}
                      </div>

                      <h4>Top SHAP feature contributions</h4>
                      {site.explanations?.length > 0 ? (
                        <div className="shap-feature-list">
                          {site.explanations.map((item, index) => (
                            <div className="shap-feature-row" key={`${item.feature}-${index}`}>
                              <div className="shap-feature-main">
                                <strong>{item.feature}</strong>
                                <span className={`shap-direction ${item.direction || 'neutral'}`}>
                                  {item.direction === 'supports_prediction' ? 'Supports prediction' : item.direction === 'opposes_prediction' ? 'Opposes prediction' : 'Neutral'}
                                </span>
                              </div>
                              <div className="shap-feature-values">
                                <span>Value: {Number(item.feature_value).toFixed(3)}</span>
                                <strong>SHAP: {Number(item.shap_value).toFixed(5)}</strong>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="muted">No SHAP feature explanations are available for this site.</p>
                      )}
                    </article>
                  ))}
                </div>
                <p className="shap-footnote">
                  SHAP values explain how features contribute to the selected class output relative to its baseline.
                  They are not probabilities and do not establish cause and effect. A positive contribution supports
                  the predicted class; a negative contribution opposes it.
                </p>
              </>
            )}
          </section>
        </>
      )}

      {currentPage === 'results' && (
        <>
          <button className="back-btn" onClick={goToSetup}>← Back to Setup</button>

          {validation && (
            <section className="card">
              <h2>Model Accuracy — Held-Out Validation</h2>
              <p className="muted">Trained on the first {validation.train_rows} real readings, tested on the remaining {validation.test_rows}. Lower is better.</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={validationChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 13 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 13 }} label={{ value: 'MAE', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Prophet" fill="var(--chart-prophet)" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Random Forest" fill="var(--chart-rf)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <table className="data-table">
                <thead><tr><th>Metric</th><th>Prophet MAE</th><th>Prophet RMSE</th><th>RF MAE</th><th>RF RMSE</th></tr></thead>
                <tbody>
                  {Object.entries(validation.results).map(([metric, vals]) => {
                    const pBetter = vals.prophet.mae < (vals.random_forest.mae ?? Infinity)
                    return (
                      <tr key={metric}>
                        <td className="row-label">{metric}</td>
                        <td className={pBetter ? 'better' : ''}>{vals.prophet.mae}</td>
                        <td className={pBetter ? 'better' : ''}>{vals.prophet.rmse}</td>
                        <td className={!pBetter ? 'better' : ''}>{vals.random_forest.mae ?? '—'}</td>
                        <td className={!pBetter ? 'better' : ''}>{vals.random_forest.rmse ?? '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {validation.test_rows < 15 && (
                <p className="warning">Only {validation.test_rows} test rows were available — treat this as a directional signal, not a statistically robust result.</p>
              )}
            </section>
          )}

          {summary.length > 0 && (
            <section className="card">
              <h2>Quality Distribution — {model === 'prophet' ? 'Prophet' : 'Random Forest'}</h2>
              <div className="quality-pills">
                {summary.map((row) => (
                  <div key={row.Quality} className="pill" style={{ borderColor: qualityColors[row.Quality] }}>
                    <span className="pill-dot" style={{ background: qualityColors[row.Quality] }} />
                    <span className="pill-label">{row.Quality}</span>
                    <span className="pill-value">{row.Percent}%</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {loading && forecast.length === 0 && (
            <section className="card">
              <div className="skeleton-line skeleton-title"></div>
              <div className="skeleton-chart"></div>
            </section>
          )}

          {forecast.length > 0 && (
            <section className="card">
              <div className="view-toggle-row">
                <div className="tabs">
                  <button className={`tab ${viewMode === 'charts' ? 'tab-active' : ''}`} onClick={() => setViewMode('charts')}>📈 Charts</button>
                  <button className={`tab ${viewMode === 'map' ? 'tab-active' : ''}`} onClick={() => { setViewMode('map'); if (locations.length === 0) loadLocations() }}>🗺 Map View</button>
                </div>
              </div>

              {viewMode === 'charts' && (
                <>
                  <div className="filter-row">
                    <div className="control-group">
                      <label>From</label>
                      <input type="date" value={dateStart} onChange={(e) => setDateStart(e.target.value)} className="date-input" />
                    </div>
                    <div className="control-group">
                      <label>To</label>
                      <input type="date" value={dateEnd} onChange={(e) => setDateEnd(e.target.value)} className="date-input" />
                    </div>
                    <label className="checkbox-label">
                      <input type="checkbox" checked={showHistorical} onChange={(e) => setShowHistorical(e.target.checked)} /> Show historical
                    </label>
                    <label className="checkbox-label">
                      <input type="checkbox" checked={showAnomalies} onChange={(e) => setShowAnomalies(e.target.checked)} /> Highlight anomalies
                    </label>
                  </div>

                  <div className="tabs-row">
                    <div className="tabs">
                      {['RSRP', 'RSRQ', 'MOS'].map((tab) => (
                        <button key={tab} className={`tab ${activeTab === tab ? 'tab-active' : ''}`} onClick={() => setActiveTab(tab)}>{tab}</button>
                      ))}
                    </div>
                    <button onClick={exportCSV} className="btn btn-export">⬇ Export CSV</button>
                  </div>

                  {activeTab === 'RSRP' && (<><h2>RSRP (dBm)</h2>{renderChart('RSRP', 'var(--chart-rsrp)', ['auto', 'auto'])}</>)}
                  {activeTab === 'RSRQ' && (<><h2>RSRQ (dB)</h2>{renderChart('RSRQ', 'var(--chart-rsrq)', ['auto', 'auto'])}</>)}
                  {activeTab === 'MOS' && (<><h2>MOS Score</h2>{renderChart('MOS', 'var(--chart-mos)', [1, 5])}</>)}

                  <div className="legend-note">
                    <span className="dot" style={{ background: '#ef4444' }}></span> Poor &nbsp;
                    <span className="dot" style={{ background: '#f59e0b' }}></span> Average &nbsp;
                    {showHistorical && !compareFilesMode && <span>— dashed line marks where the forecast begins</span>}
                  </div>
                </>
              )}

              {viewMode === 'map' && (
                <>
                  <h2>Drive-Test Locations (Real Data)</h2>
                  {loadingLocations && <p className="muted">Loading location data…</p>}
                  {!loadingLocations && locations.length > 0 && (
                    <MapContainer center={[locations[0].Latitude, locations[0].Longitude]} zoom={14} style={{ height: '420px', width: '100%', borderRadius: '10px' }}>
                      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; OpenStreetMap contributors' />
                      {locations.map((loc, idx) => (
                        <CircleMarker key={idx} center={[loc.Latitude, loc.Longitude]} radius={5}
                          pathOptions={{ color: QUALITY_COLORS[loc.Quality] || '#888', fillColor: QUALITY_COLORS[loc.Quality] || '#888', fillOpacity: 0.7 }}>
                          <Popup>MOS: {loc.MOS} — {loc.Quality}</Popup>
                        </CircleMarker>
                      ))}
                    </MapContainer>
                  )}
                  {!loadingLocations && locations.length === 0 && <p className="muted">No location data available for this file.</p>}
                </>
              )}
            </section>
          )}

          <p className="disclaimer">
            This forecast is extrapolated from real measurements using a synthetically expanded history
            (Random Forest) or seasonal decomposition (Prophet). Only the held-out validation numbers above
            are measured against real data.
          </p>
        </>
      )}
    </div>
  )
}

export default App