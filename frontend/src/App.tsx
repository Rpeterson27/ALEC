import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [apiData, setApiData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [submittedName, setSubmittedName] = useState<string | null>(null)
  const [nativeLanguage, setNativeLanguage] = useState('')
  const [targetLanguage, setTargetLanguage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

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

  const submitName = async () => {
    if (!name.trim()) {
      setError('Please enter a name')
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/submit-name', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: name.trim() })
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setApiData(data)
      setSubmittedName(name.trim())
      setName('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const startRecording = async () => {
    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      
      const chunks: BlobPart[] = []
      
      mediaRecorder.ondataavailable = (event) => {
        chunks.push(event.data)
      }
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' })
        setAudioBlob(audioBlob)
      }
      
      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      setError('Failed to access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      streamRef.current?.getTracks().forEach(track => track.stop())
      setIsRecording(false)
    }
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  const processAudio = async () => {
    if (!audioBlob) {
      setError('No audio recorded')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.wav')
      
      const response = await fetch('/api/process-audio', {
        method: 'POST',
        body: formData
      })
      
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
        <h1>ALEC</h1>
        <p>Accent and Language Education Coach</p>
      </div>
      
      {submittedName ? (
        <div className="greeting-header">
          <h2>Hi, {submittedName}!</h2>
        </div>
      ) : (
        <div className="card">
          <h2>Enter Your Name</h2>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter your name"
            onKeyDown={(e) => e.key === 'Enter' && submitName()}
          />
          <button onClick={submitName} disabled={loading} style={{marginLeft: '1rem'}}>
            Submit Name
          </button>
        </div>
      )}

      {submittedName && (
        <div className="card">
          <h2>Language Selection</h2>
          <div style={{marginBottom: '15px'}}>
            <label htmlFor="nativeLanguage" style={{display: 'block', marginBottom: '5px'}}>
              What language do you speak?
            </label>
            <select
              id="nativeLanguage"
              value={nativeLanguage}
              onChange={(e) => setNativeLanguage(e.target.value)}
              style={{width: '100%', padding: '8px'}}
            >
              <option value="">Select your native language</option>
              <option value="English">English</option>
            </select>
          </div>
          
          <div style={{marginBottom: '15px'}}>
            <label htmlFor="targetLanguage" style={{display: 'block', marginBottom: '5px'}}>
              What language would you like to learn?
            </label>
            <select
              id="targetLanguage"
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              style={{width: '100%', padding: '8px'}}
            >
              <option value="">Select target language</option>
              <option value="French">French</option>
            </select>
          </div>
        </div>
      )}

      {submittedName && nativeLanguage && targetLanguage && (
        <div className="card">
          <h2>Record Your Pronunciation</h2>
          <button 
            onClick={toggleRecording} 
            disabled={loading}
            style={{
              backgroundColor: isRecording ? '#dc3545' : '#007bff',
              color: 'white',
              marginRight: '10px'
            }}
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
          
          {audioBlob && (
            <button onClick={processAudio} disabled={loading || isRecording}>
              Process Audio
            </button>
          )}
          
          {isRecording && <div className="recording-indicator recording">🎤 Recording...</div>}
        </div>
      )}
    </>
  )
}

export default App
