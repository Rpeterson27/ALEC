import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [apiData, setApiData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [submittedName, setSubmittedName] = useState<string | null>(null)
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
        <p>Accent and Language Learning Coach</p>
      </div>
      
      <div className="card">
        {submittedName ? (
          <h2>Hi, {submittedName}!</h2>
        ) : (
          <>
            <h2>Enter Your Name</h2>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              onKeyDown={(e) => e.key === 'Enter' && submitName()}
            />
            <button onClick={submitName} disabled={loading}>
              Submit Name
            </button>
          </>
        )}
      </div>

      {submittedName && (
        <div className="card">
          <h2>Audio Recording</h2>
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
          
          {isRecording && <p>🎤 Recording...</p>}
          {audioBlob && !isRecording && <p>✅ Audio recorded successfully</p>}
        </div>
      )}
    </>
  )
}

export default App
