import { useState, useCallback, useEffect } from 'react'
import {
  Header,
  PDFUpload,
  DocumentStatusBar,
  ExaminerQuestion,
  StudentAnswer,
  FeedbackPanel,
  SystemStatus,
  type DocumentStatus,
  type Verdict,
} from '@/components'
import {
  uploadPdf,
  getQuestion,
  submitAnswer,
  checkHealth,
  type ClaimResult,
} from '@/lib/api'


// Examination state
interface ExaminationState {
  sessionId: string | null
  documentStatus: DocumentStatus
  sectionCount: number | undefined
  currentQuestion: string | null
  isLoadingQuestion: boolean
  isSubmitting: boolean
  score: number | null
  verdict: Verdict
  questionCount: number
  cognitiveLevel: number | null
  claims: ClaimResult[]
  error: string | null
  backendAvailable: boolean
}

const initialState: ExaminationState = {
  sessionId: null,
  documentStatus: 'idle',
  sectionCount: undefined,
  currentQuestion: null,
  isLoadingQuestion: false,
  isSubmitting: false,
  score: null,
  verdict: null,
  questionCount: 0,
  cognitiveLevel: null,
  claims: [],
  error: null,
  backendAvailable: false,
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [state, setState] = useState<ExaminationState>(initialState)

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      const available = await checkHealth()
      setState(prev => ({ ...prev, backendAvailable: available }))
      
      if (!available) {
        setState(prev => ({
          ...prev,
          error: 'Backend server not available. Please start the API server.',
        }))
      }
    }
    
    checkBackend()
    // Check every 5 seconds if not available
    const interval = setInterval(async () => {
      const available = await checkHealth()
      setState(prev => ({
        ...prev,
        backendAvailable: available,
        error: available ? null : 'Backend server not available. Please start the API server.',
      }))
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  // Handle PDF upload
  const handleFileSelected = useCallback(async (file: File) => {
    setSelectedFile(file)
    setState(prev => ({
      ...prev,
      documentStatus: 'processing',
      currentQuestion: null,
      score: null,
      verdict: null,
      error: null,
      claims: [],
    }))

    try {
      // Upload to backend
      const uploadResponse = await uploadPdf(file)
      
      setState(prev => ({
        ...prev,
        sessionId: uploadResponse.session_id,
        documentStatus: 'ready',
        sectionCount: uploadResponse.section_count,
        isLoadingQuestion: true,
      }))

      // Get first question
      const questionResponse = await getQuestion(uploadResponse.session_id)
      
      setState(prev => ({
        ...prev,
        isLoadingQuestion: false,
        currentQuestion: questionResponse.question,
        questionCount: questionResponse.question_number,
      }))
    } catch (error) {
      console.error('Upload error:', error)
      setState(prev => ({
        ...prev,
        documentStatus: 'error',
        error: error instanceof Error ? error.message : 'Failed to upload document',
      }))
    }
  }, [])

  // Handle answer submission
  const handleSubmitAnswer = useCallback(async (answer: string) => {
    if (!state.sessionId) {
      setState(prev => ({ ...prev, error: 'No active session' }))
      return
    }

    setState(prev => ({
      ...prev,
      isSubmitting: true,
      error: null,
    }))

    try {
      // Submit answer to backend
      const evaluation = await submitAnswer(state.sessionId, answer)
      
      setState(prev => ({
        ...prev,
        isSubmitting: false,
        score: evaluation.score,
        verdict: evaluation.verdict,
        claims: evaluation.claims,
        cognitiveLevel: evaluation.cognitive_level,
        currentQuestion: evaluation.followup_question,
        questionCount: evaluation.question_number,
      }))
    } catch (error) {
      console.error('Submit error:', error)
      setState(prev => ({
        ...prev,
        isSubmitting: false,
        error: error instanceof Error ? error.message : 'Failed to submit answer',
      }))
    }
  }, [state.sessionId])

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-[900px] mx-auto px-6 pb-16">
        {/* Header */}
        <Header />

        {/* Backend Status Warning */}
        {!state.backendAvailable && (
          <div className="mb-6 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
            <strong>Backend not available.</strong> Start the API server with:{' '}
            <code className="bg-amber-100 px-1 py-0.5 rounded">python api_server.py</code>
          </div>
        )}

        {/* Error Display */}
        {state.error && state.backendAvailable && (
          <div className="mb-6 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
            {state.error}
          </div>
        )}

        {/* Main Content */}
        <main className="space-y-6">
          {/* PDF Upload */}
          <PDFUpload
            onFileSelected={handleFileSelected}
            selectedFile={selectedFile}
          />

          {/* Document Status */}
          <DocumentStatusBar
            status={state.documentStatus}
            sectionCount={state.sectionCount}
            fileName={selectedFile?.name}
          />

          {/* Examiner Question */}
          <ExaminerQuestion
            question={state.currentQuestion}
            isLoading={state.isLoadingQuestion}
          />

          {/* Student Answer */}
          <StudentAnswer
            onSubmitAnswer={handleSubmitAnswer}
            disabled={
              state.documentStatus !== 'ready' ||
              state.isLoadingQuestion ||
              state.isSubmitting ||
              !state.backendAvailable
            }
            isSubmitting={state.isSubmitting}
          />

          {/* Feedback Panel */}
          <FeedbackPanel
            score={state.score}
            verdict={state.verdict}
            cognitiveLevel={state.cognitiveLevel}
          />

          {/* Claims Detail (optional - show if there are claims) */}
          {state.claims.length > 0 && (
            <ClaimsPanel claims={state.claims} />
          )}

          {/* System Status */}
          <SystemStatus
            voiceActive={true}
            adaptiveDifficulty={true}
            backendConnected={state.backendAvailable}
          />
        </main>
      </div>
    </div>
  )
}

// Claims breakdown panel component
function ClaimsPanel({ claims }: { claims: ClaimResult[] }) {
  const [expanded, setExpanded] = useState(false)
  
  if (claims.length === 0) return null
  
  return (
    <div className="bg-white rounded-lg border border-slate-100 shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
      >
        <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
          Claim Analysis ({claims.length} claims)
        </span>
        <span className="text-slate-400 text-sm">
          {expanded ? '▲' : '▼'}
        </span>
      </button>
      
      {expanded && (
        <div className="px-5 pb-4 space-y-3">
          {claims.map((claim, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg text-sm ${
                claim.verdict === 'CORRECT'
                  ? 'bg-emerald-50 border border-emerald-100'
                  : claim.verdict === 'PARTIALLY_CORRECT'
                  ? 'bg-amber-50 border border-amber-100'
                  : 'bg-red-50 border border-red-100'
              }`}
            >
              <p className="text-slate-700">"{claim.claim}"</p>
              <p className="mt-1 text-xs text-slate-500">
                {claim.verdict.replace('_', ' ')} • Score: {claim.score}/10
              </p>
              {claim.reasoning && (
                <p className="mt-1 text-xs text-slate-600 italic">
                  {claim.reasoning}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App
