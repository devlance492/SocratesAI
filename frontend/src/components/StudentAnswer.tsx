import { useState, useCallback } from "react"
import { Mic, MicOff, Send, Loader2 } from "lucide-react"
import { Button, Card, CardContent } from "@/components/ui"
import { cn } from "@/lib/utils"

interface StudentAnswerProps {
  onSubmitAnswer: (answer: string) => void
  disabled?: boolean
  isSubmitting?: boolean
}

export function StudentAnswer({ onSubmitAnswer, disabled, isSubmitting }: StudentAnswerProps) {
  const [answer, setAnswer] = useState("")
  const [isListening, setIsListening] = useState(false)

  const handleMicClick = useCallback(() => {
    setIsListening(prev => !prev)
    // In a real implementation, this would trigger speech recognition
    if (!isListening) {
      // Simulate listening state
      setTimeout(() => {
        setIsListening(false)
        setAnswer(prev => prev + " [Voice input would appear here]")
      }, 3000)
    }
  }, [isListening])

  const handleSubmit = useCallback(() => {
    if (answer.trim()) {
      onSubmitAnswer(answer.trim())
      setAnswer("")
    }
  }, [answer, onSubmitAnswer])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  return (
    <Card>
      <CardContent className="p-6">
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-4">
          Your Answer
        </p>
        
        <div className="flex gap-4">
          {/* Microphone Button */}
          <Button
            variant={isListening ? "default" : "outline"}
            size="icon-lg"
            onClick={handleMicClick}
            disabled={disabled}
            className={cn(
              "shrink-0 rounded-full transition-all",
              isListening && "bg-red-600 hover:bg-red-700 border-red-600"
            )}
            aria-label={isListening ? "Stop listening" : "Start voice input"}
          >
            {isListening ? (
              <MicOff className="w-6 h-6" />
            ) : (
              <Mic className="w-6 h-6" />
            )}
          </Button>

          {/* Text Input */}
          <div className="flex-1 relative">
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your answer or use the microphone…"
              disabled={disabled}
              rows={3}
              className={cn(
                "w-full px-4 py-3 text-base rounded-lg border border-slate-200",
                "focus:outline-none focus:ring-2 focus:ring-slate-400 focus:border-transparent",
                "resize-none bg-white placeholder:text-slate-400",
                "disabled:bg-slate-50 disabled:text-slate-400"
              )}
            />
            
            {/* Submit Button */}
            <Button
              variant="default"
              size="sm"
              onClick={handleSubmit}
              disabled={disabled || !answer.trim() || isSubmitting}
              className="absolute bottom-3 right-3"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Submit</span>
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Listening Indicator */}
        {isListening && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-600">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            </span>
            Listening…
          </div>
        )}
      </CardContent>
    </Card>
  )
}
