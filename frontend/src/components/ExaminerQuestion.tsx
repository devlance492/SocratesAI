import { Card, CardContent } from "@/components/ui"

interface ExaminerQuestionProps {
  question: string | null
  isLoading?: boolean
}

export function ExaminerQuestion({ question, isLoading }: ExaminerQuestionProps) {
  return (
    <Card className="border-l-4 border-l-slate-700">
      <CardContent className="p-6">
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-4">
          Examiner Question
        </p>
        
        {isLoading ? (
          <div className="space-y-3">
            <div className="h-6 bg-slate-100 rounded animate-pulse w-full" />
            <div className="h-6 bg-slate-100 rounded animate-pulse w-4/5" />
          </div>
        ) : question ? (
          <p className="text-xl leading-relaxed text-slate-800 font-medium">
            {question}
          </p>
        ) : (
          <p className="text-lg text-slate-400 italic">
            Upload a document to begin the examination.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
