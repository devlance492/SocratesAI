import { CheckCircle2, AlertCircle, XCircle } from "lucide-react"
import { Card, CardContent, Badge } from "@/components/ui"
import { cn } from "@/lib/utils"

export type Verdict = "correct" | "partial" | "incorrect" | null

interface FeedbackPanelProps {
  score: number | null
  maxScore?: number
  verdict: Verdict
  cognitiveLevel?: number | null
}

const verdictConfig = {
  correct: {
    label: "Correct",
    variant: "success" as const,
    icon: CheckCircle2,
  },
  partial: {
    label: "Partially Correct",
    variant: "warning" as const,
    icon: AlertCircle,
  },
  incorrect: {
    label: "Incorrect",
    variant: "error" as const,
    icon: XCircle,
  },
}

const cognitiveLevelLabels: Record<number, string> = {
  3: "Clarify Misconceptions",
  4: "Analyze / Compare",
  5: "Critique Assumptions",
  6: "Propose Alternatives",
}

export function FeedbackPanel({ score, maxScore = 10, verdict, cognitiveLevel }: FeedbackPanelProps) {
  if (score === null && verdict === null) return null

  const config = verdict ? verdictConfig[verdict] : null
  const Icon = config?.icon

  return (
    <Card className="bg-slate-50 border-slate-100">
      <CardContent className="p-5">
        <div className="flex items-center justify-between flex-wrap gap-2">
          {/* Score Display */}
          <div className="flex items-baseline gap-1">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide mr-2">
              Score
            </span>
            <span className={cn(
              "text-2xl font-semibold",
              score !== null && score >= 7 && "text-emerald-600",
              score !== null && score >= 4 && score < 7 && "text-amber-600",
              score !== null && score < 4 && "text-red-600"
            )}>
              {score ?? "—"}
            </span>
            <span className="text-slate-400 text-lg">/ {maxScore}</span>
          </div>

          {/* Verdict Badge */}
          <div className="flex items-center gap-2">
            {cognitiveLevel && (
              <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                Level {cognitiveLevel}: {cognitiveLevelLabels[cognitiveLevel] || "Unknown"}
              </span>
            )}
            {config && Icon && (
              <Badge variant={config.variant} className="text-sm">
                <Icon className="w-3.5 h-3.5" />
                {config.label}
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
