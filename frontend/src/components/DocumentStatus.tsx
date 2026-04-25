import { Brain, CheckCircle2, FileText, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

export type DocumentStatus = "idle" | "processing" | "ready" | "error"

interface DocumentStatusProps {
  status: DocumentStatus
  sectionCount?: number
  fileName?: string
}

export function DocumentStatusBar({ status, sectionCount, fileName }: DocumentStatusProps) {
  if (status === "idle") return null

  return (
    <div className={cn(
      "flex items-center gap-3 px-5 py-3 rounded-lg text-sm",
      status === "processing" && "bg-slate-100 text-slate-600",
      status === "ready" && "bg-emerald-50 text-emerald-700",
      status === "error" && "bg-red-50 text-red-700"
    )}>
      {status === "processing" && (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>Processing document…</span>
        </>
      )}
      
      {status === "ready" && (
        <>
          <CheckCircle2 className="w-4 h-4" />
          <span>Document ingested successfully</span>
          {sectionCount !== undefined && (
            <span className="text-slate-500 ml-1">
              • {sectionCount} sections detected
            </span>
          )}
        </>
      )}

      {status === "error" && (
        <>
          <FileText className="w-4 h-4" />
          <span>Error processing document</span>
        </>
      )}
      
      {fileName && status === "ready" && (
        <div className="ml-auto flex items-center gap-2 text-slate-500">
          <Brain className="w-4 h-4" />
          <span className="text-xs">Ready for examination</span>
        </div>
      )}
    </div>
  )
}
