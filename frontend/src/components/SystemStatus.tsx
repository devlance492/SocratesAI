import * as React from "react"
import { Zap, Brain, Mic } from "lucide-react"
import { cn } from "@/lib/utils"

interface StatusIndicatorProps {
  label: string
  icon: React.ReactNode
  active?: boolean
}

function StatusIndicator({ label, icon, active = true }: StatusIndicatorProps) {
  return (
    <div className={cn(
      "flex items-center gap-1.5 text-xs",
      active ? "text-slate-500" : "text-slate-300"
    )}>
      {icon}
      <span>{label}</span>
    </div>
  )
}

interface SystemStatusProps {
  voiceActive?: boolean
  adaptiveDifficulty?: boolean
  backendConnected?: boolean
}

export function SystemStatus({ 
  voiceActive = true, 
  adaptiveDifficulty = true,
  backendConnected = false 
}: SystemStatusProps) {
  return (
    <div className="flex items-center justify-center gap-6 py-4">
      <StatusIndicator
        label={backendConnected ? "Backend Connected" : "Backend Offline"}
        icon={<Zap className="w-3.5 h-3.5" />}
        active={backendConnected}
      />
      <StatusIndicator
        label="Adaptive Difficulty"
        icon={<Brain className="w-3.5 h-3.5" />}
        active={adaptiveDifficulty}
      />
      <StatusIndicator
        label="Voice Input"
        icon={<Mic className="w-3.5 h-3.5" />}
        active={voiceActive}
      />
    </div>
  )
}
