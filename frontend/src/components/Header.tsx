interface HeaderProps {
  title?: string
  subtitle?: string
}

export function Header({ 
  title = "Socrates AI",
  subtitle = "Adaptive Viva Voce Examiner for Research Papers"
}: HeaderProps) {
  return (
    <header className="text-center py-10">
      <h1 className="text-3xl font-semibold text-slate-800 tracking-tight">
        {title}
      </h1>
      <p className="mt-2 text-sm text-slate-500 font-normal">
        {subtitle}
      </p>
    </header>
  )
}
