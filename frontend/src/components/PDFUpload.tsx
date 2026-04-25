import { useCallback, useState } from "react"
import { FileText, Upload } from "lucide-react"
import { Card, CardContent } from "@/components/ui"
import { cn } from "@/lib/utils"

interface PDFUploadProps {
  onFileSelected: (file: File) => void
  selectedFile: File | null
}

export function PDFUpload({ onFileSelected, selectedFile }: PDFUploadProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0 && files[0].type === "application/pdf") {
      onFileSelected(files[0])
    }
  }, [onFileSelected])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onFileSelected(files[0])
    }
  }, [onFileSelected])

  return (
    <Card className={cn(
      "transition-colors duration-150",
      isDragging && "border-slate-400 bg-slate-50"
    )}>
      <CardContent className="p-8">
        <label
          htmlFor="pdf-upload"
          className="cursor-pointer block"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-4">
            <div className={cn(
              "w-16 h-16 rounded-full flex items-center justify-center transition-colors",
              selectedFile 
                ? "bg-emerald-50 text-emerald-600" 
                : "bg-slate-100 text-slate-500"
            )}>
              {selectedFile ? (
                <FileText className="w-7 h-7" />
              ) : (
                <Upload className="w-7 h-7" />
              )}
            </div>
            
            {selectedFile ? (
              <div className="text-center">
                <p className="text-sm font-medium text-slate-800">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Click or drag to replace
                </p>
              </div>
            ) : (
              <div className="text-center">
                <p className="text-sm font-medium text-slate-700">
                  Upload Research Paper (PDF)
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Drag and drop or click to browse
                </p>
              </div>
            )}
          </div>
          
          <input
            id="pdf-upload"
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
            className="sr-only"
          />
        </label>
      </CardContent>
    </Card>
  )
}
