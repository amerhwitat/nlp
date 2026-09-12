import { useRef, useState } from 'react'

export default function FileDropzone({ onFile, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const accept = '.txt,.md,.csv,.json,.xml,.html,.htm'

  const choose = (file) => file && onFile(file)
  return (
    <div
      className={`dropzone ${dragging ? 'dropzone-active' : ''}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files?.[0]) }}
    >
      <div className="dropzone-icon">𐪀</div>
      <strong>Drop an inscription or document</strong>
      <span>UTF-8 text and structured text files are accepted.</span>
      <button type="button" disabled={disabled} onClick={() => inputRef.current?.click()}>Choose file</button>
      <input ref={inputRef} hidden type="file" accept={accept} disabled={disabled} onChange={(event) => choose(event.target.files?.[0])} />
    </div>
  )
}
