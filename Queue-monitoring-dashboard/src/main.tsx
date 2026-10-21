import React from 'react'
import ReactDOM from 'react-dom/client'
import { ThemeProvider } from '@figma/astraui'
import App from './App'
import './index.css'

// Operations console is a fixed dark control-room surface.
try {
  localStorage.setItem('astra-theme', 'dark')
} catch {}
document.documentElement.classList.add('dark')

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
)
