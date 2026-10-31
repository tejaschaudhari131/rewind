import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { FurnitureWorkbench } from './components/FurnitureWorkbench';
import './index.css';

function Root() {
  const [view, setView] = useState<'furniture' | 'cad'>('furniture');
  return (
    <div className="min-h-screen bg-slate-950">
      <nav className="flex gap-2 px-4 py-2 border-b border-slate-800 bg-slate-900 text-xs">
        <span className="font-bold text-white mr-3">REWIND</span>
        <button onClick={() => setView('furniture')} className={view === 'furniture' ? 'text-sky-300' : 'text-slate-400'}>Flat-pack assembly check</button>
        <button onClick={() => setView('cad')} className={view === 'cad' ? 'text-sky-300' : 'text-slate-400'}>Slider-crank / CAD workbench (advanced)</button>
      </nav>
      {view === 'furniture' ? <FurnitureWorkbench /> : <App />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
