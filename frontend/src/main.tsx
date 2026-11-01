import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { FurnitureWorkbench } from './components/FurnitureWorkbench';
import { LookPaper } from './looks/LookPaper';
import { LookHud } from './looks/LookHud';
import { LookPoster } from './looks/LookPoster';
import './index.css';

// ?look=a (paper manual) | b (camera HUD) | c (brutalist poster); anything else = v2 default.
const LOOK = new URLSearchParams(window.location.search).get('look');

function Root() {
  const [view, setView] = useState<'furniture' | 'cad'>('furniture');
  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,#0f172a,#020617_60%)]">
      <nav className="flex items-center gap-4 px-5 py-3 border-b border-slate-800/80">
        <span className="font-black text-white text-lg tracking-tight">⏪ REWIND</span>
        <span className="text-slate-400 text-sm hidden md:inline">watches your hands and says stop</span>
        <span className="hidden lg:inline text-[11px] text-slate-500 border border-slate-700 rounded-full px-2 py-0.5">fail-closed · blinded · audited</span>
        <div className="ml-auto flex gap-1 text-xs">
          <button onClick={() => setView('furniture')} className={`px-3 py-1.5 rounded-lg ${view === 'furniture' ? 'bg-slate-800 text-white' : 'text-slate-400'}`}>Assembly check</button>
          <button onClick={() => setView('cad')} className={`px-3 py-1.5 rounded-lg ${view === 'cad' ? 'bg-slate-800 text-white' : 'text-slate-500'}`}>Advanced · CAD workbench</button>
        </div>
      </nav>
      {view === 'furniture' ? <FurnitureWorkbench /> : <App />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    {LOOK === 'a' ? <LookPaper /> : LOOK === 'b' ? <LookHud /> : LOOK === 'c' ? <LookPoster /> : <Root />}
  </React.StrictMode>
);
