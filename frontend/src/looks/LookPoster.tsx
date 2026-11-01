import { useRewindCheck, FRAME, REFERENCE, SCENARIO_LABEL, fixHints, lanes, Phase } from './useRewindCheck';
import { FrameSvg, FlipHint } from './Shared';

// Look C, brutalist / editorial: safety orange + black, "MANUAL vs YOU", huge counters.
const OR = '#ff5a1f';
const HEAD: Record<Phase, string> = { idle: 'READY.', checking: 'LOOKING.', stop: 'STOP.', ok: 'OK.', provisional: 'LOOKS RIGHT.', unknown: "CAN'T TELL.", error: 'ERROR.' };

export function LookPoster() {
  const c = useRewindCheck();
  const { phase, s1, res } = c;
  const hints = fixHints(res);
  const hp = s1?.hole_pattern_witness;
  const holes = hp?.live_holes_px ?? [];
  const stop = phase === 'stop';
  const bg = stop ? OR : '#f2f0eb';
  return (
    <div className="min-h-screen h-screen flex flex-col font-poster text-black" style={{ background: bg }} data-look="poster" data-phase={phase}>
      {/* marquee */}
      <div className="bg-black text-white overflow-hidden whitespace-nowrap py-1 text-sm font-mono-tech">
        <div className="inline-block rw-marquee">{Array(2).fill(0).map((_, k) => <span key={k}>{Array(6).fill(`REWIND · WATCHES YOUR HANDS AND SAYS STOP · FAIL-CLOSED · ${c.fixture ? 'FIXTURE MODE · NO MODEL CALLS' : 'LIVE MODE'} · ${c.cfg?.synthetic ? 'SYNTHETIC FRAMES · ' : ''}`).join('')}</span>)}</div>
      </div>
      <header className="flex items-stretch border-b-[6px] border-black">
        <div className="px-6 py-2 text-6xl font-black border-r-[6px] border-black">REWIND</div>
        <div className="px-6 py-2 flex flex-col justify-center border-r-[6px] border-black"><div className="text-xs font-bold">STEP</div><div className="text-5xl font-black leading-none">{String(c.stepNum).padStart(2, '0')}<span className="text-2xl opacity-40">/{String(c.total).padStart(2, '0')}</span></div></div>
        <div className="px-6 py-2 flex items-center text-2xl font-bold flex-1">{c.step?.title}</div>
        <a href="?look=v2" className="px-6 flex items-center underline border-l-[6px] border-black text-sm">classic view</a>
      </header>

      <div className="relative flex-1 grid grid-cols-2 min-h-0">
        {/* MANUAL */}
        <section className="border-r-[6px] border-black flex flex-col min-h-0">
          <div className="flex items-baseline justify-between px-6 pt-3"><h2 className="text-[110px] font-black leading-[.8]">MANUAL</h2><div className="text-right"><div className="text-[110px] font-black leading-[.8]">{hp?.reference_hole_count ?? '—'}</div><div className="text-sm font-bold">HOLES EXPECTED</div></div></div>
          <div className="flex-1 min-h-0 m-6 border-[6px] border-black bg-white"><img src={REFERENCE} alt="manual reference" className="w-full h-full object-contain grayscale contrast-125" /></div>
        </section>
        {/* YOU */}
        <section className="flex flex-col min-h-0 relative">
          <div className="flex items-baseline justify-between px-6 pt-3"><h2 className="text-[110px] font-black leading-[.8]">YOU</h2><div className="text-right"><div className={`text-[110px] font-black leading-[.8] ${stop ? 'text-white' : ''}`}>{hp ? hp.live_hole_count : '—'}</div><div className="text-sm font-bold">HOLES SEEN</div></div></div>
          <div className="relative flex-1 min-h-0 m-6 border-[6px] border-black bg-black">
            <FrameSvg src={FRAME[c.scenario]} holes={holes} className="w-full h-full" markerColor={OR} strokeW={6} />
            {phase === 'checking' && <div className="rw-scanline" style={{ background: OR, boxShadow: 'none' }} />}
            <div className="absolute left-0 top-0 bg-black text-white px-2 text-xs font-mono-tech">{c.fixture ? 'FIXTURE FRAME' : 'CAMERA'}</div>
          </div>
        </section>
        {/* giant verdict block across the seam */}
        <div className="absolute left-1/2 -translate-x-1/2 top-[44%] pointer-events-none" role="status" aria-live="assertive" data-testid="furniture-verdict">
          <div key={phase} className={`rw-pop bg-black px-10 py-4 font-black leading-none whitespace-nowrap ${stop ? 'text-[220px]' : 'text-[120px]'}`} style={{ color: stop ? OR : phase === 'ok' ? '#34d399' : '#fff' }}>{HEAD[phase]}</div>
        </div>
      </div>

      <footer className="border-t-[6px] border-black grid grid-cols-[1.2fr_1fr_auto] min-h-[190px]">
        <div className="p-4 border-r-[6px] border-black">
          {hints.length ? <div data-testid="fix-hint"><div className="text-sm font-black">FIX ↓ (deterministic, from the failed check)</div>{hints.map((h) => <div key={h.kind} className="flex items-center gap-4"><FlipHint kind={h.kind} size={150} /><div className="text-3xl font-black leading-tight">{h.text.toUpperCase()}</div></div>)}</div>
            : <div className="text-3xl font-black leading-tight">{phase === 'provisional' ? 'LOCAL CHECKS PASS. FIXTURE MODE: NOT MODEL-CONFIRMED, MANUAL NOT TICKED.' : phase === 'ok' ? 'BOTH STAGES AGREE. NEXT STEP.' : phase === 'checking' ? (c.mode === 'fixture' ? 'FIXTURE: LOCAL WITNESSES ONLY.' : 'SAM 3.1 + MUSE SPARK OPEN LOOKING.') : 'HOLD THE PART UP. PRESS SPACE.'}</div>}
        </div>
        <table className="text-sm font-mono-tech border-r-[6px] border-black" data-testid="model-trace"><tbody>
          {(lanes(res).length ? lanes(res) : [{ model: 'NO RUN YET', state: 'not_measured', value: '' }]).map((l: any) => <tr key={l.model} className="border-b-2 border-black"><td className="px-3 py-1 font-bold">{l.model}</td><td className={`px-3 ${l.state === 'not_measured' ? 'opacity-50' : ''}`}>{l.value}</td></tr>)}
          <tr><td className="px-3 py-1 font-bold">Manual</td><td className="px-3">{res?.manual?.label ?? 'Manual not updated'}</td></tr>
        </tbody></table>
        <div className="flex flex-col p-3 gap-2 w-[420px]">
          <div className="flex gap-1">{(c.cfg?.scenarios ?? []).map((s, i) => <button key={s} onClick={() => c.setScenario(s)} className={`flex-1 border-[3px] border-black px-1 py-1 text-xs font-bold ${c.scenario === s ? 'bg-black text-white' : 'bg-white'}`}>{i + 1} {SCENARIO_LABEL[s]}</button>)}</div>
          <select value={c.mode} onChange={(e) => c.setMode(e.target.value as any)} className="border-[3px] border-black bg-white font-bold"><option value="fixture">FIXTURE</option><option value="live">LIVE</option></select>
          <button onClick={c.run} disabled={phase === 'checking'} className="flex-1 bg-black text-white text-4xl font-black hover:opacity-90">CHECK ␣</button>
          {c.err && <div className="text-xs">{c.err}</div>}
        </div>
      </footer>
    </div>
  );
}
