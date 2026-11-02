import { useRewindCheck, FRAME, REFERENCE, SCENARIO_LABEL, fixHints, lanes, Phase } from './useRewindCheck';
import { FlipHint } from './Shared';
import { FrameStage, SourceSwitch, ApiBanner } from './FrameInput';

// Look B, full-bleed camera HUD: the camera feed is the UI. Kinetic verdict letters, AR hole markers, ghost of the correct orientation, session rewind scrubber.
const WORD: Record<Phase, string> = { idle: 'READY', checking: 'LOOKING', stop: 'STOP', ok: 'OK', provisional: 'LOOKS RIGHT', unknown: "CAN'T TELL", error: 'ERROR' };
const COL: Record<Phase, string> = { idle: '#e2e8f0', checking: '#67e8f9', stop: '#ff2d55', ok: '#34d399', provisional: '#fbbf24', unknown: '#a3a3a3', error: '#ff2d55' };
const DOT: Record<Phase, string> = { idle: '#64748b', checking: '#67e8f9', stop: '#ff2d55', ok: '#34d399', provisional: '#fbbf24', unknown: '#a3a3a3', error: '#ff2d55' };

export function LookHud() {
  const c = useRewindCheck();
  const { phase, s1, res } = c;
  const hints = fixHints(res);
  const holes = s1?.hole_pattern_witness?.live_holes_px ?? [];
  const rec = new URLSearchParams(window.location.search).get('record') === '1';
  const col = COL[phase];
  const word = WORD[phase];
  return (
    <div className="fixed inset-0 bg-black text-white overflow-hidden font-poster" data-look="hud" data-phase={phase}>
      <ApiBanner c={c} />
      <FrameStage c={c} ghost={phase === 'stop' ? REFERENCE : null} fit="slice" markerColor={col} dark className="absolute inset-0 w-full h-full" />
      <div className="absolute inset-0 pointer-events-none" style={{ background: `radial-gradient(ellipse at center, transparent 45%, ${phase === 'stop' ? 'rgba(255,45,85,.45)' : 'rgba(0,0,0,.7)'} 100%)` }} />
      {phase === 'stop' && <div className="absolute inset-0 border-[14px] rw-blink pointer-events-none" style={{ borderColor: col }} />}
      {phase === 'checking' && <div className="rw-scanline" />}
      {/* corner brackets */}
      {['left-8 top-8 border-l-4 border-t-4', 'right-8 top-8 border-r-4 border-t-4', 'left-8 bottom-36 border-l-4 border-b-4', 'right-8 bottom-36 border-r-4 border-b-4'].map((k) => <div key={k} className={`absolute w-16 h-16 ${k}`} style={{ borderColor: col }} />)}

      {/* top HUD bar */}
      <div className="absolute top-10 left-28 right-28 flex items-center gap-4 text-sm font-mono-tech">
        <span className="font-black text-xl tracking-tight">⏪ REWIND</span>
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-full rw-blink" style={{ background: DOT[phase] }} />{c.realInput ? c.inputLabel : c.fixture ? 'FIXTURE FEED · not a live camera' : 'LIVE · fixture frame'}</span>
        {c.cfg?.synthetic && !c.realInput && <span className="bg-yellow-300 text-black px-2 font-bold">SYNTHETIC</span>}
        <span className="opacity-70">STEP {c.stepNum}/{c.total} · {c.step?.title}</span>
        {!rec && <a href="?look=v2" className="ml-auto underline opacity-60 pointer-events-auto">classic view</a>}
      </div>

      {/* kinetic verdict */}
      <div className="absolute left-0 right-0 top-[16%] text-center pointer-events-none" role="status" aria-live="assertive" data-testid="furniture-verdict">
        <div key={word + c.history.length} className="font-black leading-none" style={{ fontSize: word.length > 6 ? 170 : 300, color: col, textShadow: '0 10px 60px rgba(0,0,0,.6)' }}>
          {word.split('').map((ch, i) => <span key={i} className="rw-letter" style={{ animationDelay: `${i * 55}ms` }}>{ch === ' ' ? '\u00a0' : ch}</span>)}
        </div>
        {s1 && <div className="text-3xl font-bold mt-2 drop-shadow-lg">
          {phase === 'stop' ? `${s1.hole_pattern_witness?.live_hole_count ?? 0} / ${s1.hole_pattern_witness?.reference_hole_count} holes visible · panel is flipped` : phase === 'provisional' ? (c.fixture ? 'Local checks pass · fixture mode, not model-confirmed' : 'Local checks pass · model did not confirm, not ticked') : phase === 'unknown' ? 'Not enough evidence · will not guess' : phase === 'ok' ? 'Both stages agree' : ''}
        </div>}
        {phase === 'stop' && <div className="text-base mt-2 opacity-80 font-mono-tech">faint overlay = the manual's reference frame (how it should look)</div>}
      </div>

      {/* fix card */}
      {hints.length > 0 && (
        <div className="absolute right-12 top-[44%] bg-white text-black rounded-2xl p-4 w-[380px] shadow-2xl" data-testid="fix-hint">
          <div className="text-xs font-black tracking-widest mb-1">FIX</div>
          {hints.map((h) => <div key={h.kind} className="flex items-center gap-3"><FlipHint kind={h.kind} size={130} /><div className="text-xl font-bold leading-tight">{h.text}</div></div>)}
          <div className="text-[10px] opacity-60 mt-1">deterministic, from the failed check</div>
        </div>
      )}

      {/* model ticker */}
      <div className="absolute left-12 top-[44%] space-y-1 font-mono-tech text-sm" data-testid="model-trace">
        {lanes(res).map((l) => (
          <div key={l.model} className="bg-black/60 backdrop-blur px-3 py-1 rounded flex gap-3 w-[340px]"><span className="font-bold w-36">{l.model}</span><span className={l.state === 'not_measured' ? 'opacity-50' : ''}>{l.value}</span></div>
        ))}
        {res && <div className="bg-black/60 px-3 py-1 rounded w-[340px]">Manual: <b>{res.manual?.label}</b></div>}
      </div>

      {/* bottom: rewind scrubber + controls */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black via-black/90 to-transparent px-12 pt-6 flex items-center gap-6">
        <div className="flex-1">
          <div className="text-xs font-mono-tech opacity-60 mb-1">⏪ REWIND THIS SESSION · {c.history.length} real result{c.history.length === 1 ? '' : 's'} (click to restore, no new call)</div>
          <div className="relative h-10 flex items-center gap-1">
            <div className="absolute left-0 right-0 top-1/2 h-[3px] bg-white/20" />
            {c.history.length === 0 && <span className="relative text-sm opacity-50">no checks yet. Press space</span>}
            {c.history.map((h, i) => (
              <button key={i} onClick={() => c.rewindTo(h)} title={`${h.at} · ${SCENARIO_LABEL[h.scenario]}`} className="relative w-7 h-7 rounded-full border-2 border-black hover:scale-125 transition" style={{ background: DOT[h.phase] }} />
            ))}
          </div>
        </div>
        {!rec && <div className="flex gap-2 items-center flex-wrap max-w-[900px]">
          <SourceSwitch c={c} compact={false} btn="px-3 py-2 rounded-lg text-sm font-bold bg-white/10" active="!bg-white text-black" />
          {c.source === 'fixture' && (c.cfg?.scenarios ?? []).map((s, i) => (
            <button key={s} onClick={() => c.setScenario(s)} className={`px-3 py-2 rounded-lg text-sm font-bold ${c.scenario === s ? 'bg-white text-black' : 'bg-white/10'}`}>{i + 1} {SCENARIO_LABEL[s]}</button>
          ))}
          <select value={c.mode} onChange={(e) => c.setMode(e.target.value as any)} className="bg-white/10 rounded-lg px-2 text-sm"><option value="fixture">Fixture</option><option value="live">Live</option></select>
        </div>}
        <button onClick={c.run} disabled={phase === 'checking'} className="w-24 h-24 rounded-full border-[6px] border-white grid place-items-center" aria-label="Check this step">
          <span className="w-16 h-16 rounded-full" style={{ background: col }} />
        </button>
      </div>
      {c.err && <div className="absolute top-20 left-28 text-red-400">{c.err}</div>}
    </div>
  );
}
