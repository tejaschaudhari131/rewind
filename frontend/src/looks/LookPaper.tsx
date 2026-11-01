import { useRewindCheck, FRAME, REFERENCE, SCENARIO_LABEL, fixHints, lanes, Phase } from './useRewindCheck';
import { FrameSvg, FlipHint } from './Shared';

// Look A, "Paper manual comes alive": a flat-pack instruction page with a line-art helper character.
const INK = '#111';

function Buddy({ phase }: { phase: Phase }) {
  const sad = phase === 'stop' || phase === 'error';
  const happy = phase === 'ok';
  const unsure = phase === 'unknown' || phase === 'provisional';
  const thinking = phase === 'checking';
  return (
    <svg viewBox="0 0 200 260" className={`w-full h-full ${sad ? 'rw-wobble' : happy ? 'rw-bounce' : ''}`} aria-label={`helper ${phase}`}>
      <g fill="none" stroke={INK} strokeWidth="6" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="100" cy="62" rx="36" ry="40" fill="#fff" />
        {/* eyes */}
        {happy ? <><path d="M84 58 q6 -8 12 0" /><path d="M106 58 q6 -8 12 0" /></>
          : <><circle cx="88" cy="58" r="3" fill={INK} /><circle cx="112" cy="58" r="3" fill={INK} /></>}
        {/* brows */}
        {sad && <><path d="M78 44 l16 6" /><path d="M122 44 l-16 6" /></>}
        {unsure && <path d="M106 46 l14 -4" />}
        {/* mouth */}
        {happy ? <path d="M84 76 q16 16 32 0" /> : sad ? <path d="M86 86 q14 -12 28 0" /> : thinking ? <circle cx="100" cy="82" r="5" /> : <path d="M88 80 h24" />}
        {/* body */}
        <path d="M100 102 v80" />
        <path d="M100 182 l-26 60" /><path d="M100 182 l26 60" />
        {/* arms */}
        {happy ? <><path d="M100 128 l-40 -10" /><path d="M100 128 l36 -34" /><path d="M136 94 v-16" strokeWidth="8" /></>
          : sad ? <><path d="M100 128 l-34 -36" /><path d="M100 128 l34 -36" /><path d="M58 84 l16 0" /><path d="M126 84 l16 0" /></>
          : thinking ? <><path d="M100 128 l-34 22" /><path d="M100 128 l20 -26 l-16 -16" /></>
          : unsure ? <><path d="M100 128 l-40 10" /><path d="M100 128 l36 -6 l6 -22" /></>
          : <><path d="M100 128 l-36 26" /><path d="M100 128 l36 26" /></>}
      </g>
      {sad && <text x="150" y="40" fontSize="54" fontWeight="900" fill={INK}>!</text>}
      {unsure && <text x="148" y="40" fontSize="50" fontWeight="900" fill={INK}>?</text>}
      {thinking && <g fill={INK}><circle cx="150" cy="36" r="5" className="rw-blink" /><circle cx="166" cy="24" r="7" className="rw-blink" style={{ animationDelay: '.3s' }} /></g>}
    </svg>
  );
}

const BIG: Record<Phase, string> = { idle: 'Ready?', checking: 'Looking…', stop: 'STOP', ok: 'OK', provisional: 'Looks right', unknown: "Can't tell", error: 'Error' };

export function LookPaper() {
  const c = useRewindCheck();
  const { phase, s1, res } = c;
  const hints = fixHints(res);
  const holes = s1?.hole_pattern_witness?.live_holes_px ?? [];
  const stop = phase === 'stop';
  return (
    <div className="min-h-screen font-paper text-[#111]" style={{ background: '#efe9dc' }} data-look="paper" data-phase={phase}>
      <div className="mx-auto max-w-[1880px] p-6 h-screen flex flex-col">
        {/* page header like an instruction booklet */}
        <header className="flex items-end gap-6 border-b-[6px] border-[#111] pb-3">
          <div className="text-5xl font-black tracking-tight">REWIND</div>
          <div className="text-lg pb-1">watches your hands and says stop</div>
          <div className="ml-auto flex items-center gap-3 pb-1">
            {c.cfg?.synthetic && <span className="border-[3px] border-[#111] px-2 py-0.5 text-sm font-bold bg-[#ffd400]">SYNTHETIC FIXTURE</span>}
            <span className="border-[3px] border-[#111] px-2 py-0.5 text-sm font-bold">{c.fixture ? 'FIXTURE · no model calls' : 'LIVE · Meta models'}</span>
            <a href="?look=v2" className="text-sm underline">classic view</a>
          </div>
        </header>

        <div className="flex-1 grid grid-cols-[84px_1fr_330px] 2xl:grid-cols-[150px_1fr_460px] gap-4 2xl:gap-6 pt-5 min-h-0">
          {/* step numbers like a booklet */}
          <div className="flex flex-col items-center gap-3">
            {Array.from({ length: c.total }, (_, i) => i + 1).map((n) => (
              <div key={n} className={`grid place-items-center rounded-full border-[5px] border-[#111] font-black ${n === c.stepNum ? 'w-20 h-20 text-4xl 2xl:w-28 2xl:h-28 2xl:text-6xl bg-white' : 'w-10 h-10 text-lg 2xl:w-12 2xl:h-12 2xl:text-xl opacity-30'}`}>
                {n === c.stepNum && c.ticked ? '✓' : n}
              </div>
            ))}
            <div className="text-[11px] text-center opacity-60 leading-tight">only step {c.stepNum} has a checkpoint (DECLARED)</div>
          </div>

          {/* the page: manual drawing + camera inset */}
          <div className="relative min-h-0 flex flex-col">
            <div className="flex items-start gap-6">
              <div className="w-52 2xl:w-72 shrink-0">
                <div className="border-[5px] border-[#111] bg-white p-1 rotate-[-1.5deg]"><img src={REFERENCE} alt="manual reference" className="grayscale contrast-150" /></div>
                <div className="text-sm font-bold mt-2">The manual says: {c.step?.title ?? '…'}</div>
              </div>
              <div className="flex-1 pt-2">
                <div className={`font-black leading-[.85] ${stop ? 'text-[#d71920]' : ''}`} style={{ fontSize: stop ? 'clamp(110px,10vw,190px)' : 'clamp(72px,6.3vw,120px)' }} data-testid="furniture-verdict" role="status" aria-live="assertive">{BIG[phase]}</div>
                <div className="text-xl 2xl:text-2xl mt-3 max-w-[640px]">
                  {stop ? "This panel is in, but it's the wrong way round." : phase === 'ok' ? 'Both stages agree. Next step.' : phase === 'provisional' ? 'Local checks pass. Fixture mode: not model-confirmed, so not ticked.' : phase === 'unknown' ? "REWIND won't guess. Reshoot it." : phase === 'checking' ? (c.mode === 'fixture' ? 'Fixture: local witnesses only, no model calls.' : 'SAM 3.1 and Muse Spark Open are looking.') : 'Hold the part up to the camera, then press space.'}
                </div>
              </div>
            </div>
            {/* camera inset as a figure on the page */}
            <div className={`relative mt-5 flex-1 min-h-0 border-[6px] border-[#111] bg-white p-2 ${stop ? 'shadow-[14px_14px_0_#d71920]' : 'shadow-[14px_14px_0_#111]'}`}>
              <FrameSvg src={FRAME[c.scenario]} holes={holes} className="w-full h-full" markerColor={INK} strokeW={6} />
              <div className="absolute left-4 top-4 bg-white border-[3px] border-[#111] px-2 text-sm font-bold">YOU · camera</div>
              {c.ticked && <div className="absolute right-6 top-6 rotate-[6deg] bg-[#1f9d55] text-white border-[4px] border-[#111] px-4 py-2 text-2xl font-black rw-pop" data-testid="manual-ticked">✓ Step {c.stepNum} ticked by Muse Wonder</div>}
              {phase === 'checking' && <div className="rw-scanline" style={{ background: 'linear-gradient(90deg,transparent,#111,transparent)', boxShadow: 'none' }} />}
              {s1 && holes.length === 0 && s1.hole_pattern_witness?.status === 'fail' &&
                <div className="absolute left-1/2 -translate-x-1/2 bottom-6 bg-[#d71920] text-white text-2xl 2xl:text-3xl whitespace-nowrap font-black px-5 py-2 border-[4px] border-[#111]">0 of {s1.hole_pattern_witness.reference_hole_count} holes showing</div>}
            </div>
          </div>

          {/* the helper + fix */}
          <aside className="flex flex-col min-h-0 overflow-y-auto">
            <div className="h-[140px] 2xl:h-[300px] shrink-0"><Buddy phase={phase} /></div>
            {hints.length > 0 && (
              <div className="border-[5px] border-[#111] bg-white p-3 2xl:p-4 mt-2" data-testid="fix-hint">
                <div className="text-sm font-black uppercase tracking-wide mb-2">How to fix it</div>
                {hints.map((h) => (
                  <div key={h.kind} className="flex items-center gap-4 py-1"><FlipHint kind={h.kind} size={96} /><div className="text-base 2xl:text-xl font-bold">{h.text}</div></div>
                ))}
                <div className="text-[11px] opacity-60 mt-1">DETERMINISTIC hint from the failed witness, not model-generated</div>
              </div>
            )}
            {s1 && (
              <div className="mt-3 space-y-1 text-sm 2xl:text-lg">
                <div>{s1.hole_pattern_witness?.status === 'pass' ? '☑' : s1.hole_pattern_witness?.status === 'fail' ? '☒' : '◻'} Holes: {s1.hole_pattern_witness?.status === 'insufficient_evidence' || s1.hole_pattern_witness?.live_hole_count == null ? 'not measurable in this frame' : `expected ${s1.hole_pattern_witness?.reference_hole_count}, saw ${s1.hole_pattern_witness?.live_hole_count}`}</div>
                <div>{s1.tape_edge_witness?.status === 'pass' ? '☑' : s1.tape_edge_witness?.status === 'fail' ? '☒' : '◻'} Taped edge: {s1.tape_edge_witness?.polarity_cosine == null ? 'not measurable in this frame' : `cos ${s1.tape_edge_witness.polarity_cosine}`}</div>
              </div>
            )}
            <div className="mt-auto border-t-[4px] border-[#111] pt-2 text-xs 2xl:text-sm font-mono-tech" data-testid="model-trace">
              {(lanes(res).length ? lanes(res) : [{ model: 'No run yet', state: 'not_measured', value: '' } as any]).map((l: any) => (
                <div key={l.model} className="flex justify-between"><span className="font-bold">{l.model}</span><span className={l.state === 'measured' ? '' : 'opacity-50'}>{l.value}</span></div>
              ))}
              <div className="mt-1 font-paper">Manual: <b>{res?.manual?.label ?? 'Manual not updated'}</b></div>
            </div>
          </aside>
        </div>

        <footer className="flex items-center gap-3 pt-3 text-sm">
          {(c.cfg?.scenarios ?? []).map((s, i) => (
            <button key={s} onClick={() => c.setScenario(s)} className={`border-[3px] border-[#111] px-3 py-1 font-bold ${c.scenario === s ? 'bg-[#111] text-[#efe9dc]' : 'bg-white'}`}>{i + 1} · {SCENARIO_LABEL[s]}</button>
          ))}
          <select value={c.mode} onChange={(e) => c.setMode(e.target.value as any)} className="border-[3px] border-[#111] bg-white px-2 py-1 font-bold"><option value="fixture">Fixture</option><option value="live">Live</option></select>
          <button onClick={c.run} disabled={phase === 'checking'} className="ml-auto bg-[#ffd400] border-[4px] border-[#111] px-8 py-2 text-2xl font-black shadow-[5px_5px_0_#111] active:translate-x-1 active:translate-y-1 active:shadow-none">Check this step ␣</button>
        </footer>
        {c.err && <div className="text-[#d71920]">{c.err}</div>}
      </div>
    </div>
  );
}
