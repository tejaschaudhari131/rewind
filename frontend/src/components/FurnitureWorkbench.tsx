import { useCallback, useEffect, useRef, useState } from 'react';

// REWIND v2 assembly check. Every number shown here comes from the /api/furniture response
// (measured) or is labelled DECLARED / DETERMINISTIC_LOCAL / FIXTURE. Nothing is invented client-side.

type Step = { step_id: string; label: string; title: string; sam_concept_prompt: string };
type Cfg = { synthetic: boolean; total_steps: number; scenarios: string[]; steps: Step[]; product?: string };
type Phase = 'idle' | 'checking' | 'stop' | 'ok' | 'provisional' | 'unknown' | 'error';

const SCENARIO_LABEL: Record<string, string> = {
  ikea_pass_step3: 'Holes facing in',
  ikea_fail_holes_out: 'Flipped: holes out',
  ikea_unmeasurable_blur: 'Dim / blurred',
};
const FRAME: Record<string, string> = {
  ikea_pass_step3: '/fixtures/ikea/checkpoint_pass_step3.png',
  ikea_fail_holes_out: '/fixtures/ikea/checkpoint_fail_holes_out.png',
  ikea_unmeasurable_blur: '/fixtures/ikea/checkpoint_unmeasurable_blur.png',
};
const REFERENCE = '/fixtures/ikea/reference_step3.png';
const FRAME_W = 1280, FRAME_H = 720; // fixture frame size (all fixtures are 1280x720)

const REASON_TEXT: Record<string, string> = {
  panel_face_flipped_holes_not_visible: 'Cam-lock holes are not visible: the panel is showing its outer face.',
  holes_not_visible_panel_face_flipped: 'Cam-lock holes are not visible: the panel is showing its outer face.',
  tape_edge_reversed: 'The taped front edge is on the wrong side: the panel is turned around.',
  tape_edge_matches_reference: 'Taped front edge is on the same side as the manual.',
  hole_face_matches_reference: 'Hole pattern matches the manual: holes face inward.',
  sam_no_instances: 'SAM 3.1 found no panel in the frame, so a pass is withheld.',
  sam_provider_error: 'SAM 3.1 call failed, so the gate fails closed.',
};

function phaseOf(r: any): Phase {
  if (!r) return 'idle';
  if (r.decision === 'accepted') return 'ok';
  if (r.decision === 'rejected') return 'stop';
  if (r.stage1?.provisional_verdict === 'provisional_accepted') return 'provisional';
  return 'unknown';
}

const PHASE_UI: Record<Phase, { word: string; sub: string; ring: string; bg: string; text: string }> = {
  idle: { word: 'READY', sub: 'Point the camera at the part, then check the step', ring: 'border-slate-600', bg: 'bg-slate-900/80', text: 'text-slate-200' },
  checking: { word: 'CHECKING', sub: 'Stage 1: local witnesses + SAM 3.1 → Stage 2: Muse Spark Open', ring: 'border-sky-400', bg: 'bg-sky-950/85', text: 'text-sky-200' },
  stop: { word: 'STOP', sub: "Don't go on. This panel doesn't match the manual.", ring: 'border-rose-500', bg: 'bg-rose-950/90', text: 'text-rose-100' },
  ok: { word: 'OK', sub: 'Confirmed by both stages. Go to the next step.', ring: 'border-emerald-400', bg: 'bg-emerald-950/90', text: 'text-emerald-100' },
  provisional: { word: 'LOOKS RIGHT', sub: 'Local checks pass. Not model-confirmed, so not ticked.', ring: 'border-amber-400', bg: 'bg-amber-950/85', text: 'text-amber-100' },
  unknown: { word: "CAN'T TELL", sub: "Reshoot with more light. REWIND won't guess.", ring: 'border-amber-400', bg: 'bg-amber-950/85', text: 'text-amber-100' },
  error: { word: 'ERROR', sub: 'Request failed. Nothing was ticked.', ring: 'border-rose-500', bg: 'bg-rose-950/90', text: 'text-rose-100' },
};

// --- optional audio + haptic cues (off by default; WebAudio tones, no assets) ---
function cue(kind: 'ok' | 'stop' | 'tick', on: boolean) {
  if (!on) return;
  try {
    const AC = (window as any).AudioContext || (window as any).webkitAudioContext;
    const ctx = new AC();
    const notes = kind === 'stop' ? [[220, 0], [165, 0.16]] : kind === 'ok' ? [[660, 0], [990, 0.1]] : [[1320, 0]];
    notes.forEach(([f, t]) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = kind === 'stop' ? 'square' : 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, ctx.currentTime + t);
      g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + t + 0.22);
      o.connect(g).connect(ctx.destination); o.start(ctx.currentTime + t); o.stop(ctx.currentTime + t + 0.25);
    });
    navigator.vibrate?.(kind === 'stop' ? [120, 60, 120] : 40);
  } catch { /* audio is optional */ }
}

function Tag({ children, tone = 'slate' }: { children: React.ReactNode; tone?: string }) {
  const m: Record<string, string> = {
    slate: 'border-slate-600 text-slate-300', amber: 'border-amber-500/60 text-amber-300 bg-amber-500/10',
    green: 'border-emerald-500/60 text-emerald-300 bg-emerald-500/10', rose: 'border-rose-500/60 text-rose-300 bg-rose-500/10',
    sky: 'border-sky-500/60 text-sky-300 bg-sky-500/10',
  };
  return <span className={`px-2 py-0.5 rounded-full border text-[11px] font-semibold tracking-wide whitespace-nowrap ${m[tone]}`}>{children}</span>;
}

function WitnessRow({ name, w, expected, seen }: { name: string; w: any; expected: string; seen: string }) {
  const st = w?.status;
  const tone = st === 'pass' ? 'text-emerald-300' : st === 'fail' ? 'text-rose-300' : 'text-amber-300';
  const icon = st === 'pass' ? '✓' : st === 'fail' ? '✕' : '?';
  return (
    <div className="flex items-start gap-3 py-2 border-b border-slate-800 last:border-0">
      <div className={`w-7 h-7 shrink-0 rounded-full grid place-items-center font-black ${tone} bg-slate-800`}>{icon}</div>
      <div className="min-w-0">
        <div className="text-[15px] font-semibold text-slate-100">{name}</div>
        <div className="text-[13px] text-slate-400">{REASON_TEXT[w?.reason] ?? w?.reason ?? 'not checked'}</div>
        <div className="text-[12px] font-mono-tech text-slate-500 mt-0.5">expected {expected} · seen {seen} · {w?.producer ?? 'DETERMINISTIC_LOCAL'}</div>
      </div>
    </div>
  );
}

type TraceLane = { model: string; role: string; state: 'measured' | 'not_measured' | 'local' | 'error'; value: string };
function traceLanes(r: any): TraceLane[] {
  if (!r) return [];
  const s1 = r.stage1 ?? {}, s2 = r.stage2 ?? {}, man = r.manual ?? {};
  const lanes: TraceLane[] = [];
  lanes.push({ model: 'Local witnesses', role: 'hole pattern + tape edge (OpenCV)', state: 'local',
    value: s1.stage1_elapsed_ms != null ? `${s1.stage1_elapsed_ms} ms stage 1 wall` : 'n/a' });
  const sam = s1.sam ?? {};
  lanes.push({ model: 'SAM 3.1', role: `segment "${sam.concept_prompt ?? 'wooden panel'}"`,
    state: !sam.called ? 'not_measured' : sam.error ? 'error' : sam.measured ? 'measured' : 'not_measured',
    value: !sam.called ? 'NOT_MEASURED · not called' : sam.error ? 'error · failed closed' : `${sam.latency_ms} ms · ${sam.instances} found` });
  lanes.push({ model: 'Muse Spark Open', role: 'blinded audit on Baseten',
    state: s2.measured ? 'measured' : s2.error ? 'error' : 'not_measured',
    value: s2.measured ? `${s2.wall_ms} ms · ${s2.model_finding}` : s2.error ? `error · ${String(s2.error).slice(0, 40)}` : 'NOT_MEASURED · not run' });
  const t = (man.turns ?? []).map((x: any) => x.latency_ms).filter((x: any) => typeof x === 'number');
  lanes.push({ model: 'Muse Wonder', role: 'computer use: tick the manual',
    state: man.measured ? 'measured' : 'not_measured',
    value: man.measured ? `${man.wall_ms ?? t.reduce((a: number, b: number) => a + b, 0)} ms · ${man.status}` : `NOT_MEASURED · ${man.reason ?? 'not run'}` });
  return lanes;
}

export function FurnitureWorkbench() {
  const [cfg, setCfg] = useState<Cfg | null>(null);
  const [scenario, setScenario] = useState('ikea_fail_holes_out');
  const [mode, setMode] = useState<'fixture' | 'live'>('fixture');
  const [res, setRes] = useState<any>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [err, setErr] = useState<string | null>(null);
  const [sound, setSound] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [history, setHistory] = useState<Array<{ scenario: string; phase: Phase; ms: number | null; at: string }>>([]);
  const runId = useRef(0);

  useEffect(() => {
    fetch('/api/furniture/config').then((r) => r.json()).then(setCfg).catch((e) => setErr(String(e)));
  }, []);

  const step = cfg?.steps[0];
  const total = cfg?.total_steps ?? 8;
  const stepNum = Number((step?.step_id ?? 'step-3').split('-')[1]) || 3;

  const run = useCallback(async () => {
    const id = ++runId.current;
    setPhase('checking'); setErr(null); setRes(null);
    try {
      const r = await fetch('/api/furniture/checkpoint/verify', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario, mode, step_id: step?.step_id, check_off_manual: true }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      if (id !== runId.current) return; // a newer check superseded this one
      const p = phaseOf(j);
      setRes(j); setPhase(p);
      cue(p === 'stop' ? 'stop' : p === 'ok' ? 'ok' : 'tick', sound);
      if (j.manual?.status === 'updated') setTimeout(() => cue('tick', sound), 1100);
      setHistory((h) => [{ scenario, phase: p, ms: j.stage1?.stage1_elapsed_ms ?? null, at: new Date().toLocaleTimeString() }, ...h].slice(0, 6));
    } catch (e) { if (id === runId.current) { setErr(String(e)); setPhase('error'); } }
  }, [scenario, mode, step?.step_id, sound]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); run(); }
      const list = cfg?.scenarios ?? [];
      const k = Number(e.key);
      if (k >= 1 && k <= list.length) { setScenario(list[k - 1]); setRes(null); setPhase('idle'); }
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [run, cfg]);

  const s1 = res?.stage1;
  const manual = res?.manual;
  const ui = PHASE_UI[phase];
  const holes: number[][] = s1?.hole_pattern_witness?.live_holes_px ?? [];
  const ticked = manual?.status === 'updated';
  const lanes = traceLanes(res);
  const fixture = res ? res.mode !== 'live' : mode === 'fixture';

  return (
    <div className="px-5 py-4 text-slate-100 grid gap-4 xl:grid-cols-[200px_minmax(0,1fr)_420px]" data-phase={phase}>
      {/* ---------- step rail ---------- */}
      <aside className="hidden xl:block" aria-label="Assembly steps">
        <div className="text-[11px] uppercase tracking-[.2em] text-slate-500 mb-2">Manual · {total} steps</div>
        <ol className="relative space-y-1.5">
          {Array.from({ length: total }, (_, i) => i + 1).map((n) => {
            const active = n === stepNum;
            const done = active && ticked;
            return (
              <li key={n} className={`flex items-center gap-3 rounded-xl px-2 py-2 ${active ? 'bg-slate-800/80 ring-1 ring-sky-500/50' : ''}`}>
                <span className={`w-8 h-8 rounded-full grid place-items-center text-sm font-bold border-2 ${
                  done ? 'bg-emerald-500 border-emerald-400 text-slate-950' : active ? (phase === 'stop' ? 'border-rose-500 text-rose-300' : 'border-sky-400 text-sky-200') : 'border-slate-700 text-slate-500'}`}>
                  {done ? '✓' : n}
                </span>
                <span className="min-w-0">
                  <span className={`block text-sm ${active ? 'text-white font-semibold' : 'text-slate-500'}`}>{active ? (step?.title ?? `Step ${n}`) : `Step ${n}`}</span>
                  {!active && <span className="block text-[10px] text-slate-600">no checkpoint in this build</span>}
                  {active && <span className="block text-[11px] text-slate-400">checkpoint armed</span>}
                </span>
              </li>
            );
          })}
        </ol>
        <p className="mt-3 text-[10px] leading-snug text-slate-600">DECLARED: only step {stepNum} has a reference image + witnesses. Other steps are listed from the manual and are not checked.</p>
      </aside>

      {/* ---------- hero camera ---------- */}
      <section className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-2xl font-black tracking-tight" data-testid="step-label">{step?.label ?? 'Loading…'}</h2>
          <span className="text-slate-400 text-lg">{step?.title}</span>
          <div className="ml-auto flex gap-2">
            {cfg?.synthetic && <Tag tone="amber">SYNTHETIC FIXTURE · not real footage</Tag>}
            <Tag tone={fixture ? 'amber' : 'sky'}>{fixture ? 'FIXTURE MODE · no model calls' : 'LIVE · Meta models'}</Tag>
          </div>
        </div>

        <div className={`relative rounded-2xl overflow-hidden border-4 transition-colors duration-300 ${ui.ring} ${phase === 'stop' ? 'rw-shake' : ''}`}>
          <img src={FRAME[scenario]} alt="camera frame" className="w-full block" />
          {phase === 'checking' && <div className="rw-scanline" />}
          {(phase === 'stop' || phase === 'ok') && <div className={`absolute inset-0 rw-flash ${phase === 'stop' ? 'bg-rose-500' : 'bg-emerald-400'}`} />}
          {overlay && s1 && (
            <svg viewBox={`0 0 ${FRAME_W} ${FRAME_H}`} className="absolute inset-0 w-full h-full pointer-events-none" data-testid="hole-overlay">
              {holes.map(([x, y, a], i) => {
                const r = Math.max(6, Math.sqrt(a / Math.PI) + 5);
                return <circle key={i} cx={x} cy={y} r={r} fill="none" stroke="#34d399" strokeWidth={3} className="rw-fadeup" style={{ animationDelay: `${i * 30}ms` }} />;
              })}
              {holes.length === 0 && s1.hole_pattern_witness?.status === 'fail' && (
                <g>
                  <rect x={40} y={FRAME_H - 92} width={560} height={52} rx={12} fill="rgba(15,23,42,.85)" />
                  <text x={60} y={FRAME_H - 58} fill="#fda4af" fontSize={26} fontWeight={700}>0 of {s1.hole_pattern_witness?.reference_hole_count} holes visible</text>
                </g>
              )}
            </svg>
          )}
          {/* verdict badge */}
          <div className={`absolute left-5 top-5 rounded-2xl px-6 py-4 border-2 backdrop-blur ${ui.bg} ${ui.ring} ${phase === 'checking' ? 'rw-pulse' : phase !== 'idle' ? 'rw-pop' : ''}`}
               data-testid="furniture-verdict" role="status" aria-live="assertive">
            <div className={`text-6xl font-black leading-none tracking-tight ${ui.text}`}>{ui.word}</div>
            <div className={`mt-1 text-lg ${ui.text} opacity-90 max-w-[520px]`}>{phase === 'checking' && mode === 'fixture' ? 'Fixture: local witnesses only, no model calls' : ui.sub}</div>
            {phase === 'provisional' && fixture && <div className="mt-1 text-sm text-amber-300">PROVISIONAL PASS · FIXTURE ONLY (stage 2 not run)</div>}
          </div>
          <div className="absolute right-4 bottom-4 flex gap-2 text-[11px]">
            {overlay && s1 && <Tag tone="green">○ detected holes ({holes.length}) · local CV</Tag>}
            <Tag>SAM mask outline: drawn only when SAM returns a mask</Tag>
          </div>
        </div>

        {/* controls */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] uppercase tracking-widest text-slate-500 mr-1">Demo frame</span>
          {(cfg?.scenarios ?? []).map((s, i) => (
            <button key={s} onClick={() => { setScenario(s); setRes(null); setPhase('idle'); }}
              className={`px-3 py-2 rounded-xl text-sm border transition ${scenario === s ? 'bg-sky-500/20 border-sky-400 text-white' : 'border-slate-700 text-slate-300 hover:border-slate-500'}`}>
              <kbd className="text-[10px] text-slate-500 mr-1">{i + 1}</kbd>{SCENARIO_LABEL[s] ?? s}
            </button>
          ))}
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} className="bg-slate-900 border border-slate-700 rounded-xl text-sm px-3 py-2">
            <option value="fixture">Fixture (no model calls)</option>
            <option value="live">Live (SAM + Baseten + Wonder)</option>
          </select>
          <label className="text-sm text-slate-400 flex items-center gap-1"><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> overlay</label>
          <label className="text-sm text-slate-400 flex items-center gap-1"><input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} /> sound</label>
          <button onClick={run} disabled={phase === 'checking'}
            className="ml-auto px-6 py-3 rounded-2xl bg-sky-500 hover:bg-sky-400 text-slate-950 text-base font-black shadow-[0_4px_0_#0369a1] active:translate-y-1 active:shadow-none transition disabled:opacity-60">
            {phase === 'checking' ? 'Checking…' : 'Check this step'} <kbd className="ml-1 text-[11px] opacity-70">space</kbd>
          </button>
        </div>

        {/* rewind filmstrip: this session's real checks */}
        {history.length > 0 && (
          <div className="flex gap-2 items-center overflow-x-auto" aria-label="Check history">
            <span className="text-[11px] uppercase tracking-widest text-slate-500 mr-1">Rewind</span>
            {history.map((h, i) => (
              <div key={i} className={`flex items-center gap-2 rounded-lg border px-2 py-1 text-xs ${h.phase === 'stop' ? 'border-rose-600/60' : h.phase === 'ok' ? 'border-emerald-600/60' : 'border-amber-600/50'}`}>
                <img src={FRAME[h.scenario]} className="w-14 h-8 object-cover rounded" alt="" />
                <span className="font-bold">{PHASE_UI[h.phase].word}</span>
                <span className="text-slate-500 font-mono-tech">{h.at}{h.ms != null ? ` · ${h.ms} ms` : ''}</span>
              </div>
            ))}
          </div>
        )}
        {err && <div className="text-rose-300 text-sm">{err}</div>}
      </section>

      {/* ---------- why + trace + manual ---------- */}
      <aside className="space-y-3 min-w-0">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-[11px] uppercase tracking-[.2em] text-slate-500 mb-2">Why</div>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <figure><img src={REFERENCE} alt="manual reference" className="rounded-lg border border-slate-700" />
              <figcaption className="text-[12px] text-slate-400 mt-1">Manual expects</figcaption></figure>
            <figure><img src={FRAME[scenario]} alt="what the camera sees" className={`rounded-lg border ${phase === 'stop' ? 'border-rose-500' : 'border-slate-700'}`} />
              <figcaption className="text-[12px] text-slate-400 mt-1">Camera sees</figcaption></figure>
          </div>
          {s1 ? (
            <div className="rw-fadeup">
              <WitnessRow name="Cam-lock hole pattern (which face)" w={s1.hole_pattern_witness}
                expected={`${s1.hole_pattern_witness?.reference_hole_count ?? '?'} holes`} seen={`${s1.hole_pattern_witness?.live_hole_count ?? '?'} holes`} />
              <WitnessRow name="Taped front edge (which way round)" w={s1.tape_edge_witness}
                expected="cos ≥ 0.5" seen={`cos ${s1.tape_edge_witness?.polarity_cosine ?? 'n/a'}`} />
              <div className="text-[12px] text-slate-500 pt-2">Outline match IoU {s1.panel_silhouette_iou?.toFixed?.(3) ?? 'n/a'}: the outline alone can't tell the two faces apart, which is why the witnesses decide.</div>
              {phase === 'stop' && stepNum < total && (
                <div className="mt-3 rounded-xl bg-rose-500/10 border border-rose-500/40 p-3 text-[14px] text-rose-100">
                  Caught at step {stepNum}. If missed, steps {stepNum + 1}–{total} would be built on top of this panel.
                  <span className="block text-[11px] text-rose-300/70 mt-0.5">DECLARED from the step count, not measured.</span>
                </div>
              )}
            </div>
          ) : <div className="text-sm text-slate-500">Run a check to see which witness decided.</div>}
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4" data-testid="model-trace">
          <div className="flex items-center mb-2"><span className="text-[11px] uppercase tracking-[.2em] text-slate-500">Model trace</span>
            <span className="ml-auto text-[10px] text-slate-500">only measured values show a time</span></div>
          {lanes.length === 0 ? <div className="text-sm text-slate-500">No run yet.</div> : (
            <ul className="space-y-1.5">
              {lanes.map((l) => (
                <li key={l.model} className="flex items-center gap-2 text-[13px]">
                  <span className={`w-2.5 h-2.5 rounded-full ${l.state === 'measured' ? 'bg-emerald-400' : l.state === 'local' ? 'bg-sky-400' : l.state === 'error' ? 'bg-rose-500' : 'bg-slate-600'}`} />
                  <span className="font-semibold w-36 shrink-0">{l.model}</span>
                  <span className="font-mono-tech text-slate-400 truncate" title={l.role}>{l.value}</span>
                </li>
              ))}
            </ul>
          )}
          <div className="mt-2 text-[10px] text-slate-600">green = MEASURED this run · blue = DETERMINISTIC_LOCAL · grey = NOT_MEASURED</div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 relative" data-testid="manual-pane">
          <div className="flex items-center mb-2"><span className="text-[11px] uppercase tracking-[.2em] text-slate-500">Manual checklist</span>
            <span className="ml-auto text-[10px] text-slate-500">ticked only by Muse Wonder after a confirmed OK</span></div>
          <ul className="space-y-1">
            {[stepNum - 1, stepNum, stepNum + 1].filter((n) => n >= 1 && n <= total).map((n) => {
              const me = n === stepNum;
              return (
                <li key={n} className={`relative flex items-center gap-3 rounded-lg px-2 py-1.5 ${me ? 'bg-slate-800' : 'opacity-50'}`}>
                  <span className={`w-6 h-6 rounded-md border-2 grid place-items-center ${me && ticked ? 'bg-emerald-500 border-emerald-400' : 'border-slate-500'}`}>
                    {me && ticked && <svg viewBox="0 0 24 24" className="w-4 h-4 rw-check"><path d="M5 12l5 5 9-10" fill="none" stroke="#052e16" strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round" /></svg>}
                  </span>
                  <span className="text-sm">Step {n}{me && step ? ` · ${step.title}` : ''}</span>
                  {me && ticked && <span className="absolute left-3 top-2 rw-cursor text-xl" aria-hidden>➤</span>}
                </li>
              );
            })}
          </ul>
          <div className={`mt-2 text-sm ${ticked ? 'text-emerald-300 font-semibold' : 'text-slate-300'}`}>{manual?.label ?? 'Manual not updated'}</div>
          {manual?.reason && <div className="text-[11px] text-slate-500">reason: {manual.reason}</div>}
          {ticked && manual?.clicks_executed?.[0] && <div className="text-[11px] text-slate-500 font-mono-tech">Wonder click at ({manual.clicks_executed[0].x}, {manual.clicks_executed[0].y}) · checkbox state verified in DOM</div>}
        </div>
      </aside>
    </div>
  );
}
