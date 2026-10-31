import { useEffect, useState } from 'react';

type Step = { step_id: string; label: string; title: string; sam_concept_prompt: string };
type Cfg = { synthetic: boolean; total_steps: number; scenarios: string[]; steps: Step[] };

const SCENARIO_LABEL: Record<string, string> = {
  ikea_pass_step3: 'Panel placed (holes facing in)',
  ikea_fail_holes_out: 'Panel flipped (holes facing out)',
  ikea_unmeasurable_blur: 'Dim / blurred frame',
};
const FRAME: Record<string, string> = {
  ikea_pass_step3: '/fixtures/ikea/checkpoint_pass_step3.png',
  ikea_fail_holes_out: '/fixtures/ikea/checkpoint_fail_holes_out.png',
  ikea_unmeasurable_blur: '/fixtures/ikea/checkpoint_unmeasurable_blur.png',
};

function verdictText(r: any): { text: string; tone: string } {
  if (!r) return { text: 'Not checked', tone: 'text-slate-400' };
  if (r.decision === 'accepted') return { text: 'PASS: confirmed by both stages', tone: 'text-emerald-300' };
  if (r.decision === 'rejected') return { text: 'STOP: panel face/orientation does not match the manual', tone: 'text-rose-300' };
  if (r.stage1?.provisional_verdict === 'provisional_accepted')
    return { text: r.fixture_only ? 'PROVISIONAL PASS (fixture only, not model-confirmed)' : 'Needs review', tone: 'text-amber-300' };
  return { text: "UNKNOWN: can't tell from this frame. Reshoot, don't proceed", tone: 'text-amber-300' };
}

export function FurnitureWorkbench() {
  const [cfg, setCfg] = useState<Cfg | null>(null);
  const [scenario, setScenario] = useState('ikea_fail_holes_out');
  const [mode, setMode] = useState<'fixture' | 'live'>('fixture');
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/furniture/config').then((r) => r.json()).then(setCfg).catch((e) => setErr(String(e)));
  }, []);

  const step = cfg?.steps[0];
  const run = async () => {
    setBusy(true); setErr(null); setRes(null);
    try {
      const r = await fetch('/api/furniture/checkpoint/verify', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario, mode, step_id: step?.step_id, check_off_manual: true }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setRes(await r.json());
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const v = verdictText(res);
  const s1 = res?.stage1;
  const manual = res?.manual;

  return (
    <div className="p-4 grid gap-4 lg:grid-cols-3 text-slate-100">
      <section className="lg:col-span-2 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="font-mono-tech text-lg font-bold" data-testid="step-label">{step?.label ?? 'Loading…'}</h2>
          {cfg?.synthetic && (
            <span className="px-2 py-0.5 text-[10px] rounded border border-amber-500/40 text-amber-300">SYNTHETIC FIXTURE: not real footage</span>
          )}
        </div>
        <p className="text-sm text-slate-400">{step?.title}. Checked against the manual's reference image for this step.</p>
        <div className="flex flex-wrap gap-2">
          {(cfg?.scenarios ?? []).map((s) => (
            <button key={s} onClick={() => setScenario(s)}
              className={`px-3 py-1.5 rounded text-xs border ${scenario === s ? 'bg-sky-500/20 border-sky-400' : 'border-slate-700'}`}>
              {SCENARIO_LABEL[s] ?? s}
            </button>
          ))}
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} className="bg-slate-900 border border-slate-700 rounded text-xs px-2">
            <option value="fixture">Fixture (no model calls)</option>
            <option value="live">Live (SAM + Baseten + Wonder)</option>
          </select>
          <button onClick={run} disabled={busy} className="px-4 py-1.5 rounded bg-sky-600 text-xs font-bold">{busy ? 'Checking…' : 'Check this step'}</button>
        </div>
        <img src={FRAME[scenario]} alt="camera frame" className="w-full rounded border border-slate-800" />
      </section>
      <aside className="space-y-3">
        <div className="rounded border border-slate-800 p-3">
          <div className={`font-bold ${v.tone}`} data-testid="furniture-verdict">{v.text}</div>
          {s1 && (
            <ul className="mt-2 text-xs text-slate-300 space-y-1 font-mono-tech">
              <li>Hole pattern (face) witness: {s1.hole_pattern_witness?.status}</li>
              <li>Tape-edge (orientation) witness: {s1.tape_edge_witness?.status}</li>
              <li>Panel outline IoU: {s1.panel_silhouette_iou?.toFixed?.(3) ?? 'n/a'} (outline alone can't tell the faces apart)</li>
              <li>SAM concept "{step?.sam_concept_prompt}": {s1.sam?.called ? (s1.sam.error ? 'error' : `${s1.sam.instances} found`) : 'not called (fixture)'}</li>
              <li>Stage 1: {s1.stage1_elapsed_ms} ms · Stage 2: {res.stage2?.measured ? `${res.stage2.model_finding} (${res.stage2.wall_ms} ms)` : 'not run'}</li>
              {s1.provisional_reason_codes?.length > 0 && <li>Reasons: {s1.provisional_reason_codes.join(', ')}</li>}
            </ul>
          )}
          {err && <div className="text-rose-300 text-xs mt-2">{err}</div>}
        </div>
        <div className="rounded border border-slate-800 p-3" data-testid="manual-pane">
          <div className="text-xs uppercase text-slate-400 mb-1">Manual checklist</div>
          <div className={manual?.status === 'updated' ? 'text-emerald-300' : 'text-slate-300'}>
            {manual?.label ?? 'Manual not updated'}
          </div>
          {manual?.reason && <div className="text-[11px] text-slate-500">reason: {manual.reason}</div>}
        </div>
      </aside>
    </div>
  );
}
