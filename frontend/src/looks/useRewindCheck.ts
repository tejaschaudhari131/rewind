import { useCallback, useEffect, useRef, useState } from 'react';
// Shared data hook for the v3 looks. Every value comes from /api/furniture/*; nothing is invented here.
export type Phase = 'idle' | 'checking' | 'stop' | 'ok' | 'provisional' | 'unknown' | 'error';
export type Cfg = { synthetic: boolean; total_steps: number; scenarios: string[]; steps: Array<{ step_id: string; label: string; title: string; sam_concept_prompt: string }> };
export type HistoryItem = { scenario: string; phase: Phase; res: any; at: string };

export const FRAME: Record<string, string> = {
  ikea_pass_step3: '/fixtures/ikea/checkpoint_pass_step3.png',
  ikea_fail_holes_out: '/fixtures/ikea/checkpoint_fail_holes_out.png',
  ikea_unmeasurable_blur: '/fixtures/ikea/checkpoint_unmeasurable_blur.png',
};
export const REFERENCE = '/fixtures/ikea/reference_step3.png';
export const FRAME_W = 1280, FRAME_H = 720;
export const SCENARIO_LABEL: Record<string, string> = {
  ikea_pass_step3: 'Holes facing in', ikea_fail_holes_out: 'Flipped: holes out', ikea_unmeasurable_blur: 'Dim / blurred',
};

export function phaseOf(r: any): Phase {
  if (!r) return 'idle';
  if (r.decision === 'accepted') return 'ok';
  if (r.decision === 'rejected') return 'stop';
  if (r.stage1?.provisional_verdict === 'provisional_accepted') return 'provisional';
  return 'unknown';
}

/** Fix hints are DETERMINISTIC: mapped 1:1 from the gate's reason codes, never generated. */
export function fixHints(r: any): Array<{ kind: 'flip' | 'turn' | 'light'; text: string }> {
  const codes: string[] = r?.stage1?.provisional_reason_codes ?? [];
  const out: Array<{ kind: 'flip' | 'turn' | 'light'; text: string }> = [];
  if (codes.some((c) => c.includes('holes_not_visible') || c.includes('face_flipped'))) out.push({ kind: 'flip', text: 'Flip the panel over so the cam-lock holes face in' });
  if (codes.includes('tape_edge_reversed')) out.push({ kind: 'turn', text: 'Turn it end-for-end: taped edge goes to the front' });
  const h = r?.stage1?.hole_pattern_witness?.status, t = r?.stage1?.tape_edge_witness?.status;
  if (!out.length && phaseOf(r) === 'unknown' && (h === 'insufficient_evidence' || t === 'insufficient_evidence')) out.push({ kind: 'light', text: 'More light, hold steady, reshoot' });
  return out;
}

export type Lane = { model: string; state: 'measured' | 'local' | 'not_measured' | 'error'; value: string };
export function lanes(r: any): Lane[] {
  if (!r) return [];
  const s1 = r.stage1 ?? {}, s2 = r.stage2 ?? {}, m = r.manual ?? {}, sam = s1.sam ?? {};
  return [
    { model: 'Local witnesses', state: 'local', value: s1.stage1_elapsed_ms != null ? `${s1.stage1_elapsed_ms} ms` : 'n/a' },
    { model: 'SAM 3.1', state: !sam.called ? 'not_measured' : sam.error ? 'error' : 'measured', value: !sam.called ? 'NOT_MEASURED' : sam.error ? 'error · failed closed' : `${sam.latency_ms} ms · ${sam.instances} found` },
    { model: 'Muse Spark Open', state: s2.measured ? 'measured' : s2.error ? 'error' : 'not_measured', value: s2.measured ? `${s2.wall_ms} ms · ${s2.model_finding}` : s2.error ? 'error' : 'NOT_MEASURED' },
    { model: 'Muse Wonder', state: m.measured ? 'measured' : 'not_measured', value: m.measured ? `${m.wall_ms ?? 'n/a'} ms · ${m.status}` : 'NOT_MEASURED' },
  ];
}

export function useRewindCheck() {
  const q = new URLSearchParams(window.location.search);
  const [cfg, setCfg] = useState<Cfg | null>(null);
  const [scenario, setScenarioRaw] = useState(q.get('scenario') ?? 'ikea_fail_holes_out');
  const [mode, setMode] = useState<'fixture' | 'live'>(q.get('mode') === 'live' ? 'live' : 'fixture');
  const [res, setRes] = useState<any>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [err, setErr] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const runId = useRef(0);
  useEffect(() => { fetch('/api/furniture/config').then((r) => r.json()).then(setCfg).catch((e) => setErr(String(e))); }, []);
  const step = cfg?.steps[0];
  const total = cfg?.total_steps ?? 8;
  const stepNum = Number((step?.step_id ?? 'step-3').split('-')[1]) || 3;
  const setScenario = (s: string) => { runId.current++; setScenarioRaw(s); setRes(null); setPhase('idle'); };
  const run = useCallback(async () => {
    const id = ++runId.current;
    setPhase('checking'); setErr(null); setRes(null);
    try {
      const r = await fetch('/api/furniture/checkpoint/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario, mode, step_id: step?.step_id, check_off_manual: true }) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      if (id !== runId.current) return;
      const p = phaseOf(j);
      setRes(j); setPhase(p);
      setHistory((h) => [...h, { scenario, phase: p, res: j, at: new Date().toLocaleTimeString() }].slice(-12));
    } catch (e) { if (id === runId.current) { setErr(String(e)); setPhase('error'); } }
  }, [scenario, mode, step?.step_id]);
  /** Rewind: restore an earlier real result from this session (no new call). */
  const rewindTo = (h: HistoryItem) => { runId.current++; setScenarioRaw(h.scenario); setRes(h.res); setPhase(h.phase); };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); run(); }
      const k = Number(e.key), list = cfg?.scenarios ?? [];
      if (k >= 1 && k <= list.length) setScenario(list[k - 1]);
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [run, cfg]);
  const fixture = res ? res.mode !== 'live' : mode === 'fixture';
  return { cfg, step, total, stepNum, scenario, setScenario, mode, setMode, res, phase, err, run, history, rewindTo, fixture,
    ticked: res?.manual?.status === 'updated', s1: res?.stage1 };
}
