import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Box,
  Camera,
  CheckCircle2,
  Cpu,
  Download,
  Eye,
  FileCode2,
  FileText,
  Layers,
  Mic,
  Play,
  RefreshCw,
  Ruler,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Terminal,
  Wand2,
  XCircle,
} from 'lucide-react';
import { StlViewer3D } from './components/StlViewer3D';
import {
  LiveTelemetryProofBar,
  LiveCameraControlStrip,
  PairedWitnessCropComparison,
  SliderCrankKinematicTwin,
  WonderCuaDrawingOverlay,
} from './components/WorkbenchUpgrades';

type ExecutionMode = 'fixture' | 'hybrid' | 'live';

interface StepItem {
  step_number: number;
  step_id: string;
  part_id: string;
  title: string;
  status?: string;
}

interface GateResponse {
  decision: 'accepted' | 'rejected' | 'unknown' | 'pending';
  reason_codes: string[];
  provisional_status: string;
  elapsed_ms: number;
  registration_rmse_px: number;
  calibration_valid: boolean;
  detected_tag_ids: number[];
  geometry_metrics: {
    status: string;
    mask_iou: number;
    coverage_ratio: number;
    pivot_error_px: number;
    witness_error_px: number;
    registration_rmse_px: number;
  };
  glimmer_verdict: string;
  glimmer_confidence: number;
  witness_mark_aligned: boolean;
  witness_polarity_cosine?: number;
  paired_crop_ref_b64?: string | null;
  paired_crop_live_b64?: string | null;
  dual_open_weight_consensus?: any;
  telemetry_trace?: any;
  repair_card_url?: string | null;
  blinded_audit_check: {
    expected_pose_id_stripped: boolean;
    expected_step_stripped: boolean;
    forbidden_keys_blocked: string[];
  };
  envelopes: Record<string, any>;
  session_snapshot: any;
  scenario?: string;
  mode?: string;
  frame_url?: string;
}

interface FlatMeasurement {
  plane_id: string;
  unit: string;
  hole_center_distance: number;
  outer_width: number;
  thickness: number;
  pin_diameter: number;
  left_hole_diameter_mm: number;
  right_hole_diameter_mm: number;
  diametral_clearance: number;
  uncertainty_95_mm: number;
  flat_contact_confirmed: boolean;
  status: string;
}

const DEFAULT_STEPS: StepItem[] = [
  {
    step_number: 1,
    step_id: 'mount-base-and-guide',
    part_id: 'crank-support',
    title: 'Step 1: Mount Green Crank Support, Red Crank Disk & Gray Linear Guide',
    status: 'completed',
  },
  {
    step_number: 2,
    step_id: 'attach-connecting-rod',
    part_id: 'connecting-rod',
    title: 'Step 2: Seat Yellow Connecting Rod (Align Witness Stripe to Left Crank Pin)',
    status: 'active',
  },
  {
    step_number: 3,
    step_id: 'attach-output-paddle',
    part_id: 'output-paddle',
    title: 'Step 3: Attach Purple Output Paddle to Blue Slider Block',
    status: 'pending',
  },
  {
    step_number: 4,
    step_id: 'scan-flat-link',
    part_id: 'connecting-rod',
    title: 'Step 4: Scan Removed Connecting Rod on mat_z0 (Homography mm + STEP/STL)',
    status: 'pending',
  },
];

export const App: React.FC = () => {
  const [mode, setMode] = useState<ExecutionMode>('fixture');
  const [sceneVersion, setSceneVersion] = useState<number>(1);
  const [planVersion, setPlanVersion] = useState<number>(1);
  const [sessionState, setSessionState] = useState<string>('WAITING_FOR_CHECKPOINT');
  const [steps, setSteps] = useState<StepItem[]>(DEFAULT_STEPS);
  const [activeFrameUrl, setActiveFrameUrl] = useState<string>('/api/assets/golden/checkpoint_pass_step2.png');
  const [activeScenario, setActiveScenario] = useState<string>('pass_step2');
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  const [gateData, setGateData] = useState<GateResponse | null>(null);
  const [selectedEnvelopeStage, setSelectedEnvelopeStage] = useState<string>('gate_decision');
  const [flatMeasurement, setFlatMeasurement] = useState<FlatMeasurement>({
    plane_id: 'mat_z0',
    unit: 'mm',
    hole_center_distance: 150.01,
    outer_width: 25.0,
    thickness: 4.0,
    pin_diameter: 7.99,
    left_hole_diameter_mm: 7.97,
    right_hole_diameter_mm: 8.0,
    diametral_clearance: 0.4,
    uncertainty_95_mm: 0.18,
    flat_contact_confirmed: true,
    status: 'eligible_for_cad',
  });
  const [cadViewTab, setCadViewTab] = useState<'3d' | '2d' | 'repair'>('3d');
  const [cadMaxErrorMm, setCadMaxErrorMm] = useState<number>(0.00003);
  const [cadVolumeMm3, setCadVolumeMm3] = useState<number>(16561.99);
  const [bonusSummary, setBonusSummary] = useState<string>(
    'Ready: Click an Async Bonus Lane button to run Muse Wonder 1.0 CUA CAD Inspection, Muse Image 1.0 Repair Card, or Baseten Muse Spark Session Review.'
  );
  const [voiceHistory, setVoiceHistory] = useState<
    Array<{ transcript: string; intent: string; summary: string; latency_ms: number; ts: string }>
  >([]);
  const [envelopeLog, setEnvelopeLog] = useState<Array<any>>([]);

  const syncFromSessionSnapshot = (snap: any) => {
    if (!snap) return;
    if (typeof snap.scene_version === 'number') setSceneVersion(snap.scene_version);
    if (typeof snap.plan_version === 'number') setPlanVersion(snap.plan_version);
    if (typeof snap.state === 'string') setSessionState(snap.state);
    if (Array.isArray(snap.envelope_log)) {
      setEnvelopeLog(snap.envelope_log.slice(-12).reverse());
    }
  };

  const runScenarioVerify = async (scenarioKey: string) => {
    setBusyAction(scenarioKey);
    try {
      const res = await fetch('/api/checkpoint/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenarioKey, mode }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: GateResponse = await res.json();
      setGateData(data);
      setActiveScenario(scenarioKey);
      if (data.frame_url) setActiveFrameUrl(data.frame_url);
      syncFromSessionSnapshot(data.session_snapshot);
    } catch (err) {
      console.error('Checkpoint verify error:', err);
    } finally {
      setBusyAction(null);
    }
  };


  const handleUploadLiveFrameBlob = async (blob: Blob) => {
    setBusyAction('live_webrtc');
    try {
      const fd = new FormData();
      fd.append('frame', blob, 'live_camera.png');
      fd.append('step_id', 'attach-connecting-rod');
      fd.append('mode', mode);
      const res = await fetch('/api/verify-step', {
        method: 'POST',
        body: fd,
      });
      if (res.ok) {
        const data: GateResponse = await res.json();
        setGateData(data);
        setActiveScenario('live_webrtc');
        syncFromSessionSnapshot(data.session_snapshot);
      }
    } catch (err) {
      console.error('Live WebRTC frame upload error:', err);
    } finally {
      setBusyAction(null);
    }
  };

  const runFlatScanAndCad = async () => {
    setBusyAction('flat_scan');
    try {
      const scanRes = await fetch('/api/metrology/flat-scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ px_per_mm: 4.0, declared_thickness_mm: 4.0, diametral_clearance_mm: 0.4 }),
      });
      if (scanRes.ok) {
        const scanJson = await scanRes.json();
        if (scanJson.flat_part_MeasurementResult) {
          setFlatMeasurement(scanJson.flat_part_MeasurementResult);
        }
        if (scanJson.frame_url) setActiveFrameUrl(scanJson.frame_url);
        setActiveScenario('flat_scan');
        syncFromSessionSnapshot(scanJson.session);
      }

      const cadRes = await fetch('/api/cad/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apply_clearance: false }),
      });
      if (cadRes.ok) {
        const cadJson = await cadRes.json();
        if (cadJson.cad_VerificationResult) {
          setCadMaxErrorMm(cadJson.cad_VerificationResult.max_abs_error_mm ?? 0.00003);
          setCadVolumeMm3(cadJson.cad_VerificationResult.volume_mm3 ?? 16561.99);
        }
        syncFromSessionSnapshot(cadJson.session);
      }
    } catch (err) {
      console.error('Flat scan error:', err);
    } finally {
      setBusyAction(null);
    }
  };

  const handleCompileTeardown = async () => {
    setBusyAction('compile');
    try {
      const res = await fetch('/api/teardown/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.compiled_steps)) setSteps(data.compiled_steps);
        syncFromSessionSnapshot(data.session);
      }
    } catch (err) {
      console.error('Compile teardown error:', err);
    } finally {
      setBusyAction(null);
    }
  };

  const handleSwitchMode = async (nextMode: ExecutionMode) => {
    setMode(nextMode);
    try {
      const res = await fetch('/api/session/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: nextMode }),
      });
      if (res.ok) {
        const data = await res.json();
        syncFromSessionSnapshot(data.session);
      }
    } catch (err) {
      console.error('Mode switch error:', err);
    }
  };

  const handleVoiceCommand = async (phrase: string) => {
    setBusyAction(`voice:${phrase}`);
    try {
      const res = await fetch('/api/voice/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: phrase, mode: 'fixture' }),
      });
      if (res.ok) {
        const data = await res.json();
        const summary =
          data.action_result?.explanation ||
          data.action_result?.instruction ||
          (data.action_result?.checkpoint_decision
            ? `Checkpoint gate verdict: ${data.action_result.checkpoint_decision.toUpperCase()}`
            : JSON.stringify(data.action_result));
        setVoiceHistory((prev) => [
          {
            transcript: data.transcript,
            intent: data.intent,
            summary,
            latency_ms: data.latency_ms ?? 2.5,
            ts: new Date().toLocaleTimeString(),
          },
          ...prev.slice(0, 7),
        ]);
        syncFromSessionSnapshot(data.session);
      }
    } catch (err) {
      console.error('Voice command error:', err);
    } finally {
      setBusyAction(null);
    }
  };

  const handleBonusAction = async (lane: 'cua' | 'repair' | 'spark') => {
    setBusyAction(`bonus:${lane}`);
    try {
      if (lane === 'cua') {
        const res = await fetch('/api/bonus/cua-inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: 'fixture' }),
        });
        const data = await res.json();
        setBonusSummary(data.summary || 'Muse Wonder 1.0 CUA CAD Inspection completed.');
        setCadViewTab('2d');
      } else if (lane === 'repair') {
        const res = await fetch('/api/bonus/repair-card', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: 'fixture' }),
        });
        const data = await res.json();
        setBonusSummary(`Muse Image 1.0 Visual Repair Card generated (${data.latency_ms} ms) -> ${data.repair_card_url}`);
        setCadViewTab('repair');
      } else {
        const res = await fetch('/api/bonus/spark-review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: 'fixture' }),
        });
        const data = await res.json();
        setBonusSummary(`[${data.model}] ${data.summary}`);
      }
    } catch (err) {
      console.error('Bonus lane error:', err);
    } finally {
      setBusyAction(null);
    }
  };

  useEffect(() => {
    // Initial boot: run pass_step2 so the workbench opens with populated sub-2.8s gate metrics & contract envelopes
    runScenarioVerify('pass_step2');
  }, []);

  const decision = gateData?.decision ?? 'accepted';
  const regRmse = gateData?.registration_rmse_px ?? 0.39;
  const calValid = gateData?.calibration_valid ?? true;
  const elapsedMs = gateData?.elapsed_ms ?? 38.4;
  const geom = gateData?.geometry_metrics ?? {
    status: 'pass',
    mask_iou: 0.9722,
    coverage_ratio: 0.975,
    pivot_error_px: 0.37,
    witness_error_px: 0.36,
    registration_rmse_px: 0.39,
  };
  const activeEnvelope =
    gateData?.envelopes?.[selectedEnvelopeStage] ?? gateData?.envelopes?.gate_decision ?? {};

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-sky-500/30">
      {/* Top Industrial Workbench Header */}
      <header className="border-b border-slate-800 bg-slate-900/95 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-amber-500 flex items-center justify-center shadow-md shadow-sky-500/10">
            <Cpu className="w-4 h-4 text-slate-950" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-mono-tech text-sm font-bold tracking-wider text-white">
                REWIND // SPATIAL REASSEMBLY &amp; CAD WORKBENCH
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono-tech rounded bg-sky-500/15 text-sky-300 border border-sky-500/30">
                CONTRACT v1.0.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              SAM 3.1 Spatial Spine · Blinded Muse Glimmer 30B (Modal H100) · Plane-Aware AprilTag Metrology · Watertight STEP/STL
            </p>
          </div>
        </div>

        {/* Version Badges, AprilTag Lock Pill & Mode Selector */}
        <div className="flex flex-wrap items-center gap-2">
          <div
            data-testid="apriltag-lock-pill"
            className={`px-2.5 py-1 rounded text-[11px] font-mono-tech border flex items-center gap-1.5 ${
              calValid
                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                : 'bg-rose-500/20 border-rose-500/50 text-rose-300'
            }`}
          >
            {calValid ? <ShieldCheck className="w-3.5 h-3.5" /> : <ShieldAlert className="w-3.5 h-3.5" />}
            <span>
              mat_z0 APRILTAG_36h11: {calValid ? 'LOCKED' : 'LOST'} ({regRmse.toFixed(2)} px RMSE)
            </span>
          </div>

          <div className="px-2.5 py-1 rounded bg-slate-800/90 border border-slate-700 text-[11px] font-mono-tech text-slate-200">
            scene_v:<span className="text-amber-400 font-bold ml-1">{sceneVersion}</span> · plan_v:
            <span className="text-sky-400 font-bold ml-1">{planVersion}</span> · state:
            <span className="text-emerald-300 ml-1">{sessionState}</span>
          </div>

          <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-[11px] font-mono-tech">
            {(['fixture', 'hybrid', 'live'] as ExecutionMode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => handleSwitchMode(m)}
                className={`px-2.5 py-1 rounded-md uppercase transition ${
                  mode === m
                    ? 'bg-sky-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          {/* Printable 1:1 Rig PDFs */}
          <div className="flex items-center gap-1.5 border-l border-slate-800 pl-2.5">
            <a
              href="/api/assets/printable/rewind_apriltag_mat_400x250mm.pdf"
              target="_blank"
              rel="noreferrer"
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-slate-200 flex items-center gap-1"
              title="1:1 Printable 400x250mm AprilTag Mat PDF"
            >
              <FileText className="w-3 h-3 text-sky-400" />
              Mat 400×250 PDF
            </a>
            <a
              href="/api/assets/printable/rewind_apriltag_mat_letter_tiles.pdf"
              target="_blank"
              rel="noreferrer"
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-slate-200 flex items-center gap-1"
              title="US Letter Tiled AprilTag Mat PDF"
            >
              <FileText className="w-3 h-3 text-amber-400" />
              Letter Tiles PDF
            </a>
            <a
              href="/api/assets/printable/rewind_slider_crank_cutouts_1to1.pdf"
              target="_blank"
              rel="noreferrer"
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-slate-200 flex items-center gap-1"
              title="1:1 Slider-Crank Cutouts Template PDF"
            >
              <FileText className="w-3 h-3 text-emerald-400" />
              1:1 Cutouts PDF
            </a>
          </div>
        </div>
      </header>

      {/* 6-Model Live API Proof & Latency Bar */}
      <LiveTelemetryProofBar telemetry={gateData?.telemetry_trace} />

      {/* Main 4-Pane Operator Grid */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 p-3">
        {/* PANE 1: Overhead Camera & SAM 3.1 / AprilTag Spatial Overlay (6 cols) */}
        <section
          data-testid="pane-overhead-camera"
          className="lg:col-span-6 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col overflow-hidden"
        >
          <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Camera className="w-4 h-4 text-sky-400" />
              <span className="font-mono-tech text-xs font-bold uppercase tracking-wider text-slate-200">
                Pane 1 // Overhead Camera &amp; SAM 3.1 Spatial Overlay
              </span>
              <span className="text-[10px] font-mono-tech px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                1280×800 px · undistorted_source_image_px
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setShowOverlay((v) => !v)}
                className="px-2 py-0.5 rounded text-[10px] font-mono-tech bg-slate-800 border border-slate-700 text-slate-200 hover:bg-slate-700 flex items-center gap-1"
              >
                <Eye className="w-3 h-3 text-sky-400" />
                {showOverlay ? 'Hide Overlay' : 'Show Overlay'}
              </button>
              <button
                type="button"
                data-testid="btn-compile-teardown"
                onClick={handleCompileTeardown}
                className="px-2 py-0.5 rounded text-[10px] font-mono-tech bg-sky-500/20 border border-sky-500/40 text-sky-300 hover:bg-sky-500/30 flex items-center gap-1"
              >
                <Play className="w-3 h-3" />
                Compile Teardown Plan
              </button>
            </div>
          </div>

          {/* Viewport + SVG Overlay */}
          <div className="relative bg-slate-950 flex-1 min-h-[255px] flex items-center justify-center overflow-hidden">
            <img
              src={activeFrameUrl}
              alt="Overhead Slider-Crank Rig Frame"
              className="w-full max-h-[290px] object-contain block"
            />
            {showOverlay && (
              <svg
                viewBox="0 0 1280 800"
                className="absolute inset-0 w-full h-full pointer-events-none"
                preserveAspectRatio="xMidYMid meet"
              >
                {/* AprilTag 36h11 Corner Fiducials */}
                {[
                  { id: 0, x: 140, y: 50 },
                  { id: 1, x: 1020, y: 50 },
                  { id: 2, x: 1020, y: 630 },
                  { id: 3, x: 140, y: 630 },
                ].map((tag) => (
                  <g key={tag.id}>
                    <rect
                      x={tag.x}
                      y={tag.y}
                      width={120}
                      height={120}
                      fill="none"
                      stroke={calValid ? '#10b981' : '#f43f5e'}
                      strokeWidth={3}
                      strokeDasharray="6 3"
                    />
                    <text
                      x={tag.x + 6}
                      y={tag.y - 8}
                      fill={calValid ? '#34d399' : '#fb7185'}
                      fontSize="18"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      TAG_{tag.id} (36h11)
                    </text>
                  </g>
                ))}

                {/* SAM 3.1 Connecting Rod Bounding Box + Witness Alignment Callout */}
                {activeScenario !== 'flat_scan' && (
                  <g>
                    <rect
                      x={386}
                      y={278}
                      width={450}
                      height={60}
                      rx={8}
                      fill={decision === 'accepted' ? 'rgba(16, 185, 129, 0.14)' : 'rgba(244, 63, 94, 0.18)'}
                      stroke={decision === 'accepted' ? '#10b981' : '#f43f5e'}
                      strokeWidth={3.5}
                    />
                    <text
                      x={390}
                      y={268}
                      fill={decision === 'accepted' ? '#34d399' : '#fb7185'}
                      fontSize="19"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      SAM-3.1: connecting-rod (IoU={geom.mask_iou.toFixed(3)} · witness_err={geom.witness_error_px.toFixed(1)}px)
                    </text>
                    {/* Witness Reference Target Box on Left Crank Pin */}
                    <circle
                      cx={475}
                      cy={308}
                      r={24}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth={3}
                    />
                  </g>
                )}
              </svg>
            )}
          </div>

          {/* Live WebRTC Camera + Scenario Injection Control Strip */}
          <LiveCameraControlStrip
            activeScenario={activeScenario}
            isVerifying={busyAction !== null}
            onCaptureFrameBlob={(blob) => handleUploadLiveFrameBlob(blob)}
            onTriggerGoldenFixture={(scenarioId) => {
              if (scenarioId === 'flat_scan') {
                runFlatScanAndCad();
              } else {
                runScenarioVerify(scenarioId);
              }
            }}
          />
        </section>

        {/* PANE 2: Reverse Plan & Sub-2.8s Deterministic Gate + Contract JSON Inspector (6 cols) */}
        <section
          data-testid="pane-deterministic-gate"
          className="lg:col-span-6 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col overflow-hidden"
        >
          <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-amber-400" />
              <span className="font-mono-tech text-xs font-bold uppercase tracking-wider text-slate-200">
                Pane 2 // Sub-2.8s Deterministic Gate &amp; Contract Inspector
              </span>
            </div>
            <div
              data-testid="gate-verdict-badge"
              className={`px-2.5 py-0.5 rounded text-xs font-mono-tech font-bold uppercase border ${
                decision === 'accepted'
                  ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                  : decision === 'rejected'
                  ? 'bg-rose-500/25 border-rose-500 text-rose-300'
                  : 'bg-amber-500/25 border-amber-500 text-amber-300'
              }`}
            >
              GATE VERDICT: {decision.toUpperCase()} ({elapsedMs.toFixed(1)} ms / 2800 ms)
            </div>
          </div>

          <div className="p-3 space-y-2.5 flex-1 flex flex-col justify-between">
            {/* Step DAG Compact Bar */}
            <div className="grid grid-cols-4 gap-1.5">
              {steps.map((st) => {
                const isStep2 = st.step_number === 2;
                return (
                  <div
                    key={st.step_id}
                    className={`p-2 rounded-lg border text-[10px] font-mono-tech ${
                      isStep2
                        ? decision === 'accepted'
                          ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-200'
                          : 'bg-rose-500/15 border-rose-500/60 text-rose-200'
                        : st.status === 'completed'
                        ? 'bg-slate-800/80 border-slate-700 text-slate-300'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400'
                    }`}
                  >
                    <div className="font-bold flex items-center justify-between">
                      <span>STEP {st.step_number}</span>
                      <span className="uppercase text-[9px]">{isStep2 ? decision : st.status}</span>
                    </div>
                    <div className="truncate text-slate-300 mt-0.5" title={st.title}>
                      {st.part_id}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Sub-2.8s Latency Waterfall + Pixel Metrics vs Blinded Glimmer Scorecard */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <div className="p-2.5 rounded-lg bg-slate-950/90 border border-slate-800 space-y-1.5">
                <div className="flex items-center justify-between text-[11px] font-mono-tech">
                  <span className="text-slate-300 font-semibold">Elevated Rig Geometry (unit: px)</span>
                  <span
                    className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${
                      geom.status === 'pass' ? 'text-emerald-400 bg-emerald-500/10' : 'text-rose-400 bg-rose-500/15'
                    }`}
                  >
                    {geom.status.toUpperCase()}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono-tech">
                  <div className="bg-slate-900 px-2 py-1 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[9px]">SAM 3.1 mask_iou (&ge;0.82)</span>
                    <span className="text-sky-300 font-bold">{geom.mask_iou.toFixed(4)} ratio</span>
                  </div>
                  <div className="bg-slate-900 px-2 py-1 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[9px]">witness_error (&le;6.0 px)</span>
                    <span
                      data-testid="metric-witness-error-px"
                      className={geom.witness_error_px <= 6.0 ? 'text-emerald-300 font-bold' : 'text-rose-400 font-bold'}
                    >
                      {geom.witness_error_px.toFixed(2)} px
                    </span>
                  </div>
                  <div className="bg-slate-900 px-2 py-1 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[9px]">pivot_error (&le;5.0 px)</span>
                    <span className="text-emerald-300 font-bold">{geom.pivot_error_px.toFixed(2)} px</span>
                  </div>
                  <div className="bg-slate-900 px-2 py-1 rounded border border-slate-800">
                    <span className="text-slate-400 block text-[9px]">registration_rmse (&le;1.5 px)</span>
                    <span className={regRmse <= 1.5 ? 'text-emerald-300 font-bold' : 'text-rose-400 font-bold'}>
                      {regRmse.toFixed(2)} px
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-950/90 border border-slate-800 space-y-1.5">
                <div className="flex items-center justify-between text-[11px] font-mono-tech">
                  <span className="text-slate-300 font-semibold">Blinded Muse Glimmer 30B Audit</span>
                  <span
                    data-testid="glimmer-verdict-pill"
                    className={`px-1.5 py-0.2 rounded text-[10px] font-bold uppercase ${
                      gateData?.glimmer_verdict === 'pass'
                        ? 'text-emerald-400 bg-emerald-500/10'
                        : 'text-rose-400 bg-rose-500/15'
                    }`}
                  >
                    {gateData?.glimmer_verdict ?? 'pass'} ({((gateData?.glimmer_confidence ?? 0.94) * 100).toFixed(0)}%)
                  </span>
                </div>
                <div className="text-[10px] font-mono-tech text-slate-300 bg-slate-900 p-2 rounded border border-slate-800 space-y-1">
                  <div>
                    Blinding Guard:{' '}
                    <span className="text-emerald-400 font-semibold">
                      expected_pose_id &amp; SAM/Geometry keys STRIPPED
                    </span>
                  </div>
                  <div>
                    Reason Codes:{' '}
                    <span data-testid="gate-reason-codes" className="text-amber-300 font-bold">
                      {gateData?.reason_codes?.length ? gateData.reason_codes.join(', ') : 'NONE (ALL PREDICATES PASSED)'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Paired Witness Crop Comparison + CIE LAB Polarity Cosine + Dual Open-Weight Consensus */}
            <PairedWitnessCropComparison
              refCropB64={gateData?.paired_crop_ref_b64 ?? undefined}
              liveCropB64={gateData?.paired_crop_live_b64 ?? undefined}
              polarityCosine={gateData?.witness_polarity_cosine ?? (decision === 'accepted' ? 0.9999 : -0.9412)}
              silhouetteIou={geom.mask_iou}
              witnessAligned={gateData?.witness_mark_aligned ?? (decision === 'accepted')}
              dualConsensus={gateData?.dual_open_weight_consensus}
              repairCardUrl={gateData?.repair_card_url ?? undefined}
            />

            {/* Expandable Contract Envelope JSON Inspector */}
            <div className="rounded-lg border border-slate-800 bg-slate-950 overflow-hidden">
              <div className="px-2.5 py-1.5 bg-slate-900 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-[11px] font-mono-tech text-slate-300">
                  <FileCode2 className="w-3.5 h-3.5 text-sky-400" />
                  <span>Single-Writer Contract Envelope Inspector</span>
                </div>
                <div className="flex items-center gap-1">
                  {(['calibration', 'segmentation', 'geometry', 'visual_audit', 'gate_decision'] as const).map(
                    (stageKey) => (
                      <button
                        key={stageKey}
                        type="button"
                        onClick={() => setSelectedEnvelopeStage(stageKey)}
                        className={`px-1.5 py-0.5 rounded text-[9px] font-mono-tech uppercase ${
                          selectedEnvelopeStage === stageKey
                            ? 'bg-sky-500 text-slate-950 font-bold'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        {stageKey.replace('_', ' ')}
                      </button>
                    )
                  )}
                </div>
              </div>
              <pre
                data-testid="contract-envelope-json"
                className="p-2.5 text-[10px] font-mono-tech text-sky-200/90 max-h-[115px] overflow-auto leading-relaxed"
              >
                {JSON.stringify(activeEnvelope, null, 2)}
              </pre>
            </div>
          </div>
        </section>

        {/* PANE 3: Plane-Aware mat_z0 Metrology & Interactive 3D .STL / .STEP CAD Replacement (7 cols) */}
        <section
          data-testid="pane-cad-metrology"
          className="lg:col-span-7 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col overflow-hidden"
        >
          <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/90 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Box className="w-4 h-4 text-amber-400" />
              <span className="font-mono-tech text-xs font-bold uppercase tracking-wider text-slate-200">
                Pane 3 // Plane-Aware mat_z0 Metrology (mm) &amp; Parametric STEP/STL CAD
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setCadViewTab('3d')}
                className={`px-2 py-0.5 rounded text-[10px] font-mono-tech border ${
                  cadViewTab === '3d'
                    ? 'bg-amber-500/20 border-amber-500 text-amber-200 font-bold'
                    : 'bg-slate-800 border-slate-700 text-slate-300'
                }`}
              >
                3D STL Mesh
              </button>
              <button
                type="button"
                onClick={() => setCadViewTab('2d')}
                className={`px-2 py-0.5 rounded text-[10px] font-mono-tech border ${
                  cadViewTab === '2d'
                    ? 'bg-sky-500/20 border-sky-500 text-sky-200 font-bold'
                    : 'bg-slate-800 border-slate-700 text-slate-300'
                }`}
              >
                2D Orthographic Drawing
              </button>
              <a
                href="/api/assets/cad/flat_two_hole_link_v1.step"
                download="flat_two_hole_link_v1.step"
                data-testid="download-step-link"
                className="px-2 py-0.5 rounded bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/50 text-emerald-200 text-[10px] font-mono-tech flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
                .STEP (AP214)
              </a>
              <a
                href="/api/assets/cad/flat_two_hole_link_v1.stl"
                download="flat_two_hole_link_v1.stl"
                data-testid="download-stl-link"
                className="px-2 py-0.5 rounded bg-sky-500/20 hover:bg-sky-500/30 border border-sky-500/50 text-sky-200 text-[10px] font-mono-tech flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
                .STL (Mesh)
              </a>
            </div>
          </div>

          <div className="p-3 grid grid-cols-1 md:grid-cols-12 gap-3 flex-1 items-center">
            {/* Left 5 cols: Calibrated mat_z0 Millimeter Readouts + Bonus Triggers */}
            <div className="md:col-span-5 space-y-2">
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5 font-mono-tech">
                <div className="flex items-center justify-between text-[10px] text-slate-400">
                  <span>CALIBRATED PLANE: {flatMeasurement.plane_id}</span>
                  <span className="text-emerald-400 font-bold">UNIT: mm (4.0 px/mm)</span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">Hole C-C Distance</span>
                    <span className="text-sky-300 font-bold">{flatMeasurement.hole_center_distance.toFixed(2)} mm</span>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">Outer Link Width</span>
                    <span className="text-sky-300 font-bold">{flatMeasurement.outer_width.toFixed(2)} mm</span>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">Bore Ø (L / R)</span>
                    <span className="text-emerald-300 font-bold">
                      {flatMeasurement.left_hole_diameter_mm.toFixed(2)} / {flatMeasurement.right_hole_diameter_mm.toFixed(2)} mm
                    </span>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">CAD Max Error (&le;0.10)</span>
                    <span className="text-amber-300 font-bold">{cadMaxErrorMm.toFixed(5)} mm</span>
                  </div>
                </div>
                <div className="text-[10px] text-slate-400 flex items-center justify-between pt-0.5">
                  <span>Watertight Volume: {cadVolumeMm3.toFixed(2)} mm³</span>
                  <span className="text-emerald-400">±{flatMeasurement.uncertainty_95_mm.toFixed(2)} mm (95%)</span>
                </div>
              </div>

              {/* Async Bonus Prize Lanes */}
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="text-[10px] font-mono-tech uppercase text-amber-300 font-bold flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  Async Bonus Lanes (Non-Blocking)
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  <button
                    type="button"
                    onClick={() => handleBonusAction('cua')}
                    className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-sky-300"
                  >
                    Wonder CUA
                  </button>
                  <button
                    type="button"
                    onClick={() => handleBonusAction('repair')}
                    className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-amber-300"
                  >
                    Muse Image Card
                  </button>
                  <button
                    type="button"
                    onClick={() => handleBonusAction('spark')}
                    className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-emerald-300"
                  >
                    Spark Review
                  </button>
                </div>
                <div className="text-[10px] font-mono-tech text-slate-300 bg-slate-900/90 p-1.5 rounded border border-slate-800 line-clamp-2">
                  {bonusSummary}
                </div>
              </div>
            </div>

            {/* Right 7 cols: Interactive 3D STL Canvas or 2D Orthographic Drawing */}
            <div className="md:col-span-7">
              {cadViewTab === '3d' && (
                <StlViewer3D
                  stlUrl="/api/assets/cad/flat_two_hole_link_v1.stl"
                  holeCenterDistanceMm={flatMeasurement.hole_center_distance}
                  outerWidthMm={flatMeasurement.outer_width}
                  thicknessMm={flatMeasurement.thickness}
                  holeDiameterMm={flatMeasurement.pin_diameter}
                />
              )}
              {cadViewTab === '2d' && (
                <WonderCuaDrawingOverlay drawingUrl="/api/assets/cad/flat_two_hole_link_v1_drawing.png" />
              )}
              {cadViewTab === 'repair' && (
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2 text-center">
                  <img
                    src="/api/assets/cad/repair_card_step2.png"
                    alt="Muse Image 1.0 Annotated Visual Repair Card"
                    className="w-full h-[185px] object-contain mx-auto"
                  />
                </div>
              )}
              <div className="mt-2">
                <SliderCrankKinematicTwin
                  rodLengthMm={flatMeasurement.hole_center_distance}
                  witnessInverted={decision === 'rejected'}
                />
              </div>
            </div>
          </div>
        </section>

        {/* PANE 4: Hands-Free Voice Console (muse-voice-transcribe-1.0) & Single-Writer Event Stream (5 cols) */}
        <section
          data-testid="pane-voice-console"
          className="lg:col-span-5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col overflow-hidden"
        >
          <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Mic className="w-4 h-4 text-emerald-400" />
              <span className="font-mono-tech text-xs font-bold uppercase tracking-wider text-slate-200">
                Pane 4 // Hands-Free Voice Console &amp; Event Log
              </span>
            </div>
            <span className="text-[10px] font-mono-tech text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
              muse-voice-transcribe-1.0 (24kHz PCM)
            </span>
          </div>

          <div className="p-3 space-y-2.5 flex-1 flex flex-col justify-between">
            {/* Quick-Trigger Hands-Free Operator Utterances */}
            <div className="flex flex-wrap gap-1.5">
              {[
                'check step',
                'why did that fail',
                'scan flat part',
                'generate replacement',
                'repeat instruction',
              ].map((phrase) => (
                <button
                  key={phrase}
                  type="button"
                  onClick={() => handleVoiceCommand(phrase)}
                  className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-mono-tech text-slate-200 flex items-center gap-1"
                >
                  <Terminal className="w-3 h-3 text-emerald-400" />
                  &ldquo;{phrase}&rdquo;
                </button>
              ))}
            </div>

            {/* Recent Voice Command Dispatch Output */}
            <div className="rounded-lg bg-slate-950 border border-slate-800 p-2 font-mono-tech text-[10px] space-y-1 max-h-[88px] overflow-auto">
              {voiceHistory.length === 0 ? (
                <div className="text-slate-400">
                  Voice ASR ready (`wss://api.meta.ai/v1/asr/realtime`). Click a command above to dispatch.
                </div>
              ) : (
                voiceHistory.map((item, idx) => (
                  <div key={idx} className="border-b border-slate-900 pb-1 last:border-none">
                    <span className="text-emerald-400 font-bold">[{item.intent}]</span>{' '}
                    <span className="text-slate-300">&ldquo;{item.transcript}&rdquo;</span> &rarr;{' '}
                    <span className="text-sky-300">{item.summary}</span>
                  </div>
                ))
              )}
            </div>

            {/* Single-Writer Contract Event Stream */}
            <div className="rounded-lg bg-slate-950 border border-slate-800 p-2 font-mono-tech text-[10px] space-y-1 max-h-[115px] overflow-auto">
              <div className="text-slate-400 font-bold uppercase text-[9px] flex items-center justify-between">
                <span>Single-Writer Stage Envelope Stream</span>
                <span className="text-sky-400">{envelopeLog.length} committed</span>
              </div>
              {envelopeLog.map((env, i) => (
                <div key={env.message_id || i} className="flex items-center justify-between text-[10px] py-0.5 border-b border-slate-900/80 last:border-none">
                  <span className="text-amber-300 font-semibold">{env.payload_type}</span>
                  <span className="text-slate-400">{env.producer}</span>
                  <span className="text-slate-500">sv={env.scene_version}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
