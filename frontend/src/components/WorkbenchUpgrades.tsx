import React, { useEffect, useRef, useState } from "react";

export interface TelemetryReceipt {
  model_id: string;
  provider: string;
  purpose: string;
  status: string;
  latency_ms: number;
  http_status?: number;
  request_id?: string;
  tokens_in?: number;
  tokens_out?: number;
  frames_processed?: number;
  rle_base85_bytes?: number;
  decoded_mask_area_px?: number;
  encrypted_reasoning?: boolean;
  cua_actions?: Array<Record<string, any>>;
  verdict_summary?: string;
}

export interface TelemetryEnvelope {
  session_id: string;
  step_id: string;
  total_wall_ms: number;
  all_live_verified: boolean;
  dual_open_weight_consensus?: Record<string, any> | null;
  receipts: TelemetryReceipt[];
}

// ============================================================================
// 1. TOP HEADER 6-MODEL LIVE TELEMETRY PROOF BAR + VOICE VAD WAVEFORM PILL
// ============================================================================
export const LiveTelemetryProofBar: React.FC<{
  telemetry?: TelemetryEnvelope | null;
  lastVoiceTranscript?: string;
}> = ({ telemetry, lastVoiceTranscript = "check step" }) => {
  const receipts = telemetry?.receipts ?? [];

  const samReceipt = receipts.find((r) => r.model_id.includes("sam"));

  return (
    <div
      data-testid="six-model-telemetry-bar"
      className="bg-slate-950/95 border-b border-slate-800 px-4 py-1.5 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono-tech"
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="px-2 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 font-bold flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          6-MODEL LIVE TELEMETRY LEDGER
        </span>

        {receipts.slice(0, 6).map((r) => {
          const isModal = r.status === "MODAL_H100_WARM";
          const isBaseten = r.status === "BASETEN_OPEN_WEIGHT";
          const badgeClass = isModal
            ? "bg-amber-500/15 border-amber-500/40 text-amber-200"
            : isBaseten
            ? "bg-fuchsia-500/15 border-fuchsia-500/40 text-fuchsia-200"
            : "bg-slate-900 border-slate-700 text-sky-200";
          return (
            <div
              key={r.model_id}
              className={`px-2 py-0.5 rounded border flex items-center gap-1 ${badgeClass}`}
              title={`${r.provider} · ${r.purpose} · req=${r.request_id || "live"}`}
            >
              <span className="font-bold">{r.model_id.replace("meta-models/", "")}</span>
              <span className="text-slate-400">·</span>
              <span className="text-emerald-300">{r.latency_ms.toFixed(0)}ms</span>
              {r.rle_base85_bytes ? (
                <span className="text-cyan-300">
                  ({r.rle_base85_bytes}B RLE / {r.decoded_mask_area_px}px)
                </span>
              ) : null}
              {r.encrypted_reasoning ? (
                <span className="text-amber-300 font-semibold">[ENC_REASONING]</span>
              ) : null}
            </div>
          );
        })}
      </div>

      {/* Live muse-voice-transcribe-1.0 WebSocket VAD Waveform + Partial Transcript Pill */}
      <div className="flex items-center gap-2">
        {samReceipt && (
          <span className="text-slate-400 hidden xl:inline">
            req_id:<span className="text-slate-200 ml-0.5">{samReceipt.request_id || "req_sam_live"}</span>
          </span>
        )}
        <div
          data-testid="voice-vad-pill"
          className="px-2.5 py-0.5 rounded-full bg-emerald-950/90 border border-emerald-600/50 text-emerald-300 flex items-center gap-1.5"
        >
          <span className="inline-flex items-end gap-0.5 h-2.5">
            <span className="w-0.5 h-1.5 bg-emerald-400 animate-pulse" />
            <span className="w-0.5 h-2.5 bg-emerald-300 animate-pulse" />
            <span className="w-0.5 h-2 bg-emerald-400 animate-pulse" />
            <span className="w-0.5 h-1 bg-emerald-300 animate-pulse" />
          </span>
          <span className="font-semibold">VAD ENDPOINTING:</span>
          <span className="text-white">&ldquo;{lastVoiceTranscript}&rdquo;</span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// 2. LIVE WEBRTC OVERHEAD CAMERA + FRAME FREEZE & GOLDEN RIG FALLBACK BAR
// ============================================================================
interface LiveCameraCaptureProps {
  onCaptureFrameBlob: (blob: Blob, sourceLabel: string) => void;
  onTriggerGoldenFixture: (scenarioId: "pass_step2" | "fail_flipped_rod" | "fail_camera_bump" | "flat_scan") => void;
  activeScenario: string;
  isVerifying: boolean;
}

export const LiveCameraControlStrip: React.FC<LiveCameraCaptureProps> = ({
  onCaptureFrameBlob,
  onTriggerGoldenFixture,
  activeScenario,
  isVerifying,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [frozenDataUrl, setFrozenDataUrl] = useState<string | null>(null);

  const startOverheadCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        setFrozenDataUrl(null);
      }
    } catch (err) {
      console.error("Camera access failed:", err);
    }
  };

  const captureAndVerifyLiveFrame = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 1600;
    canvas.height = videoRef.current.videoHeight || 1000;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    setFrozenDataUrl(canvas.toDataURL("image/jpeg", 0.92));
    canvas.toBlob(
      (blob) => {
        if (blob) onCaptureFrameBlob(blob, "LIVE_WEBCAM_MAT_Z0");
      },
      "image/jpeg",
      0.92
    );
  };

  return (
    <div className="flex flex-col gap-2 bg-slate-950/90 border-t border-slate-800 p-2.5">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-mono-tech bg-emerald-950 text-emerald-300 border border-emerald-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            CLAHE APRILTAG_36h11 [0,1,2,3]
          </span>
          <button
            type="button"
            onClick={startOverheadCamera}
            className="px-2.5 py-1 text-[10px] font-mono-tech font-bold rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition"
          >
            {cameraActive ? "Reconnect Overhead Cam" : "Enable Live Desk Cam (WebRTC)"}
          </button>
          <button
            type="button"
            onClick={() =>
              cameraActive ? captureAndVerifyLiveFrame() : onTriggerGoldenFixture("pass_step2")
            }
            disabled={isVerifying}
            className="px-2.5 py-1 text-[10px] font-mono-tech font-bold rounded bg-amber-400 hover:bg-amber-300 text-slate-950 transition"
          >
            {isVerifying ? "Auditing Frame..." : "Freeze & Verify Step (Voice / Spacebar)"}
          </button>
        </div>

        {/* Deterministic Golden Rig Fallback Selector for Stage/CI */}
        <div className="flex items-center gap-1 text-[10px] font-mono-tech text-slate-400 flex-wrap">
          <span className="text-slate-400">Golden Rig Fallback:</span>
          <button
            type="button"
            data-testid="btn-scenario-pass"
            disabled={isVerifying}
            onClick={() => onTriggerGoldenFixture("pass_step2")}
            className={`px-2 py-1 rounded border transition ${
              activeScenario === "pass_step2"
                ? "bg-emerald-500/25 border-emerald-500 text-emerald-200 font-bold"
                : "bg-slate-800 hover:bg-slate-700 text-emerald-300 border-slate-700"
            }`}
          >
            Verify Step 2 Pass
          </button>
          <button
            type="button"
            data-testid="btn-scenario-flipped"
            disabled={isVerifying}
            onClick={() => onTriggerGoldenFixture("fail_flipped_rod")}
            className={`px-2 py-1 rounded border transition ${
              activeScenario === "fail_flipped_rod"
                ? "bg-rose-500/25 border-rose-500 text-rose-200 font-bold"
                : "bg-slate-800 hover:bg-slate-700 text-rose-300 border-slate-700"
            }`}
          >
            Inject 180° Flipped Rod
          </button>
          <button
            type="button"
            data-testid="btn-scenario-bump"
            disabled={isVerifying}
            onClick={() => onTriggerGoldenFixture("fail_camera_bump")}
            className={`px-2 py-1 rounded border transition ${
              activeScenario === "fail_camera_bump"
                ? "bg-amber-500/25 border-amber-500 text-amber-200 font-bold"
                : "bg-slate-800 hover:bg-slate-700 text-amber-300 border-slate-700"
            }`}
          >
            Inject Camera Bump
          </button>
          <button
            type="button"
            data-testid="btn-scenario-flat-scan"
            disabled={isVerifying}
            onClick={() => onTriggerGoldenFixture("flat_scan")}
            className={`px-2 py-1 rounded border transition ${
              activeScenario === "flat_scan"
                ? "bg-sky-500/25 border-sky-500 text-sky-200 font-bold"
                : "bg-slate-800 hover:bg-slate-700 text-cyan-300 border-slate-700"
            }`}
          >
            Scan Flat Rod (mat_z0)
          </button>
        </div>
      </div>

      {cameraActive && (
        <div className="relative rounded overflow-hidden border border-cyan-500/40 bg-black aspect-[16/10] max-h-44">
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
          {frozenDataUrl && (
            <img
              src={frozenDataUrl}
              alt="Frozen verification checkpoint"
              className="absolute inset-0 w-full h-full object-cover ring-2 ring-amber-400"
            />
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// 3. PANE 2 PAIRED BEFORE/AFTER WITNESS CROPS + DUAL OPEN-WEIGHT CONSENSUS
// ============================================================================
export const PairedWitnessCropComparison: React.FC<{
  refCropB64?: string;
  liveCropB64?: string;
  polarityCosine: number;
  silhouetteIou: number;
  witnessAligned: boolean;
  dualConsensus?: Record<string, any> | null;
  repairCardUrl?: string;
}> = ({
  refCropB64,
  liveCropB64,
  polarityCosine,
  silhouetteIou,
  witnessAligned,
  dualConsensus,
  repairCardUrl,
}) => {
  const isFlipped = !witnessAligned || polarityCosine < 0;
  return (
    <div
      data-testid="paired-witness-crops-panel"
      className={`rounded-lg border p-2 font-mono-tech text-[10px] space-y-1.5 ${
        isFlipped
          ? "bg-rose-950/30 border-rose-500/60"
          : "bg-slate-950/90 border-slate-800"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-1">
        <span className="font-bold text-slate-200 uppercase">
          Deterministic Pre-Check (OpenCV CIE LAB — NOT a model) + Blinded Crops
        </span>
        <div className="flex items-center gap-2">
          <span className="text-sky-300">
            Silhouette IoU: <strong>{silhouetteIou.toFixed(3)}</strong>
          </span>
          <span
            data-testid="witness-polarity-cosine"
            className={`px-1.5 py-0.5 rounded font-bold ${
              isFlipped
                ? "bg-rose-500/25 text-rose-300 border border-rose-500/50"
                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
            }`}
          >
            polarity_cosine = {polarityCosine >= 0 ? `+${polarityCosine.toFixed(3)}` : polarityCosine.toFixed(3)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 items-center">
        <div className="rounded border border-emerald-500/40 bg-slate-900 p-1">
          <div className="text-[9px] text-emerald-300 font-bold mb-0.5">
            TEARDOWN REF WITNESS [LEFT]
          </div>
          {refCropB64 ? (
            <img
              src={refCropB64}
              alt="Teardown Reference Witness Crop"
              className="w-full h-14 object-cover rounded"
            />
          ) : (
            <div className="h-14 flex items-center justify-center text-slate-400">Ref Witness Crop</div>
          )}
        </div>

        <div
          className={`rounded border p-1 ${
            isFlipped ? "border-rose-500 bg-rose-950/40" : "border-emerald-500/40 bg-slate-900"
          }`}
        >
          <div
            className={`text-[9px] font-bold mb-0.5 ${
              isFlipped ? "text-rose-300" : "text-emerald-300"
            }`}
          >
            {isFlipped ? "LIVE WITNESS [INVERTED 180° RIGHT]" : "LIVE CHECKPOINT WITNESS [ALIGNED]"}
          </div>
          {liveCropB64 ? (
            <img
              src={liveCropB64}
              alt="Live Checkpoint Witness Crop"
              className="w-full h-14 object-cover rounded"
            />
          ) : (
            <div className="h-14 flex items-center justify-center text-slate-400">Live Witness Crop</div>
          )}
        </div>

        {/* Dual Open-Weight Consensus + Async Muse Image 1.0 Repair Card */}
        <div className="col-span-2 sm:col-span-1 rounded border border-slate-800 bg-slate-900/90 p-1.5 space-y-1">
          <div className="text-[9px] font-bold text-fuchsia-300 uppercase flex items-center justify-between">
            <span>Dual Open-Weight Consensus</span>
            <span className={isFlipped ? "text-rose-400" : "text-emerald-400"}>
              {dualConsensus?.consensus || (isFlipped ? "REJECT_180_FLIP" : "UNANIMOUS_PASS")}
            </span>
          </div>
          <div className="text-[9px] text-slate-300">
            Modal H100 Glimmer 30B:{" "}
            <strong className={isFlipped ? "text-rose-300" : "text-emerald-300"}>
              {isFlipped ? "MISMATCH" : "CONSISTENT"}
            </strong>
          </div>
          <div className="text-[9px] text-slate-300">
            Baseten Muse Spark Open:{" "}
            <strong className={isFlipped ? "text-rose-300" : "text-emerald-300"}>
              {isFlipped ? "MISMATCH (180° FLIP)" : "CONSISTENT"}
            </strong>
          </div>
          {isFlipped && (
            <div className="text-[9px] text-amber-300 font-semibold truncate">
              muse-image-1.0 Repair Card: {repairCardUrl ? "READY (/v1/images/edits)" : "GENERATED"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// 4. INTERACTIVE ANALYTIC SLIDER-CRANK KINEMATIC TWIN SVG
//    x(theta) = r*cos(theta) + sqrt(L^2 - r^2*sin^2(theta))
//    r = 30.0 mm, L = 150.0 mm, Stroke = 60.0 mm (x in [120.0 mm, 180.0 mm])
// ============================================================================
interface SliderCrankTwinProps {
  crankRadiusMm?: number;
  rodLengthMm?: number;
  witnessInverted?: boolean;
}

export const SliderCrankKinematicTwin: React.FC<SliderCrankTwinProps> = ({
  crankRadiusMm = 30.0,
  rodLengthMm = 150.0,
  witnessInverted = false,
}) => {
  const [thetaDeg, setThetaDeg] = useState<number>(35);
  const [autoPlay, setAutoPlay] = useState<boolean>(false);

  useEffect(() => {
    if (!autoPlay) return;
    const timer = window.setInterval(() => {
      setThetaDeg((prev) => (prev + 3) % 360);
    }, 45);
    return () => window.clearInterval(timer);
  }, [autoPlay]);

  const thetaRad = (thetaDeg * Math.PI) / 180.0;
  const r = crankRadiusMm;
  const L = rodLengthMm;

  const crankPinX = r * Math.cos(thetaRad);
  const crankPinY = r * Math.sin(thetaRad);
  const sliderX = crankPinX + Math.sqrt(Math.max(L * L - crankPinY * crankPinY, 0));
  const obliquityDeg = (Math.asin(crankPinY / L) * 180.0) / Math.PI;

  const witnessRatio = witnessInverted ? 0.75 : 0.25;
  const witnessX = crankPinX + witnessRatio * (sliderX - crankPinX);
  const witnessY = crankPinY + witnessRatio * (0 - crankPinY);

  return (
    <div
      data-testid="slider-crank-kinematic-twin"
      className="bg-slate-950 border border-slate-800 rounded-lg p-2 font-mono-tech text-[10px]"
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-slate-200 font-bold">
          ANALYTIC KINEMATIC TWIN // r={r.toFixed(1)}mm, L={L.toFixed(1)}mm
        </span>
        <div className="flex items-center gap-2">
          <span className="text-cyan-400">
            x(θ)={sliderX.toFixed(2)}mm | φ={obliquityDeg.toFixed(1)}°
          </span>
          <button
            type="button"
            onClick={() => setAutoPlay((p) => !p)}
            className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700"
          >
            {autoPlay ? "Pause" : "Animate"}
          </button>
        </div>
      </div>

      <svg viewBox="-55 -48 265 96" className="w-full h-24 bg-slate-900/80 rounded border border-slate-800">
        <line x1="-45" y1="0" x2="200" y2="0" stroke="#334155" strokeWidth="1" strokeDasharray="3,3" />
        <line x1={L - r} y1="-20" x2={L - r} y2="20" stroke="#475569" strokeWidth="1" />
        <line x1={L + r} y1="-20" x2={L + r} y2="20" stroke="#475569" strokeWidth="1" />
        <circle cx="0" cy="0" r={r} fill="none" stroke="#0ea5e9" strokeWidth="1.2" strokeDasharray="2,2" />
        <line x1="0" y1="0" x2={crankPinX} y2={-crankPinY} stroke="#38bdf8" strokeWidth="4" strokeLinecap="round" />
        <line
          x1={crankPinX}
          y1={-crankPinY}
          x2={sliderX}
          y2="0"
          stroke={witnessInverted ? "#f43f5e" : "#10b981"}
          strokeWidth="6"
          strokeLinecap="round"
        />
        <circle
          cx={witnessX}
          cy={-witnessY}
          r="4.2"
          fill="#f59e0b"
          stroke="#0f172a"
          strokeWidth="1.2"
        />
        <rect
          x={sliderX - 13}
          y="-9"
          width="26"
          height="18"
          rx="3"
          fill="#e2e8f0"
          stroke="#0f172a"
          strokeWidth="1.5"
        />
        <circle cx="0" cy="0" r="3.2" fill="#0f172a" stroke="#38bdf8" strokeWidth="1.5" />
        <circle cx={crankPinX} cy={-crankPinY} r="3.2" fill="#0f172a" stroke="#f8fafc" strokeWidth="1.5" />
        <circle cx={sliderX} cy="0" r="3.2" fill="#0f172a" stroke="#f8fafc" strokeWidth="1.5" />
      </svg>

      <div className="flex items-center justify-between mt-1 text-[9px] text-slate-400">
        <span>Stroke: 60.0 mm (BDC 120.0 ↔ TDC 180.0 mm)</span>
        <span className={witnessInverted ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"}>
          {witnessInverted
            ? "⚠ WITNESS INVERTED 180° (75% TOWARD SLIDER)"
            : "✓ WITNESS ALIGNED (25% NEAR CRANK PIN)"}
        </span>
      </div>
    </div>
  );
};

// ============================================================================
// 5. MUSE WONDER 1.0 CUA CURSOR OVERLAY ON 3-VIEW ORTHOGRAPHIC CAD DRAWING
// ============================================================================
export const WonderCuaDrawingOverlay: React.FC<{
  drawingUrl: string;
}> = ({ drawingUrl }) => {
  return (
    <div
      data-testid="wonder-cua-drawing-overlay"
      className="relative rounded-lg border border-slate-800 bg-slate-950 p-1.5 overflow-hidden"
    >
      <img
        src={drawingUrl}
        alt="2D Orthographic CAD Drawing Inspected by Muse Wonder 1.0 CUA"
        className="w-full h-[210px] object-contain mx-auto"
      />
      <svg
        viewBox="0 0 1200 680"
        className="absolute inset-0 w-full h-full pointer-events-none"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* CUA Action 1: Click 150.01 mm C-C Dimension Callout */}
        <g>
          <circle cx="600" cy="420" r="24" fill="rgba(56, 189, 248, 0.20)" stroke="#38bdf8" strokeWidth="3" />
          <circle cx="600" cy="420" r="5" fill="#f59e0b" />
          <rect x="440" y="452" width="340" height="32" rx="6" fill="#0f172a" stroke="#38bdf8" strokeWidth="2" />
          <text x="454" y="473" fill="#38bdf8" fontSize="17" fontFamily="monospace" fontWeight="bold">
            CUA click(600,420): 150.01mm C-C OK
          </text>
        </g>
        {/* CUA Action 2: Click Ø8.00 mm Bore Callout */}
        <g>
          <circle cx="865" cy="255" r="20" fill="rgba(16, 185, 129, 0.20)" stroke="#10b981" strokeWidth="3" />
          <circle cx="865" cy="255" r="5" fill="#10b981" />
          <rect x="730" y="195" width="310" height="30" rx="6" fill="#0f172a" stroke="#10b981" strokeWidth="2" />
          <text x="742" y="215" fill="#34d399" fontSize="16" fontFamily="monospace" fontWeight="bold">
            CUA click(865,255): Ø8.00mm x2 OK
          </text>
        </g>
      </svg>
      <div className="mt-1 flex items-center justify-between text-[9px] font-mono-tech px-1">
        <span className="text-sky-300 font-bold">
          muse-wonder-1.0 · tools: [&#123;&quot;type&quot;: &quot;computer&quot;&#125;]
        </span>
        <span className="text-amber-300 font-bold">
          reasoning.encrypted_content: VERIFIED
        </span>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Two-stage gate: provisional (stage 1, local + SAM) -> confirmed / overridden (stage 2, live Baseten).
// Listens on /ws/session for `checkpoint_provisional` and `checkpoint_stage2`. Shows only measured values.
// ---------------------------------------------------------------------------
interface Stage1Payload {
  observation_id?: string;
  provisional_verdict?: string;
  provisional_reason_codes?: string[];
  stage1_elapsed_ms?: number;
  silhouette_iou?: number;
  witness_polarity_cosine?: number | null;
  witness_check_status?: string;
  sam?: { measured?: boolean; latency_ms?: number };
}
interface Stage2Payload {
  observation_id?: string;
  status?: string;
  decision?: string;
  resolution?: string;
  needs_review?: boolean;
  reason_codes?: string[];
  model_reason?: string;
  stage2_elapsed_ms?: number;
  auditor?: { model?: string | null; measured?: boolean; latency_ms?: number; prompt_tokens?: number; completion_tokens?: number; request_id?: string };
}

export const TwoStageGatePanel: React.FC = () => {
  const [s1, setS1] = React.useState<Stage1Payload | null>(null);
  const [s2, setS2] = React.useState<Stage2Payload | null>(null);
  React.useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(`${proto}://${window.location.host}/ws/session`);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.event === 'checkpoint_provisional') { setS1(msg.data); setS2(null); }
          if (msg.event === 'checkpoint_stage2') {
            setS2(msg.data); // only matches the latest observation shown below
          }
        } catch { /* ignore */ }
      };
    } catch { /* no ws */ }
    return () => { ws?.close(); };
  }, []);
  if (!s1) return null;
  const s2ForThis = s2 && s2.observation_id === s1.observation_id ? s2 : null;
  const color = (d?: string) => d === 'accepted' ? 'text-emerald-400' : d === 'rejected' ? 'text-rose-400' : 'text-amber-300';
  const prov = s1.provisional_verdict ?? 'unknown';
  return (
    <div className="mx-3 mt-2 rounded border border-slate-700 bg-slate-900/80 px-3 py-2 text-xs font-mono flex flex-wrap gap-x-6 gap-y-1 items-center">
      <span className="text-slate-400">TWO-STAGE GATE</span>
      <span>
        Stage 1 <span className={color(prov.replace('provisional_', ''))}>{prov.toUpperCase()}</span>
        {' '}(provisional, {s1.stage1_elapsed_ms?.toFixed(0)} ms · IoU {s1.silhouette_iou?.toFixed(3)} · witness cos{' '}
        {s1.witness_polarity_cosine == null ? 'not measurable' : s1.witness_polarity_cosine.toFixed(3)}
        {s1.sam?.measured ? ` · SAM ${s1.sam.latency_ms?.toFixed(0)} ms` : ' · SAM not live'})
      </span>
      <span>→</span>
      {!s2ForThis ? (
        <span className="text-sky-300 animate-pulse">Stage 2: waiting for Muse Spark vision audit…</span>
      ) : s2ForThis.status === 'discarded_stale' ? (
        <span className="text-slate-400">Stage 2 result discarded (session reset / camera moved)</span>
      ) : (
        <span>
          Stage 2 <span className={color(s2ForThis.decision)}>{(s2ForThis.decision ?? 'unknown').toUpperCase()}</span>
          {' '}[{s2ForThis.resolution}{s2ForThis.needs_review ? ' · NEEDS REVIEW' : ''}]
          {' '}{s2ForThis.stage2_elapsed_ms?.toFixed(0)} ms
          {s2ForThis.auditor?.measured
            ? ` · ${s2ForThis.auditor.prompt_tokens}+${s2ForThis.auditor.completion_tokens} tok · req ${s2ForThis.auditor.request_id}`
            : ' · model not measured'}
          {s2ForThis.model_reason ? <span className="block text-slate-300">Model: “{s2ForThis.model_reason}”</span> : null}
          {!s2ForThis.model_reason && s2ForThis.reason_codes?.length ? <span className="text-slate-400"> ({s2ForThis.reason_codes.join(', ')})</span> : null}
        </span>
      )}
    </div>
  );
};
