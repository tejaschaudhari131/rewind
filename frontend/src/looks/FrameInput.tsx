import { useEffect, useRef, useState } from 'react';
import { FRAME } from './useRewindCheck';
import type { Source } from './useRewindCheck';
import { FrameSvg } from './Shared';

type C = ReturnType<typeof import('./useRewindCheck').useRewindCheck>;

/** Fixture | Upload | Live camera switch, plus the eval-save and auto-check toggles. Styling is passed in by each look. */
export function SourceSwitch({ c, btn, active, compact = false }: { c: C; btn: string; active: string; compact?: boolean }) {
  const opts: Array<[Source, string]> = [['fixture', 'Fixture frames'], ['upload', 'Upload photo / video'], ['camera', 'Live camera']];
  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="source-switch">
      {opts.map(([s, l]) => <button key={s} onClick={() => c.setSource(s)} className={`${btn} ${c.source === s ? active : ''}`}>{l}</button>)}
      {c.realInput && <label className="flex items-center gap-1 text-xs font-bold" title="Off by default: frames are deleted after the check"><input type="checkbox" checked={c.saveEval} onChange={(e) => c.setSaveEval(e.target.checked)} /> save to eval set</label>}
      {c.source === 'camera' && !compact && <label className="flex items-center gap-1 text-xs font-bold"><input type="checkbox" checked={c.autoCheck} onChange={(e) => c.setAutoCheck(e.target.checked)} /> auto-check when still ~1s</label>}
    </div>
  );
}

export function ApiBanner({ c }: { c: C }) {
  if (!c.apiDown) return null;
  return <div className="fixed top-0 inset-x-0 z-50 bg-[#d71920] text-white text-center font-black py-2 text-lg" role="alert" data-testid="api-banner">
    Meta API unreachable. Keep your computer online. {c.preflight?.error ? <span className="font-mono-tech text-xs opacity-80">({c.preflight.error})</span> : null}
  </div>;
}

/** Renders the current input (fixture frame, uploaded photo/video, live camera) and registers the frame grabber. */
export function FrameStage({ c, fit = 'meet', markerColor, strokeW = 4, className = '', ghost = null, dark = false }:
  { c: C; fit?: 'meet' | 'slice'; markerColor: string; strokeW?: number; className?: string; ghost?: string | null; dark?: boolean }) {
  const holes = c.s1?.hole_pattern_witness?.live_holes_px ?? [];
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileVideoRef = useRef<HTMLVideoElement>(null);
  const [file, setFile] = useState<{ url: string; kind: 'image' | 'video'; f: File; w: number; h: number } | null>(null);
  const [camErr, setCamErr] = useState<string | null>(null);
  const [live, setLive] = useState(true); // camera: showing the live feed vs. the still that was sent
  const [still, setStillState] = useState<'idle' | 'moving' | 'still'>('idle');
  const [dur, setDur] = useState(0); const [t, setT] = useState(0);
  const runRef = useRef(c.run); runRef.current = c.run;
  const phaseRef = useRef(c.phase); phaseRef.current = c.phase;

  // camera stream
  useEffect(() => {
    if (c.source !== 'camera') return;
    let stream: MediaStream | null = null; let dead = false;
    setCamErr(null);
    if (!navigator.mediaDevices?.getUserMedia) { setCamErr(window.isSecureContext ? 'No camera API in this browser' : 'Camera needs HTTPS: open the phone link (scripts/serve_phone.sh)'); return; }
    navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
      .then((s) => { if (dead) { s.getTracks().forEach((x) => x.stop()); return; } stream = s; if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play().catch(() => {}); } })
      .catch((e) => setCamErr(`${e?.name ?? 'Error'}: ${e?.message ?? e}`));
    return () => { dead = true; stream?.getTracks().forEach((x) => x.stop()); };
  }, [c.source]);

  // after a check in camera mode, show the still that was actually sent; any new idle returns to live
  useEffect(() => { if (c.source === 'camera') setLive(!(c.still && c.phase !== 'idle')); }, [c.still, c.phase, c.source]);

  // grabber
  useEffect(() => {
    const toBlob = (el: HTMLVideoElement, kind: 'camera' | 'video_frame') => new Promise<any>((resolve) => {
      const w = el.videoWidth, h = el.videoHeight; if (!w || !h) return resolve(null);
      const cv = document.createElement('canvas'); cv.width = w; cv.height = h; cv.getContext('2d')!.drawImage(el, 0, 0, w, h);
      cv.toBlob((b) => resolve(b ? { blob: b, w, h, kind, name: `${kind}.jpg` } : null), 'image/jpeg', 0.92);
    });
    c.grabRef.current = async () => {
      if (c.source === 'camera') return videoRef.current ? toBlob(videoRef.current, 'camera') : null;
      if (c.source === 'upload' && file) {
        if (file.kind === 'video') return fileVideoRef.current ? toBlob(fileVideoRef.current, 'video_frame') : null;
        return { blob: file.f, w: file.w, h: file.h, kind: 'upload', name: file.f.name || 'photo.jpg' };
      }
      return null;
    };
    return () => { c.grabRef.current = null; };
  }, [c.source, file]);

  // optional auto-check: fire once when the scene has been still ~1 s after motion (labelled in the UI)
  useEffect(() => {
    if (c.source !== 'camera' || !c.autoCheck) { setStillState('idle'); return; }
    const cv = document.createElement('canvas'); cv.width = 64; cv.height = 36; const ctx = cv.getContext('2d', { willReadFrequently: true })!;
    let prev: Uint8ClampedArray | null = null, stillMs = 0, armed = true;
    const iv = setInterval(() => {
      const v = videoRef.current; if (!v || !v.videoWidth) return;
      ctx.drawImage(v, 0, 0, 64, 36); const d = ctx.getImageData(0, 0, 64, 36).data;
      if (prev) {
        let s = 0; for (let i = 0; i < d.length; i += 4) s += Math.abs(d[i] - prev[i]) + Math.abs(d[i + 1] - prev[i + 1]);
        const m = s / (64 * 36 * 2);
        if (m > 6) { stillMs = 0; armed = true; setStillState('moving'); }
        else { stillMs += 200; if (stillMs >= 1000) { setStillState('still'); if (armed && phaseRef.current !== 'checking') { armed = false; runRef.current(); } } }
      }
      prev = new Uint8ClampedArray(d);
    }, 200);
    return () => clearInterval(iv);
  }, [c.source, c.autoCheck]);

  const onFile = (f: File | undefined) => {
    if (!f) return;
    const url = URL.createObjectURL(f);
    if (f.type.startsWith('video/')) { setFile({ url, kind: 'video', f, w: 0, h: 0 }); return; }
    const im = new Image(); im.onload = () => setFile({ url, kind: 'image', f, w: im.naturalWidth, h: im.naturalHeight });
    im.onerror = () => setFile({ url, kind: 'image', f, w: 1280, h: 720 }); // e.g. HEIC on non-Safari: server converts it
    im.src = url;
  };

  const shownStill = c.still && c.phase !== 'idle' ? c.still : null;
  const objFit = fit === 'slice' ? 'object-cover' : 'object-contain';
  const pill = `absolute z-10 px-2 py-0.5 text-xs font-black ${dark ? 'bg-black/70 text-white' : 'bg-white border-2 border-black'}`;

  if (c.source === 'fixture') return <FrameSvg src={FRAME[c.scenario]} holes={holes} ghost={ghost} fit={fit} markerColor={markerColor} strokeW={strokeW} className={className} />;

  return (
    <div className={`relative ${className}`} style={{ background: '#000' }}>
      {c.source === 'camera' && <>
        <video ref={videoRef} playsInline muted autoPlay className={`absolute inset-0 w-full h-full ${objFit} ${live ? '' : 'invisible'}`} />
        {camErr && <div className="absolute inset-0 grid place-items-center text-white text-center p-6 text-lg font-bold">{camErr}</div>}
        {c.autoCheck && <div className={`${pill} left-3 bottom-3`}>AUTO-CHECK · {still === 'still' ? 'scene still' : still === 'moving' ? 'motion…' : 'waiting'}</div>}
        {!live && <button onClick={() => setLive(true)} className={`${pill} right-3 bottom-3 pointer-events-auto`}>▶ back to live</button>}
      </>}
      {c.source === 'upload' && !shownStill && (file ? (
        file.kind === 'image' ? <img src={file.url} className={`absolute inset-0 w-full h-full ${objFit}`} alt="uploaded" />
          : <>
            <video ref={fileVideoRef} src={file.url} muted playsInline preload="auto" className={`absolute inset-0 w-full h-full ${objFit}`}
              onLoadedMetadata={(e) => setDur(e.currentTarget.duration || 0)} />
            <div className="absolute inset-x-4 bottom-3 z-10 flex items-center gap-2 bg-black/70 text-white px-3 py-2 rounded pointer-events-auto">
              <span className="text-xs font-bold">pick frame</span>
              <input type="range" min={0} max={dur || 0} step={0.033} value={t} className="flex-1"
                onChange={(e) => { const v = Number(e.target.value); setT(v); if (fileVideoRef.current) fileVideoRef.current.currentTime = v; }} />
              <span className="font-mono-tech text-xs w-14 text-right">{t.toFixed(2)}s</span>
            </div>
          </>
      ) : (
        <label className="absolute inset-0 grid place-items-center cursor-pointer text-white text-center pointer-events-auto">
          <div><div className="text-3xl font-black">Tap to choose a photo or video</div><div className="text-sm opacity-70 mt-1">JPG / PNG / HEIC (≤ 8 MB), or MP4/MOV, then pick a frame</div></div>
          <input type="file" accept="image/*,video/*,.heic,.heif" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
      ))}
      {c.source === 'upload' && file && !shownStill && <label className={`${pill} right-3 top-3 cursor-pointer pointer-events-auto`}>change file<input type="file" accept="image/*,video/*,.heic,.heif" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} /></label>}
      {shownStill && (c.source === 'upload' || !live) &&
        <FrameSvg src={shownStill.url} w={c.res?.frame_size?.[0] ?? shownStill.w} h={c.res?.frame_size?.[1] ?? shownStill.h} holes={holes} fit={fit} markerColor={markerColor} strokeW={strokeW} className="absolute inset-0 w-full h-full" />}
      {shownStill && c.source === 'upload' && <button onClick={() => c.setSource('upload')} className={`${pill} right-3 top-3 pointer-events-auto`}>edit / retake</button>}
    </div>
  );
}
