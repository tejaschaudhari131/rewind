import { FRAME_W, FRAME_H } from './useRewindCheck';
/** Camera frame + real hole markers in one SVG so markers stay aligned at any crop (slice = full-bleed). */
export function FrameSvg({ src, holes, ghost, fit = 'meet', markerColor = '#34d399', className = '', strokeW = 4 }:
  { src: string; holes?: number[][]; ghost?: string | null; fit?: 'meet' | 'slice'; markerColor?: string; className?: string; strokeW?: number }) {
  return (
    <svg viewBox={`0 0 ${FRAME_W} ${FRAME_H}`} preserveAspectRatio={`xMidYMid ${fit}`} className={className}>
      <image href={src} width={FRAME_W} height={FRAME_H} />
      {ghost && <image href={ghost} width={FRAME_W} height={FRAME_H} className="rw-ghost" style={{ mixBlendMode: 'screen' }} />}
      {(holes ?? []).map(([x, y, a], i) => {
        const r = Math.max(7, Math.sqrt(a / Math.PI) + 6);
        return <g key={i} className="rw-fadeup" style={{ animationDelay: `${i * 35}ms` }}>
          <circle cx={x} cy={y} r={r} fill="none" stroke={markerColor} strokeWidth={strokeW} />
          <line x1={x - r - 8} y1={y} x2={x - r + 2} y2={y} stroke={markerColor} strokeWidth={2} />
          <line x1={x + r - 2} y1={y} x2={x + r + 8} y2={y} stroke={markerColor} strokeWidth={2} />
        </g>;
      })}
    </svg>
  );
}
/** Animated "flip it" hint: a panel that rotates to show its holed face. Purely illustrative, driven by reason codes. */
export function FlipHint({ kind, stroke = '#111', size = 160 }: { kind: 'flip' | 'turn' | 'light'; stroke?: string; size?: number }) {
  if (kind === 'light') return <svg width={size} height={size * 0.6} viewBox="0 0 160 96"><circle cx="80" cy="44" r="22" fill="none" stroke={stroke} strokeWidth="5" />{[0, 45, 90, 135, 180, 225, 270, 315].map((a) => <line key={a} x1={80 + Math.cos(a * Math.PI / 180) * 30} y1={44 + Math.sin(a * Math.PI / 180) * 30} x2={80 + Math.cos(a * Math.PI / 180) * 40} y2={44 + Math.sin(a * Math.PI / 180) * 40} stroke={stroke} strokeWidth="5" strokeLinecap="round" />)}</svg>;
  return (
    <div style={{ width: size, height: size * 0.6 }} className="relative">
      <div className="rw-flip3d absolute inset-x-4 top-4 bottom-6" style={{ transformStyle: 'preserve-3d' }}>
        <div className="absolute inset-0 rounded border-4" style={{ borderColor: stroke, background: '#e9dcc4', backfaceVisibility: 'hidden' }} />
        <div className="absolute inset-0 rounded border-4 flex items-center justify-around px-2" style={{ borderColor: stroke, background: '#e9dcc4', transform: kind === 'flip' ? 'rotateX(180deg)' : 'rotateZ(180deg) rotateX(180deg)', backfaceVisibility: 'hidden' }}>
          {[0, 1, 2, 3, 4].map((i) => <span key={i} className="w-3 h-3 rounded-full" style={{ background: stroke }} />)}
        </div>
      </div>
      <svg className="absolute inset-0" viewBox="0 0 160 96"><path d="M150 20 C 172 50, 160 86, 120 92" fill="none" stroke={stroke} strokeWidth="5" strokeLinecap="round" className="rw-draw-path" /><path d="M128 82 L118 93 L132 97" fill="none" stroke={stroke} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </div>
  );
}
