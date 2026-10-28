import React, { useEffect, useRef, useState } from 'react';
import { RotateCcw, Box, Grid, ZoomIn, ZoomOut } from 'lucide-react';

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

interface Triangle3D {
  normal: Vec3;
  v1: Vec3;
  v2: Vec3;
  v3: Vec3;
}

interface StlViewer3DProps {
  stlUrl: string;
  holeCenterDistanceMm: number;
  outerWidthMm: number;
  thicknessMm: number;
  holeDiameterMm: number;
}

function parseAsciiOrBinaryStl(buffer: ArrayBuffer): Triangle3D[] {
  const decoder = new TextDecoder('utf-8');
  const headText = decoder.decode(buffer.slice(0, Math.min(512, buffer.byteLength)));
  const triangles: Triangle3D[] = [];

  if (headText.trimStart().startsWith('solid') && headText.includes('facet')) {
    const fullText = decoder.decode(buffer);
    const facetRegex =
      /facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)[\s\S]*?vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)[\s\S]*?vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)[\s\S]*?vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)/g;
    let m: RegExpExecArray | null;
    while ((m = facetRegex.exec(fullText)) !== null) {
      triangles.push({
        normal: { x: parseFloat(m[1]), y: parseFloat(m[2]), z: parseFloat(m[3]) },
        v1: { x: parseFloat(m[4]), y: parseFloat(m[5]), z: parseFloat(m[6]) },
        v2: { x: parseFloat(m[7]), y: parseFloat(m[8]), z: parseFloat(m[9]) },
        v3: { x: parseFloat(m[10]), y: parseFloat(m[11]), z: parseFloat(m[12]) },
      });
    }
    if (triangles.length > 0) return triangles;
  }

  if (buffer.byteLength >= 84) {
    const view = new DataView(buffer);
    const triCount = view.getUint32(80, true);
    let offset = 84;
    for (let i = 0; i < triCount && offset + 50 <= buffer.byteLength; i++) {
      const nx = view.getFloat32(offset, true);
      const ny = view.getFloat32(offset + 4, true);
      const nz = view.getFloat32(offset + 8, true);
      const v1x = view.getFloat32(offset + 12, true);
      const v1y = view.getFloat32(offset + 16, true);
      const v1z = view.getFloat32(offset + 20, true);
      const v2x = view.getFloat32(offset + 24, true);
      const v2y = view.getFloat32(offset + 28, true);
      const v2z = view.getFloat32(offset + 32, true);
      const v3x = view.getFloat32(offset + 36, true);
      const v3y = view.getFloat32(offset + 40, true);
      const v3z = view.getFloat32(offset + 44, true);
      triangles.push({
        normal: { x: nx, y: ny, z: nz },
        v1: { x: v1x, y: v1y, z: v1z },
        v2: { x: v2x, y: v2y, z: v2z },
        v3: { x: v3x, y: v3y, z: v3z },
      });
      offset += 50;
    }
  }
  return triangles;
}

export const StlViewer3D: React.FC<StlViewer3DProps> = ({
  stlUrl,
  holeCenterDistanceMm,
  outerWidthMm,
  thicknessMm,
  holeDiameterMm,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [triangles, setTriangles] = useState<Triangle3D[]>([]);
  const [rotX, setRotX] = useState<number>(-0.68);
  const [rotZ, setRotZ] = useState<number>(0.48);
  const [zoom, setZoom] = useState<number>(2.15);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [dragging, setDragging] = useState<boolean>(false);
  const lastMouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const [loadStatus, setLoadStatus] = useState<string>('Loading STL mesh...');

  useEffect(() => {
    let cancelled = false;
    fetch(stlUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.arrayBuffer();
      })
      .then((buf) => {
        if (cancelled) return;
        const parsed = parseAsciiOrBinaryStl(buf);
        setTriangles(parsed);
        setLoadStatus(`${parsed.length} facets loaded from flat_two_hole_link_v1.stl`);
      })
      .catch((err) => {
        if (!cancelled) setLoadStatus(`STL load error: ${String(err)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [stlUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Subtle CAD background gradient + grid
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, '#090d16');
    grad.addColorStop(1, '#0f172a');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    const cosZ = Math.cos(rotZ);
    const sinZ = Math.sin(rotZ);
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);

    const project = (p: Vec3): { sx: number; sy: number; depth: number; rx: number; ry: number; rz: number } => {
      // Rotate around Z then X
      const x1 = p.x * cosZ - p.y * sinZ;
      const y1 = p.x * sinZ + p.y * cosZ;
      const z1 = p.z;

      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      const fov = 420;
      const cameraDist = 260;
      const persp = (fov / (cameraDist - z2)) * zoom;
      return {
        sx: width * 0.5 + x2 * persp,
        sy: height * 0.52 - y2 * persp,
        depth: z2,
        rx: x2,
        ry: y2,
        rz: z2,
      };
    };

    // Draw reference mat_z0 grid on Z = -thickness/2
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.14)';
    ctx.lineWidth = 1;
    const gridZ = -thicknessMm * 0.5;
    for (let gx = -100; gx <= 100; gx += 20) {
      const a = project({ x: gx, y: -30, z: gridZ });
      const b = project({ x: gx, y: 30, z: gridZ });
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    }
    for (let gy = -30; gy <= 30; gy += 15) {
      const a = project({ x: -100, y: gy, z: gridZ });
      const b = project({ x: 100, y: gy, z: gridZ });
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    }

    if (triangles.length > 0) {
      const lightDir = { x: 0.38, y: -0.45, z: 0.81 };
      const projectedTris = triangles.map((tri) => {
        const p1 = project(tri.v1);
        const p2 = project(tri.v2);
        const p3 = project(tri.v3);
        const avgDepth = (p1.depth + p2.depth + p3.depth) / 3.0;

        // Rotate normal
        const nx1 = tri.normal.x * cosZ - tri.normal.y * sinZ;
        const ny1 = tri.normal.x * sinZ + tri.normal.y * cosZ;
        const nz1 = tri.normal.z;
        const nx2 = nx1;
        const ny2 = ny1 * cosX - nz1 * sinX;
        const nz2 = ny1 * sinX + nz1 * cosX;

        const dot = Math.max(0.18, Math.min(1.0, Math.abs(nx2 * lightDir.x + ny2 * lightDir.y + nz2 * lightDir.z)));
        return { p1, p2, p3, avgDepth, dot, topFace: tri.normal.z > 0.8 };
      });

      projectedTris.sort((a, b) => a.avgDepth - b.avgDepth);

      for (const item of projectedTris) {
        ctx.beginPath();
        ctx.moveTo(item.p1.sx, item.p1.sy);
        ctx.lineTo(item.p2.sx, item.p2.sy);
        ctx.lineTo(item.p3.sx, item.p3.sy);
        ctx.closePath();

        if (!wireframe) {
          // Industrial golden-amber CAD metallic shading
          const r = Math.round(45 + item.dot * (item.topFace ? 200 : 145));
          const g = Math.round(38 + item.dot * (item.topFace ? 158 : 112));
          const b = Math.round(25 + item.dot * (item.topFace ? 45 : 35));
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fill();
          ctx.strokeStyle = 'rgba(15, 23, 42, 0.35)';
          ctx.lineWidth = 0.45;
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.72)';
          ctx.lineWidth = 0.75;
          ctx.stroke();
        }
      }
    }

    // Draw hole center-to-center dimension line above top face
    const halfL = holeCenterDistanceMm * 0.5;
    const topZ = thicknessMm * 0.5 + 3.0;
    const leftHole = project({ x: -halfL, y: 0, z: topZ });
    const rightHole = project({ x: halfL, y: 0, z: topZ });

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(leftHole.sx, leftHole.sy);
    ctx.lineTo(rightHole.sx, rightHole.sy);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(leftHole.sx, leftHole.sy, 3.2, 0, Math.PI * 2);
    ctx.arc(rightHole.sx, rightHole.sy, 3.2, 0, Math.PI * 2);
    ctx.fill();

    const midX = (leftHole.sx + rightHole.sx) * 0.5;
    const midY = (leftHole.sy + rightHole.sy) * 0.5 - 8;
    ctx.fillStyle = '#020617';
    ctx.fillRect(midX - 58, midY - 11, 116, 15);
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`C-C: ${holeCenterDistanceMm.toFixed(2)} mm`, midX, midY);
  }, [triangles, rotX, rotZ, zoom, wireframe, holeCenterDistanceMm, outerWidthMm, thicknessMm, holeDiameterMm]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setDragging(true);
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragging) return;
    const dx = e.clientX - lastMouseRef.current.x;
    const dy = e.clientY - lastMouseRef.current.y;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    setRotZ((prev) => prev + dx * 0.012);
    setRotX((prev) => Math.max(-1.45, Math.min(0.2, prev + dy * 0.012)));
  };

  const handleMouseUp = () => setDragging(false);

  return (
    <div className="relative rounded-lg border border-slate-800 bg-slate-950 overflow-hidden">
      <div className="flex items-center justify-between px-2.5 py-1.5 bg-slate-900/90 border-b border-slate-800 text-[11px]">
        <div className="flex items-center gap-2">
          <Box className="w-3.5 h-3.5 text-amber-400" />
          <span className="font-mono-tech text-slate-200 font-semibold">
            3D STL MESH // flat_two_hole_link_v1.stl
          </span>
          <span className="text-emerald-400 font-mono-tech text-[10px]">
            ({triangles.length} tris · watertight)
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setWireframe((w) => !w)}
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono-tech border ${
              wireframe
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                : 'bg-slate-800 border-slate-700 text-slate-300'
            }`}
            title="Toggle Wireframe"
          >
            <Grid className="w-3 h-3 inline mr-1" />
            {wireframe ? 'Wire' : 'Solid'}
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(3.4, z + 0.25))}
            className="p-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:text-white"
            title="Zoom In"
          >
            <ZoomIn className="w-3 h-3" />
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(1.1, z - 0.25))}
            className="p-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:text-white"
            title="Zoom Out"
          >
            <ZoomOut className="w-3 h-3" />
          </button>
          <button
            type="button"
            onClick={() => {
              setRotX(-0.68);
              setRotZ(0.48);
              setZoom(2.15);
            }}
            className="p-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:text-white"
            title="Reset Isometric View"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={520}
        height={210}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="w-full h-[195px] cursor-grab active:cursor-grabbing block"
      />

      <div className="flex items-center justify-between px-2.5 py-1 bg-slate-900/75 border-t border-slate-800/80 text-[10px] font-mono-tech text-slate-400">
        <span>Drag to orbit · {loadStatus}</span>
        <span className="text-sky-300">
          BBox: {(holeCenterDistanceMm + outerWidthMm).toFixed(2)}×{outerWidthMm.toFixed(2)}×{thicknessMm.toFixed(2)} mm
        </span>
      </div>
    </div>
  );
};
