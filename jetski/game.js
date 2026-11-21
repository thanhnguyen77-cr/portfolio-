// Turbo Tide: a 3D jet ski race around a lagoon circuit.
// three.js r170 (vendored), Web Audio for all sound, no other assets.

import * as THREE from './vendor/three.module.min.js';
import { GameAudio } from './audio.js';

// ------------------------------------------------------------------ config
const qs = new URLSearchParams(location.search);
const TEST = qs.has('test');                         // autopilot for automated checks
const TIME_SCALE = TEST ? Math.max(1, Number(qs.get('speed')) || 1) : 1;
const LAPS = Math.max(1, Math.min(9, Number(qs.get('laps')) || 3));

const HALF_W = 22;          // half width of the course between buoy lines
const WALL = HALF_W + 8;    // hard limit outside the buoys
const MAX_SPEED = 36;       // m/s, shown as km/h
const BOOST_SPEED = 50;
const GRAVITY = 26;
const TRICK_TIME = 0.55;
const FINISH_BONUS = [5000, 3000, 2000, 1200, 800, 500];
const BEST_KEY = 'turbotide.best';

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const damp = (a, b, k, dt) => lerp(a, b, 1 - Math.exp(-k * dt));
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const ordinal = (n) => (n === 1 ? 'ST' : n === 2 ? 'ND' : n === 3 ? 'RD' : 'TH');
const fmtTime = (t) => {
  if (!isFinite(t)) return '--:--.-';
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
};
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(20261106);
const rand = (a, b) => a + (b - a) * rng();
const mrand = (a, b) => a + (b - a) * Math.random();

let best = 0;
try { best = Number(localStorage.getItem(BEST_KEY)) || 0; } catch (e) { /* private mode */ }

// ---------------------------------------------------------------- renderer
const canvas = $('game');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
} catch (e) {
  $('loading').classList.add('hidden');
  const el = $('error');
  el.textContent = 'This device could not start WebGL, which the game needs to draw in 3D.';
  el.classList.remove('hidden');
  throw e;
}
const coarse = matchMedia('(pointer: coarse)').matches;
let pixelRatio = Math.min(window.devicePixelRatio || 1, coarse ? 2 : 2);
renderer.setPixelRatio(pixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const FOG_COLOR = new THREE.Color(0xcfe4ee);
scene.fog = new THREE.Fog(FOG_COLOR, 380, 2900);
scene.background = FOG_COLOR;

const camera = new THREE.PerspectiveCamera(68, 1, 0.5, 6000);
const SUN_DIR = new THREE.Vector3(-0.45, 0.42, -0.78).normalize();

scene.add(new THREE.HemisphereLight(0xcfe9ff, 0x2d5a4d, 1.25));
const sun = new THREE.DirectionalLight(0xfff0d8, 2.6);
sun.position.copy(SUN_DIR).multiplyScalar(100);
scene.add(sun);

// ----------------------------------------------------------- canvas textures
function canvasTex(w, h, draw, repeat = false) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  return t;
}

// ------------------------------------------------------------------- track
const TRACK_SCALE = 0.55;
const CTRL = [
  [0, 300], [250, 300], [420, 220], [450, 50], [330, -80], [330, -220], [200, -320],
  [0, -300], [-150, -200], [-300, -280], [-430, -150], [-400, 60], [-300, 220], [-150, 310],
];

class Track {
  constructor(points) {
    const curve = new THREE.CatmullRomCurve3(
      points.map(([x, z]) => new THREE.Vector3(x * TRACK_SCALE, 0, z * TRACK_SCALE)), true, 'centripetal');
    curve.arcLengthDivisions = 4000;
    const N = this.N = 2400;
    const pts = curve.getSpacedPoints(N);
    this.L = curve.getLength();
    this.ds = this.L / N;
    this.px = new Float32Array(N); this.pz = new Float32Array(N);
    this.tx = new Float32Array(N); this.tz = new Float32Array(N);
    this.curv = new Float32Array(N);
    for (let i = 0; i < N; i++) { this.px[i] = pts[i].x; this.pz[i] = pts[i].z; }
    for (let i = 0; i < N; i++) {
      const a = (i - 1 + N) % N, b = (i + 1) % N;
      let dx = this.px[b] - this.px[a], dz = this.pz[b] - this.pz[a];
      const l = Math.hypot(dx, dz) || 1;
      this.tx[i] = dx / l; this.tz[i] = dz / l;
    }
    // signed curvature (+ = right turn), measured over ~10 m
    const k = 7;
    for (let i = 0; i < N; i++) {
      const a = (i - k + N) % N, b = (i + k) % N;
      const cross = this.tx[a] * this.tz[b] - this.tz[a] * this.tx[b];
      this.curv[i] = Math.asin(clamp(cross, -1, 1)) / (2 * k * this.ds);
    }
  }
  wrapS(s) { return ((s % this.L) + this.L) % this.L; }
  // position + tangent at distance s, offset sideways by lat (+ = right)
  at(s, lat = 0, out = {}) {
    s = this.wrapS(s);
    const f = s / this.ds;
    const i = Math.floor(f) % this.N, j = (i + 1) % this.N, t = f - Math.floor(f);
    const tx = lerp(this.tx[i], this.tx[j], t), tz = lerp(this.tz[i], this.tz[j], t);
    out.tx = tx; out.tz = tz;
    out.x = lerp(this.px[i], this.px[j], t) - tz * lat;
    out.z = lerp(this.pz[i], this.pz[j], t) + tx * lat;
    out.heading = Math.atan2(tx, tz);
    return out;
  }
  nearest(x, z, hint = -1, win = 50) {
    let best = 0, bd = Infinity;
    if (hint < 0) {
      for (let i = 0; i < this.N; i++) {
        const dx = x - this.px[i], dz = z - this.pz[i], d = dx * dx + dz * dz;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    }
    for (let k = -win; k <= win; k++) {
      const i = (hint + k + this.N) % this.N;
      const dx = x - this.px[i], dz = z - this.pz[i], d = dx * dx + dz * dz;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  // smallest distance from a point to the centreline (coarse)
  distTo(x, z) {
    let bd = Infinity;
    for (let i = 0; i < this.N; i += 4) {
      const dx = x - this.px[i], dz = z - this.pz[i], d = dx * dx + dz * dz;
      if (d < bd) bd = d;
    }
    return Math.sqrt(bd);
  }
  // the lowest-curvature spot within +-range of s
  straightNear(s, range = 60) {
    let bestS = s, bc = Infinity;
    for (let d = -range; d <= range; d += 2) {
      const ss = this.wrapS(s + d);
      let c = 0;
      for (let e = 0; e <= 40; e += 4) c += Math.abs(this.curv[Math.floor(this.wrapS(ss + e) / this.ds) % this.N]);
      if (c < bc) { bc = c; bestS = ss; }
    }
    return bestS;
  }
}
const track = new Track(CTRL);
const L = track.L;
const tmp = {};

// ------------------------------------------------------------------- water
function makeWaterNormals() {
  const S = 256;
  const h = new Float32Array(S * S);
  const R = mulberry32(7);
  const octaves = [[8, 1.0], [16, 0.55], [32, 0.3], [64, 0.15]];
  for (const [P, amp] of octaves) {
    const lat = new Float32Array(P * P);
    for (let i = 0; i < lat.length; i++) lat[i] = R();
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const fx = (x / S) * P, fy = (y / S) * P;
        const ix = Math.floor(fx), iy = Math.floor(fy);
        let tx = fx - ix, ty = fy - iy;
        tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
        const a = lat[(iy % P) * P + (ix % P)], b = lat[(iy % P) * P + ((ix + 1) % P)];
        const c = lat[((iy + 1) % P) * P + (ix % P)], d = lat[((iy + 1) % P) * P + ((ix + 1) % P)];
        h[y * S + x] += amp * lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
      }
    }
  }
  // a couple of periodic swells for directional ripples
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S, v = y / S;
      h[y * S + x] += 0.35 * Math.sin(2 * Math.PI * (3 * u + 5 * v)) + 0.2 * Math.sin(2 * Math.PI * (7 * u - 4 * v));
    }
  }
  let mn = Infinity, mx = -Infinity;
  for (const v of h) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const data = new Uint8Array(S * S * 4);
  const str = 9.0;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const hl = h[y * S + ((x - 1 + S) % S)], hr = h[y * S + ((x + 1) % S)];
      const hd = h[((y - 1 + S) % S) * S + x], hu = h[((y + 1) % S) * S + x];
      let nx = (hl - hr) * str / (mx - mn), nz = (hd - hu) * str / (mx - mn), ny = 1;
      const l = Math.hypot(nx, ny, nz);
      nx /= l; ny /= l; nz /= l;
      data[i * 4] = (nx * 0.5 + 0.5) * 255;
      data[i * 4 + 1] = (nz * 0.5 + 0.5) * 255;
      data[i * 4 + 2] = (ny * 0.5 + 0.5) * 255;
      data[i * 4 + 3] = ((h[i] - mn) / (mx - mn)) * 255;
    }
  }
  const t = new THREE.DataTexture(data, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

const waterUniforms = {
  uTime: { value: 0 },
  uNormals: { value: makeWaterNormals() },
  uSunDir: { value: SUN_DIR },
  uSkyTop: { value: new THREE.Color(0x3a8bd8) },
  uSkyHor: { value: new THREE.Color(0xd6ecf6) },
  uDeep: { value: new THREE.Color(0x0b4f74) },
  uShallow: { value: new THREE.Color(0x2fb8c4) },
  uFogColor: { value: FOG_COLOR },
  uFogNear: { value: scene.fog.near },
  uFogFar: { value: scene.fog.far },
};
const water = new THREE.Mesh(
  new THREE.PlaneGeometry(9000, 9000).rotateX(-Math.PI / 2),
  new THREE.ShaderMaterial({
    uniforms: waterUniforms,
    vertexShader: /* glsl */`
      varying vec3 vW;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */`
      uniform float uTime;
      uniform sampler2D uNormals;
      uniform vec3 uSunDir, uSkyTop, uSkyHor, uDeep, uShallow, uFogColor;
      uniform float uFogNear, uFogFar;
      varying vec3 vW;
      void main() {
        vec2 p = vW.xz;
        float dist = length(cameraPosition - vW);
        vec4 a = texture2D(uNormals, p * 0.031 + vec2(uTime * 0.018, uTime * 0.011));
        vec4 b = texture2D(uNormals, p * 0.012 - vec2(uTime * 0.009, -uTime * 0.014));
        vec4 c = texture2D(uNormals, p * 0.083 + vec2(-uTime * 0.03, uTime * 0.02));
        vec2 d = (a.xy - 0.5) * 1.0 + (b.xy - 0.5) * 0.9 + (c.xy - 0.5) * 0.5;
        float strength = mix(1.25, 0.25, smoothstep(40.0, 700.0, dist));
        vec3 n = normalize(vec3(d.x * strength, 1.0, d.y * strength));
        vec3 V = normalize(cameraPosition - vW);
        float ndv = max(dot(n, V), 0.0);
        float fres = 0.03 + 0.97 * pow(1.0 - ndv, 5.0);
        vec3 R = reflect(-V, n);
        R.y = abs(R.y);
        vec3 sky = mix(uSkyHor, uSkyTop, pow(clamp(R.y, 0.0, 1.0), 0.55));
        float sd = max(dot(R, uSunDir), 0.0);
        vec3 sunSpec = vec3(1.0, 0.94, 0.82) * (pow(sd, 420.0) * 7.0 + pow(sd, 60.0) * 0.35);
        float patchV = texture2D(uNormals, p * 0.0017).a;
        vec3 body = mix(uDeep, uShallow, 0.25 + 0.55 * patchV);
        body *= 0.75 + 0.5 * max(dot(n, uSunDir), 0.0);
        vec3 col = mix(body, sky, clamp(fres * 0.9, 0.0, 1.0)) + sunSpec;
        float caps = smoothstep(0.80, 0.93, a.a * 0.6 + b.a * 0.5);
        col = mix(col, vec3(0.92, 0.96, 1.0), caps * 0.35 * (1.0 - smoothstep(60.0, 380.0, dist)));
        float f = smoothstep(uFogNear, uFogFar, dist);
        col = mix(col, uFogColor, f);
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  }),
);
water.renderOrder = -1;
scene.add(water);

// --------------------------------------------------------------------- sky
const sky = new THREE.Mesh(
  new THREE.SphereGeometry(4500, 32, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uTop: { value: new THREE.Color(0x2f7fd0) },
      uHor: { value: FOG_COLOR },
      uSunDir: { value: SUN_DIR },
    },
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() { vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`
      uniform vec3 uTop, uHor, uSunDir;
      varying vec3 vDir;
      void main() {
        vec3 d = normalize(vDir);
        vec3 col = mix(uHor, uTop, pow(clamp(d.y, 0.0, 1.0), 0.48));
        float sd = max(dot(d, uSunDir), 0.0);
        col += vec3(1.0, 0.93, 0.8) * (pow(sd, 1200.0) * 10.0 + pow(sd, 10.0) * 0.22);
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  }),
);
sky.renderOrder = -2;
scene.add(sky);

// clouds
{
  const tex = canvasTex(256, 128, (g, w, h) => {
    const R = mulberry32(3);
    for (let i = 0; i < 26; i++) {
      const x = w * (0.15 + 0.7 * R()), y = h * (0.45 + 0.25 * R()), r = 18 + 30 * R();
      const gr = g.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, 'rgba(255,255,255,0.85)');
      gr.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = gr;
      g.fillRect(0, 0, w, h);
    }
  });
  for (let i = 0; i < 16; i++) {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, fog: false, depthWrite: false, transparent: true, opacity: 0.9 }));
    const a = rand(0, Math.PI * 2), d = rand(1700, 2900);
    sp.position.set(Math.cos(a) * d, rand(260, 620), Math.sin(a) * d);
    const s = rand(500, 900);
    sp.scale.set(s, s * 0.45, 1);
    sp.renderOrder = -1;
    scene.add(sp);
  }
}

// ---------------------------------------------------------- geometry utils
function mergeGeometries(list) {
  // concatenates non-indexed geometries with position/normal/uv(/color)
  const geos = list.map((g) => (g.index ? g.toNonIndexed() : g));
  const hasColor = geos.every((g) => g.attributes.color);
  let count = 0;
  for (const g of geos) count += g.attributes.position.count;
  const pos = new Float32Array(count * 3), nor = new Float32Array(count * 3), uv = new Float32Array(count * 2);
  const col = hasColor ? new Float32Array(count * 3) : null;
  let o = 0;
  for (const g of geos) {
    const n = g.attributes.position.count;
    pos.set(g.attributes.position.array, o * 3);
    if (g.attributes.normal) nor.set(g.attributes.normal.array, o * 3);
    if (g.attributes.uv) uv.set(g.attributes.uv.array, o * 2);
    if (col) col.set(g.attributes.color.array, o * 3);
    o += n;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  out.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  if (col) out.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return out;
}
function paint(geo, color) {
  const c = new THREE.Color(color);
  const g = geo.index ? geo.toNonIndexed() : geo;
  const n = g.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b; }
  g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
  return g;
}

// ---------------------------------------------------------------- scenery
const lambert = (o) => new THREE.MeshLambertMaterial(o);

// islands with palms
const islands = [];
{
  const geos = [];
  const palmSpots = [];
  const tryPlace = (x, z, R) => {
    if (track.distTo(x, z) < HALF_W + R + 28) return false;
    for (const is of islands) if (Math.hypot(is.x - x, is.z - z) < is.R + R + 20) return false;
    islands.push({ x, z, R });
    return true;
  };
  // a ring of islands around and inside the course; the north side is the city
  for (let tries = 0; tries < 400 && islands.length < 22; tries++) {
    const a = rand(0, Math.PI * 2);
    const d = rand(60, 620);
    const x = Math.cos(a) * d, z = Math.sin(a) * d;
    if (z < -300 && Math.abs(x) < 520) continue; // keep the city waterfront clear
    tryPlace(x, z, rand(28, 85));
  }
  for (const is of islands) {
    const rings = 9, segs = 28;
    const H = is.R * rand(0.35, 0.75);
    const g = new THREE.BufferGeometry();
    const pos = [], col = [];
    const sand = new THREE.Color(0xe9d7a6), grass = new THREE.Color(0x4f9a3c), dark = new THREE.Color(0x2f6b2a), rock = new THREE.Color(0x8a8070);
    const ph = rand(0, 10);
    const height = (r, a) => {
      const rr = r / is.R;
      const n = 1 + 0.18 * Math.sin(a * 3 + ph) + 0.12 * Math.sin(a * 7 + ph * 2);
      if (rr >= 1) return -3;
      const base = Math.pow(1 - rr * rr, 1.4) * H * n;
      return rr > 0.82 ? lerp(1.2, -3, (rr - 0.82) / 0.18) : Math.max(1.2, base);
    };
    const vtx = (r, a) => {
      const y = height(r, a);
      pos.push(is.x + Math.cos(a) * r, y, is.z + Math.sin(a) * r);
      const c = y < 2.2 ? sand : y > H * 0.75 ? rock : (Math.sin(a * 5 + r) > 0 ? grass : dark);
      col.push(c.r, c.g, c.b);
    };
    for (let ri = 0; ri < rings; ri++) {
      const r0 = (ri / rings) * is.R * 1.08, r1 = ((ri + 1) / rings) * is.R * 1.08;
      for (let si = 0; si < segs; si++) {
        const a0 = (si / segs) * Math.PI * 2, a1 = ((si + 1) / segs) * Math.PI * 2;
        vtx(r0, a0); vtx(r1, a1); vtx(r1, a0);
        if (ri > 0) { vtx(r0, a0); vtx(r0, a1); vtx(r1, a1); }
      }
    }
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(pos.length / 3 * 2), 2));
    g.computeVertexNormals();
    geos.push(g);
    const np = Math.floor(rand(3, 9));
    for (let i = 0; i < np; i++) {
      const a = rand(0, Math.PI * 2), r = is.R * rand(0.45, 0.75);
      palmSpots.push([is.x + Math.cos(a) * r, height(r, a), is.z + Math.sin(a) * r]);
    }
  }
  const islandMesh = new THREE.Mesh(mergeGeometries(geos), lambert({ vertexColors: true, flatShading: true }));
  scene.add(islandMesh);

  // palms: instanced trunk + crown
  const trunkGeo = new THREE.CylinderGeometry(0.35, 0.6, 9, 6).translate(0, 4.5, 0);
  const leaves = [];
  for (let i = 0; i < 7; i++) {
    const leaf = new THREE.ConeGeometry(0.9, 6, 4).rotateZ(-Math.PI / 2 - 0.35).translate(3, 0, 0).scale(1, 0.35, 1);
    leaf.rotateY((i / 7) * Math.PI * 2).translate(0, 9, 0);
    leaves.push(leaf);
  }
  const crownGeo = mergeGeometries(leaves);
  const trunks = new THREE.InstancedMesh(trunkGeo, lambert({ color: 0x8b6b45, flatShading: true }), palmSpots.length);
  const crowns = new THREE.InstancedMesh(crownGeo, lambert({ color: 0x2e8b3a, flatShading: true, side: THREE.DoubleSide }), palmSpots.length);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), sc = new THREE.Vector3();
  palmSpots.forEach(([x, y, z], i) => {
    e.set(rand(-0.15, 0.15), rand(0, 6.28), rand(-0.15, 0.15));
    q.setFromEuler(e);
    const s = rand(0.8, 1.35);
    sc.set(s, s, s);
    m.compose(new THREE.Vector3(x, y - 0.5, z), q, sc);
    trunks.setMatrixAt(i, m);
    crowns.setMatrixAt(i, m);
  });
  scene.add(trunks, crowns);
}

// city waterfront to the north (negative z) and distant mountains
{
  const winTex = (base, glass) => canvasTex(64, 128, (g, w, h) => {
    g.fillStyle = base; g.fillRect(0, 0, w, h);
    const R = mulberry32(base.length * 31);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 4; x++) {
        const l = 0.55 + R() * 0.45;
        g.fillStyle = glass;
        g.globalAlpha = l;
        g.fillRect(x * 16 + 3, y * 8 + 2, 10, 5);
      }
    }
    g.globalAlpha = 1;
  }, true);
  const mats = [
    lambert({ map: winTex('#c9d3dc', '#5b7c99') }),
    lambert({ map: winTex('#e9e4da', '#6f8fa8') }),
    lambert({ map: winTex('#8fa3b4', '#2f4a63') }),
  ];
  const groups = [[], [], []];
  const zFront = -Math.max(...CTRL.map((p) => -p[1])) * TRACK_SCALE - 150;
  for (let row = 0; row < 4; row++) {
    for (let x = -760; x < 760; x += rand(18, 34)) {
      const w = rand(14, 30), d = rand(14, 30);
      const h = rand(18, 50) + (row * 30) + Math.max(0, 140 - Math.abs(x) * 0.25) * rng();
      const g = new THREE.BoxGeometry(w, h, d);
      const uv = g.attributes.uv;
      for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * w / 16, uv.getY(i) * h / 32);
      g.translate(x + rand(-4, 4), h / 2 + 1.5, zFront - row * 45 - rand(0, 20));
      groups[Math.floor(rng() * 3)].push(g);
    }
  }
  groups.forEach((list, i) => scene.add(new THREE.Mesh(mergeGeometries(list), mats[i])));
  // quay and seawall
  const quay = new THREE.Mesh(new THREE.BoxGeometry(1700, 4, 260), lambert({ color: 0xb9b3a4 }));
  quay.position.set(0, -0.5, zFront - 95);
  scene.add(quay);

  // mountains
  const mgeos = [];
  for (let i = 0; i < 26; i++) {
    const a = rand(0, Math.PI * 2), d = rand(1900, 2600);
    const r = rand(250, 520), hh = rand(160, 420);
    const g = paint(new THREE.ConeGeometry(r, hh, 7, 1).translate(Math.cos(a) * d, hh / 2 - 10, Math.sin(a) * d),
      rng() > 0.5 ? 0x5f8a7c : 0x6d8f86);
    mgeos.push(g);
  }
  scene.add(new THREE.Mesh(mergeGeometries(mgeos), lambert({ vertexColors: true, flatShading: true })));
}

// --------------------------------------------------------- course features
// buoys
const buoys = [];
let buoyMesh;
{
  const top = new THREE.CylinderGeometry(0.08, 0.08, 1.1, 5).translate(0, 1.1, 0);
  const ball = new THREE.SphereGeometry(0.75, 10, 8).scale(1, 0.85, 1).translate(0, 0.35, 0);
  const geo = mergeGeometries([ball, top]);
  const step = 13;
  const count = Math.floor(L / step);
  buoyMesh = new THREE.InstancedMesh(geo, lambert({ color: 0xffffff }), count * 2);
  const c1 = new THREE.Color(0xff5a1f), c2 = new THREE.Color(0xf4f4f4);
  for (let i = 0; i < count; i++) {
    for (const side of [-1, 1]) {
      track.at(i * step, side * HALF_W, tmp);
      const k = buoys.length;
      buoys.push({ x: tmp.x, z: tmp.z, ph: rng() * 6.28 });
      buoyMesh.setColorAt(k, i % 2 ? c1 : c2);
    }
  }
  scene.add(buoyMesh);
}

// chevron boards on the outside of the tight turns
{
  const chev = (flip) => canvasTex(256, 64, (g, w, h) => {
    g.fillStyle = '#ffffff'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#e8261e';
    for (let i = -1; i < 6; i++) {
      const x0 = i * 52 + 10;
      g.beginPath();
      if (!flip) { g.moveTo(x0, 0); g.lineTo(x0 + 22, 0); g.lineTo(x0 + 44, h / 2); g.lineTo(x0 + 22, h); g.lineTo(x0, h); g.lineTo(x0 + 22, h / 2); }
      else { g.moveTo(x0 + 44, 0); g.lineTo(x0 + 22, 0); g.lineTo(x0, h / 2); g.lineTo(x0 + 22, h); g.lineTo(x0 + 44, h); g.lineTo(x0 + 22, h / 2); }
      g.fill();
    }
    g.fillStyle = '#1d4ed8'; g.fillRect(0, h - 6, w, 6); g.fillRect(0, 0, w, 6);
  });
  const geo = new THREE.PlaneGeometry(7, 1.9).translate(0, 1.15, 0);
  const spots = [[], []];
  const step = 7.5;
  for (let s = 0; s < L; s += step) {
    const i = Math.floor(s / track.ds) % track.N;
    const k = track.curv[i];
    if (Math.abs(k) < 1 / 75) continue;
    const side = k > 0 ? -1 : 1; // outside of the turn
    track.at(s, side * (HALF_W + 3.5), tmp);
    // board faces the course
    const nx = tmp.tz * side, nz = -tmp.tx * side; // points from board toward centreline
    const rot = Math.atan2(nx, nz);
    const localX = { x: Math.cos(rot), z: -Math.sin(rot) };
    const along = localX.x * tmp.tx + localX.z * tmp.tz;
    spots[along > 0 ? 0 : 1].push([tmp.x, tmp.z, rot]);
  }
  const m = new THREE.Matrix4(), q = new THREE.Quaternion();
  [chev(false), chev(true)].forEach((tex, v) => {
    const list = spots[v];
    if (!list.length) return;
    const mesh = new THREE.InstancedMesh(geo, lambert({ map: tex, side: THREE.DoubleSide }), list.length);
    list.forEach(([x, z, r], i) => {
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r);
      m.compose(new THREE.Vector3(x, 0, z), q, new THREE.Vector3(1, 1, 1));
      mesh.setMatrixAt(i, m);
    });
    scene.add(mesh);
  });
}

// start / finish gantry
{
  const g = new THREE.Group();
  track.at(0, 0, tmp);
  g.position.set(tmp.x, 0, tmp.z);
  g.rotation.y = tmp.heading;
  const span = (HALF_W + 11) * 2;
  const pillarMat = lambert({ color: 0x1e2a38 });
  for (const s of [-1, 1]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(2.2, 14, 2.2), pillarMat);
    p.position.set(s * span / 2, 6, 0);
    g.add(p);
    const f = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.8, 2, 8), lambert({ color: 0xff7a1a }));
    f.position.set(s * span / 2, 0, 0);
    g.add(f);
  }
  const bannerTex = canvasTex(1024, 128, (c, w, h) => {
    const cell = 16;
    for (let y = 0; y < h; y += cell) for (let x = 0; x < w; x += cell) {
      c.fillStyle = ((x + y) / cell) % 2 ? '#111' : '#fff'; c.fillRect(x, y, cell, cell);
    }
    c.fillStyle = '#0b2233'; c.fillRect(0, 22, w, h - 44);
    c.font = 'italic 900 64px Arial Black, Arial, sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillStyle = '#ff7a1a'; c.fillText('TURBO', w / 2 - 130, h / 2 + 3);
    c.fillStyle = '#19c3d6'; c.fillText('TIDE', w / 2 + 150, h / 2 + 3);
  });
  const banner = new THREE.Mesh(new THREE.BoxGeometry(span + 2.2, 4.2, 1.2), [
    pillarMat, pillarMat, pillarMat, pillarMat, lambert({ map: bannerTex }), lambert({ map: bannerTex }),
  ]);
  banner.position.y = 12.5;
  g.add(banner);
  // floating checkered start line
  const lineTex = canvasTex(256, 16, (c, w, h) => {
    for (let x = 0; x < w; x += 8) for (let y = 0; y < h; y += 8) { c.fillStyle = ((x + y) / 8) % 2 ? '#111' : '#fff'; c.fillRect(x, y, 8, 8); }
  });
  const line = new THREE.Mesh(new THREE.PlaneGeometry(HALF_W * 2, 1.6).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ map: lineTex, transparent: true, opacity: 0.85, depthWrite: false }));
  line.position.y = 0.06;
  line.renderOrder = 2;
  g.add(line);
  scene.add(g);
}

// ramps
const RAMP_W = 9, RAMP_L = 11, RAMP_H = 2.6, RAMP_SINK = 0.5;
const ramps = [];
{
  const stripes = canvasTex(128, 128, (c, w, h) => {
    c.fillStyle = '#ffd23f'; c.fillRect(0, 0, w, h);
    c.fillStyle = '#16202a';
    for (let i = -4; i < 8; i++) { c.beginPath(); c.moveTo(i * 32, 0); c.lineTo(i * 32 + 16, 0); c.lineTo(i * 32 + 16 + h, h); c.lineTo(i * 32 + h, h); c.fill(); }
  });
  const w = RAMP_W / 2, Hc = RAMP_H, b = -RAMP_SINK, l = RAMP_L;
  const A = [-w, b, 0], B = [w, b, 0], C = [-w, Hc, l], D = [w, Hc, l], E = [-w, b, l], F = [w, b, l];
  const tris = [A, C, B, B, C, D, C, E, D, D, E, F, A, E, C, B, D, F];
  const uvs = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(tris.flat(), 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute([...uvs, ...new Array(24).fill(0)], 2));
  geo.computeVertexNormals();
  geo.addGroup(0, 6, 0);
  geo.addGroup(6, 12, 1);
  const mats = [lambert({ map: stripes }), lambert({ color: 0xf2f5f8 })];
  for (const [frac, lat] of [[0.13, -6], [0.47, 7], [0.79, 0]]) {
    const s = track.straightNear(frac * L, 70);
    track.at(s, lat, tmp);
    const mesh = new THREE.Mesh(geo, mats);
    mesh.position.set(tmp.x, 0, tmp.z);
    mesh.rotation.y = tmp.heading;
    scene.add(mesh);
    ramps.push({ s, lat, mesh });
  }
}

// boost pads
const pads = [];
let padTex;
{
  padTex = canvasTex(64, 128, (c, w, h) => {
    c.clearRect(0, 0, w, h);
    c.strokeStyle = '#7cf7ff'; c.lineWidth = 9; c.lineJoin = 'round';
    for (let k = 0; k < 2; k++) {
      const y = 22 + k * 64;
      c.beginPath(); c.moveTo(10, y + 30); c.lineTo(32, y); c.lineTo(54, y + 30); c.stroke();
    }
  }, true);
  padTex.repeat.set(1, 2);
  const mat = new THREE.MeshBasicMaterial({ map: padTex, transparent: true, depthWrite: false, color: 0xffffff, blending: THREE.AdditiveBlending });
  const baseMat = new THREE.MeshBasicMaterial({ color: 0x0b6e99, transparent: true, opacity: 0.55, depthWrite: false });
  for (const [frac, lat] of [[0.05, 8], [0.3, -8], [0.6, 0], [0.88, -6]]) {
    const s = frac * L;
    track.at(s, lat, tmp);
    const g = new THREE.Group();
    const base = new THREE.Mesh(new THREE.PlaneGeometry(6, 10).rotateX(-Math.PI / 2), baseMat);
    const arrows = new THREE.Mesh(new THREE.PlaneGeometry(5, 9).rotateX(-Math.PI / 2).rotateY(Math.PI), mat);
    base.position.y = 0.07; arrows.position.y = 0.1;
    base.renderOrder = 2; arrows.renderOrder = 3;
    g.add(base, arrows);
    g.position.set(tmp.x, 0, tmp.z);
    g.rotation.y = tmp.heading;
    scene.add(g);
    pads.push({ s, lat });
  }
}

// coins
const coins = [];
let coinMesh;
{
  const add = (s, lat, y = 1.4) => { track.at(s, lat, tmp); coins.push({ x: tmp.x, y, z: tmp.z, s: track.wrapS(s), taken: false }); };
  // weaving lines
  for (const [frac, lat0, amp] of [[0.2, 0, 10], [0.36, -10, 6], [0.55, 8, 8], [0.67, -4, 12], [0.93, 4, 9]]) {
    for (let k = 0; k < 8; k++) add(frac * L + k * 7, lat0 + Math.sin(k * 0.7) * amp);
  }
  // arcs over each ramp
  for (const r of ramps) {
    const v = 34, vy = v * (RAMP_H + RAMP_SINK) / RAMP_L * 1.25 + 4;
    for (let k = 1; k <= 6; k++) {
      const t = (k * 6) / v;
      add(r.s + RAMP_L + k * 6, r.lat, RAMP_H + vy * t - 0.5 * GRAVITY * t * t + 0.8);
    }
  }
  const geo = new THREE.CylinderGeometry(1.0, 1.0, 0.22, 20).rotateX(Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffc21a, emissive: 0x7a4b00, metalness: 0.55, roughness: 0.3 });
  coinMesh = new THREE.InstancedMesh(geo, mat, coins.length);
  coinMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(coinMesh);
}

// spectator boats outside the course
const boats = [];
{
  const hull = new THREE.BoxGeometry(5, 2, 16, 1, 1, 4);
  const p = hull.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const z = p.getZ(i), t = (z + 8) / 16;
    if (t > 0.6) p.setX(i, p.getX(i) * (1 - (t - 0.6) / 0.4 * 0.85));
    if (p.getY(i) < 0) p.setX(i, p.getX(i) * 0.7);
  }
  hull.computeVertexNormals();
  const cabin = new THREE.BoxGeometry(3.8, 2.2, 6).translate(0, 2, -2);
  const glass = new THREE.BoxGeometry(3.9, 0.8, 5).translate(0, 2.4, -1.6);
  const hullMat = lambert({ color: 0xffffff }), cabMat = lambert({ color: 0xe9edf2 }), glassMat = lambert({ color: 0x22384f });
  for (let i = 0; i < 12; i++) {
    const s = (i / 12 + rand(-0.02, 0.02)) * L;
    const side = rng() > 0.5 ? 1 : -1;
    track.at(s, side * rand(HALF_W + 30, HALF_W + 55), tmp);
    const g = new THREE.Group();
    g.add(new THREE.Mesh(hull, hullMat), new THREE.Mesh(cabin, cabMat), new THREE.Mesh(glass, glassMat));
    g.position.set(tmp.x, 0.4, tmp.z);
    g.rotation.y = tmp.heading + rand(-0.6, 0.6) + (rng() > 0.5 ? Math.PI : 0);
    scene.add(g);
    boats.push({ g, ph: rng() * 6 });
  }
}

// -------------------------------------------------------------- effects
const spray = (() => {
  const MAX = 1400;
  const pos = new Float32Array(MAX * 3), vel = new Float32Array(MAX * 3);
  const life = new Float32Array(MAX), maxLife = new Float32Array(MAX), size = new Float32Array(MAX), alpha = new Float32Array(MAX);
  const sizeAttr = new Float32Array(MAX);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('aAlpha', new THREE.BufferAttribute(alpha, 1).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('aSize', new THREE.BufferAttribute(sizeAttr, 1).setUsage(THREE.DynamicDrawUsage));
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { uScale: { value: 400 } },
    vertexShader: /* glsl */`
      attribute float aAlpha; attribute float aSize; uniform float uScale; varying float vA;
      void main() {
        vA = aAlpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * uScale / max(1.0, -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      varying float vA;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = max(0.0, 1.0 - dot(c, c) * 4.0);
        float a = vA * d * d;
        if (a < 0.01) discard;
        gl_FragColor = vec4(0.96, 0.99, 1.0, a);
      }`,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  points.renderOrder = 4;
  scene.add(points);
  let next = 0;
  return {
    mat,
    emit(x, y, z, vx, vy, vz, l, s) {
      const i = next; next = (next + 1) % MAX;
      pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
      vel[i * 3] = vx; vel[i * 3 + 1] = vy; vel[i * 3 + 2] = vz;
      life[i] = maxLife[i] = l; size[i] = s;
    },
    update(dt) {
      for (let i = 0; i < MAX; i++) {
        if (life[i] <= 0) { alpha[i] = 0; sizeAttr[i] = 0; continue; }
        life[i] -= dt;
        vel[i * 3 + 1] -= 16 * dt;
        pos[i * 3] += vel[i * 3] * dt; pos[i * 3 + 1] += vel[i * 3 + 1] * dt; pos[i * 3 + 2] += vel[i * 3 + 2] * dt;
        if (pos[i * 3 + 1] < 0) { pos[i * 3 + 1] = 0; vel[i * 3 + 1] = 0; life[i] = Math.min(life[i], 0.12); }
        const k = Math.max(0, life[i] / maxLife[i]);
        alpha[i] = k * 0.75;
        sizeAttr[i] = size[i] * (1.6 - k * 0.8);
      }
      geo.attributes.position.needsUpdate = true;
      geo.attributes.aAlpha.needsUpdate = true;
      geo.attributes.aSize.needsUpdate = true;
    },
    clear() { life.fill(0); },
  };
})();

const wakeMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  vertexShader: /* glsl */`
    attribute float aAlpha; varying float vA; varying vec2 vUv;
    void main() { vA = aAlpha; vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    varying float vA; varying vec2 vUv;
    void main() {
      float e = abs(vUv.x * 2.0 - 1.0);
      float a = vA * (0.35 + 0.65 * smoothstep(0.35, 0.95, e)) * (1.0 - smoothstep(0.9, 1.0, e));
      float n = fract(sin(dot(floor(vec2(vUv.x * 10.0, vUv.y * 40.0)), vec2(12.9898, 78.233))) * 43758.5453);
      a *= 0.75 + 0.25 * n;
      gl_FragColor = vec4(0.95, 0.98, 1.0, a);
    }`,
});

class Wake {
  constructor() {
    this.n = 40;
    this.pts = [];
    const n = this.n;
    this.pos = new Float32Array(n * 2 * 3);
    this.alpha = new Float32Array(n * 2);
    const uv = new Float32Array(n * 2 * 2), idx = [];
    for (let i = 0; i < n; i++) {
      uv[i * 4] = 0; uv[i * 4 + 1] = i / (n - 1); uv[i * 4 + 2] = 1; uv[i * 4 + 3] = i / (n - 1);
      if (i < n - 1) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setIndex(idx);
    this.geo = g;
    this.mesh = new THREE.Mesh(g, wakeMat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    scene.add(this.mesh);
  }
  reset() { this.pts.length = 0; }
  update(dt, r) {
    const LIFE = 2.0;
    for (const p of this.pts) p.age += dt;
    while (this.pts.length && this.pts[this.pts.length - 1].age > LIFE) this.pts.pop();
    const fx = Math.sin(r.heading), fz = Math.cos(r.heading);
    const rx = r.x - fx * 1.9, rz = r.z - fz * 1.9;
    const on = !r.airborne && r.speed > 3;
    const last = this.pts[0];
    if (on && (!last || Math.hypot(last.x - rx, last.z - rz) > 1.6)) {
      this.pts.unshift({ x: rx, z: rz, dx: fx, dz: fz, age: 0, str: clamp(r.speed / MAX_SPEED, 0.2, 1.2) });
      if (this.pts.length > this.n - 1) this.pts.length = this.n - 1;
    }
    const P = this.pos, A = this.alpha;
    let lx = rx, lz = rz, ldx = fx, ldz = fz, la = on ? 0.6 : 0;
    for (let i = 0; i < this.n; i++) {
      let x, z, dx, dz, w, a;
      if (i === 0) { x = rx; z = rz; dx = fx; dz = fz; w = 0.55; a = la; }
      else if (i - 1 < this.pts.length) {
        const p = this.pts[i - 1];
        x = p.x; z = p.z; dx = p.dx; dz = p.dz;
        w = 0.6 + p.age * 2.8;
        a = Math.pow(1 - p.age / LIFE, 1.6) * p.str * 0.6;
      } else { x = lx; z = lz; dx = ldx; dz = ldz; w = 0.5; a = 0; }
      P[i * 6] = x - dz * w; P[i * 6 + 1] = 0.05; P[i * 6 + 2] = z + dx * w;
      P[i * 6 + 3] = x + dz * w; P[i * 6 + 4] = 0.05; P[i * 6 + 5] = z - dx * w;
      A[i * 2] = A[i * 2 + 1] = a;
      lx = x; lz = z; ldx = dx; ldz = dz;
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.aAlpha.needsUpdate = true;
  }
}

const shadowTex = canvasTex(64, 64, (c, w, h) => {
  const g = c.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
  g.addColorStop(0, 'rgba(0,20,30,0.55)');
  g.addColorStop(1, 'rgba(0,20,30,0)');
  c.fillStyle = g; c.fillRect(0, 0, w, h);
});

// ------------------------------------------------------------ jet skis
function hullGeometry() {
  const g = new THREE.BoxGeometry(1.3, 0.55, 3.4, 2, 1, 10);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    let x = p.getX(i), y = p.getY(i);
    const z = p.getZ(i), t = clamp((z + 1.7) / 3.4, 0, 1);
    let wf = 1;
    if (t > 0.5) wf = 1 - Math.pow((t - 0.5) / 0.5, 1.7) * 0.85;
    if (t < 0.06) wf *= 0.95;
    x *= wf;
    if (y < 0) {
      if (t > 0.55) y += Math.pow((t - 0.55) / 0.45, 2) * 0.36;
      if (Math.abs(x) < 0.01) y -= 0.14 * (1 - Math.max(0, (t - 0.55) / 0.45));
    } else if (t > 0.75) y -= ((t - 0.75) / 0.25) * 0.05;
    p.setXYZ(i, x, y, z);
  }
  g.computeVertexNormals();
  return g;
}
function cowlGeometry() {
  const g = new THREE.BoxGeometry(1.05, 0.42, 1.6, 2, 1, 5);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    let x = p.getX(i), y = p.getY(i);
    const z = p.getZ(i), t = clamp((z + 0.8) / 1.6, 0, 1);
    x *= 1 - Math.pow(t, 1.6) * 0.75;
    if (y > 0) y -= t * 0.3;
    p.setXYZ(i, x, y, z);
  }
  g.computeVertexNormals();
  return g;
}
const HULL_GEO = hullGeometry(), COWL_GEO = cowlGeometry();
const UP = new THREE.Vector3(0, 1, 0);
function limb(a, b, r, mat) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(0.01, len), 3, 8), mat);
  m.position.copy(a).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(UP, dir.normalize());
  return m;
}
const V = (x, y, z) => new THREE.Vector3(x, y, z);

function buildJetski(o) {
  const root = new THREE.Group();
  const body = new THREE.Group();       // banks, pitches and does tricks
  body.position.y = 0.45;
  root.add(body);
  const std = (c, r = 0.45, m = 0.05) => new THREE.MeshStandardMaterial({ color: c, roughness: r, metalness: m });
  const hullMat = std(o.hull, 0.35), lowMat = std(0xf4f6f8, 0.4), darkMat = std(0x1b1f24, 0.7), accentMat = std(o.accent, 0.4);
  const lower = new THREE.Mesh(HULL_GEO, lowMat);
  body.add(lower);
  const upper = new THREE.Mesh(HULL_GEO, hullMat);
  upper.scale.set(1.02, 0.45, 0.97);
  upper.position.set(0, 0.2, 0.02);
  body.add(upper);
  const cowl = new THREE.Mesh(COWL_GEO, hullMat);
  cowl.position.set(0, 0.42, 0.85);
  body.add(cowl);
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(1.36, 0.08, 1.6), accentMat);
  stripe.position.set(0, 0.12, -0.8);
  body.add(stripe);
  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.24, 1.35), darkMat);
  seat.position.set(0, 0.48, -0.55);
  body.add(seat);
  const pad = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.06, 1.0), darkMat);
  pad.position.set(0, 0.33, -0.55);
  body.add(pad);
  body.add(limb(V(0, 0.45, 0.62), V(0, 0.92, 0.42), 0.05, darkMat));
  body.add(limb(V(-0.46, 0.94, 0.4), V(0.46, 0.94, 0.4), 0.04, darkMat));
  const windshield = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.3, 0.05), new THREE.MeshStandardMaterial({ color: 0x223344, roughness: 0.1, metalness: 0.3, transparent: true, opacity: 0.7 }));
  windshield.position.set(0, 0.72, 0.62); windshield.rotation.x = -0.6;
  body.add(windshield);

  // rider
  const rider = new THREE.Group();
  rider.position.set(0, 0, -0.5);
  body.add(rider);
  const suit = std(o.suit, 0.75), vest = std(o.vest, 0.6), skin = std(0xd9a07a, 0.7), helmet = std(o.helmet, 0.25, 0.1);
  const hip = V(0, 0.68, -0.05);
  for (const s of [-1, 1]) {
    const knee = V(s * 0.27, 0.78, 0.48), foot = V(s * 0.42, 0.36, 0.3);
    rider.add(limb(hip.clone().setX(s * 0.13), knee, 0.115, suit));
    rider.add(limb(knee, foot, 0.095, suit));
    const shoulder = V(s * 0.26, 1.28, 0.25), elbow = V(s * 0.42, 1.06, 0.6), hand = V(s * 0.43, 0.95, 0.9);
    rider.add(limb(shoulder, elbow, 0.08, suit));
    rider.add(limb(elbow, hand, 0.07, suit));
    const glove = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 6), darkMat);
    glove.position.copy(hand);
    rider.add(glove);
  }
  rider.add(limb(hip, V(0, 1.3, 0.22), 0.2, suit));
  rider.add(limb(V(0, 0.85, 0.02), V(0, 1.22, 0.2), 0.255, vest));
  const neck = limb(V(0, 1.3, 0.22), V(0, 1.45, 0.3), 0.07, skin);
  rider.add(neck);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 14, 10), helmet);
  head.position.set(0, 1.58, 0.36);
  rider.add(head);
  const visor = new THREE.Mesh(new THREE.SphereGeometry(0.228, 12, 8, Math.PI / 2 - 0.95, 1.9, 1.15, 0.75),
    new THREE.MeshStandardMaterial({ color: 0x0c1116, roughness: 0.1, metalness: 0.6, side: THREE.DoubleSide }));
  visor.position.set(0, 1.58, 0.36);
  rider.add(visor);
  const stripeH = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.08, 0.42), accentMat);
  stripeH.position.set(0, 1.79, 0.34);
  rider.add(stripeH);

  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 5).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }));
  shadow.renderOrder = 1;
  scene.add(shadow);

  root.scale.setScalar(1.2);
  scene.add(root);
  return { root, body, rider, shadow };
}

function nameTag(text, color) {
  const tex = canvasTex(256, 64, (c, w, h) => {
    c.font = 'italic 800 34px Avenir Next Condensed, Arial Narrow, Arial, sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.lineWidth = 6; c.strokeStyle = 'rgba(0,0,0,.6)'; c.strokeText(text, w / 2, h / 2);
    c.fillStyle = color; c.fillText(text, w / 2, h / 2);
  });
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true }));
  sp.scale.set(4.4, 1.1, 1);
  sp.renderOrder = 5;
  scene.add(sp);
  return sp;
}

// --------------------------------------------------------------- racers
const RIDERS = [
  { name: 'YOU', hull: 0xff6a00, accent: 0x111111, suit: 0x16181c, vest: 0xff8c1a, helmet: 0xf4f4f4, player: true },
  { name: 'MARLIN', hull: 0x1f6fff, accent: 0xffffff, suit: 0x0e2a4a, vest: 0xffd23f, helmet: 0xffd23f, skill: 0.95 },
  { name: 'KAI', hull: 0x22c55e, accent: 0x0b3b1f, suit: 0x1d2b22, vest: 0xf4f4f4, helmet: 0x0f172a, skill: 0.91 },
  { name: 'NOVA', hull: 0x9b5cff, accent: 0xffffff, suit: 0x2a1b47, vest: 0x19c3d6, helmet: 0xff4fa3, skill: 0.93 },
  { name: 'REEF', hull: 0xe63946, accent: 0xffffff, suit: 0x3a0d12, vest: 0xf4f4f4, helmet: 0xe63946, skill: 0.88 },
  { name: 'BLAZE', hull: 0xffc300, accent: 0x111111, suit: 0x222222, vest: 0xff3b30, helmet: 0x111111, skill: 0.9 },
];
const racers = RIDERS.map((o, i) => {
  const m = buildJetski(o);
  const r = {
    i, name: o.name, color: '#' + o.hull.toString(16).padStart(6, '0'), isPlayer: !!o.player,
    ...m, wake: new Wake(),
    tag: o.player ? null : nameTag(o.name, '#' + o.helmet.toString(16).padStart(6, '0')),
    x: 0, z: 0, y: 0, vy: 0, vx: 0, vz: 0, heading: 0, speed: 0, steer: 0,
    boostTime: 0, airborne: false, airTime: 0, onRamp: null, stunned: 0, topMul: 1,
    trick: null, tricksDone: [], bank: 0, pitch: 0,
    idx: 0, s: 0, lat: 0, lap: 0, passedHalf: false, progress: 0, finished: false, finishTime: 0,
    lapStart: 0, bestLap: Infinity, place: i + 1, wrongWay: 0,
    ai: o.player ? null : { skill: o.skill, lane: 0, laneTarget: 0, laneTimer: 0, boostCd: mrand(4, 9), seen: null },
  };
  if (r.isPlayer && TEST) r.ai = { skill: 1.0, lane: 0, laneTarget: 0, laneTimer: 0, boostCd: 3, seen: null };
  return r;
});
const player = racers[0];

// -------------------------------------------------------------- game state
const audio = new GameAudio();
const game = {
  state: 'title',        // title | countdown | race | done | results
  paused: false,
  t: 0, raceTime: 0, countdown: 0, lastBeep: 0,
  finishOrder: [],
  score: 0, heat: 0, heatTimer: 0, mult: 1,
  stats: { coins: 0, coinPts: 0, stunts: 0, stuntPts: 0, overtakes: 0, overtakePts: 0, finish: 0 },
  boost: 0.4, boosting: false,
  passed: {},
  doneTimer: 0,
  newBest: false,
};
window.__game = { game, racers, track, get player() { return player; } };

function resetRace() {
  game.raceTime = 0;
  game.finishOrder = [];
  game.score = 0; game.heat = 0; game.heatTimer = 0; game.mult = 1;
  game.stats = { coins: 0, coinPts: 0, stunts: 0, stuntPts: 0, overtakes: 0, overtakePts: 0, finish: 0 };
  game.boost = 0.4; game.boosting = false;
  game.passed = {};
  game.newBest = false;
  coins.forEach((c) => { c.taken = false; });
  spray.clear();
  player.ai = TEST ? { skill: 1.0, lane: 0, laneTarget: 0, laneTimer: 0, boostCd: 3, seen: null } : null;
  // grid: two rows of three behind the line; player on the back row
  const grid = [[1, -11], [3, 0], [2, 11], [4, -11], [0, 0], [5, 11]];
  grid.forEach(([ri, lat], k) => {
    const r = racers[ri];
    const row = Math.floor(k / 3);
    track.at(L - 12 - row * 11, lat, tmp);
    Object.assign(r, {
      x: tmp.x, z: tmp.z, y: 0, vy: 0, vx: 0, vz: 0, heading: tmp.heading, speed: 0, steer: 0,
      boostTime: 0, airborne: false, airTime: 0, onRamp: null, stunned: 0, trick: null, tricksDone: [],
      lap: 0, passedHalf: false, finished: false, finishTime: 0, lapStart: 0, bestLap: Infinity, wrongWay: 0,
      bank: 0, pitch: 0,
    });
    r.idx = track.nearest(r.x, r.z);
    locate(r);
    r.progress = (r.lap - 1) * L + r.s;
    if (r.ai) Object.assign(r.ai, { lane: lat, laneTarget: lat, laneTimer: mrand(1, 3), seen: null, boostCd: mrand(3, 8) });
    r.wake.reset();
  });
  camYaw = player.heading;
}

function locate(r) {
  r.idx = track.nearest(r.x, r.z, r.idx);
  const i = r.idx;
  const dx = r.x - track.px[i], dz = r.z - track.pz[i];
  r.lat = -dx * track.tz[i] + dz * track.tx[i];
  r.s = track.wrapS(i * track.ds + dx * track.tx[i] + dz * track.tz[i]);
}

// ------------------------------------------------------------------ input
const keys = {};
const touch = { left: false, right: false, boost: false };
let trickQueued = false;
let tiltOn = false, tiltSteer = 0;

addEventListener('keydown', (e) => {
  if (e.repeat) { if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' '].includes(e.key)) e.preventDefault(); return; }
  keys[e.code] = true;
  if (['ShiftLeft', 'ShiftRight', 'KeyX', 'KeyK', 'ArrowDown', 'KeyS'].includes(e.code)) trickQueued = true;
  if (e.code === 'Escape' || e.code === 'KeyP') togglePause();
  if (e.code === 'Enter' && game.state === 'title') startRace();
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Space'].includes(e.code)) e.preventDefault();
  audio.unlock();
});
addEventListener('keyup', (e) => { keys[e.code] = false; });

function bindPad(id, onDown, onUp) {
  const el = $(id);
  const down = (e) => { e.preventDefault(); el.classList.add('down'); try { el.setPointerCapture(e.pointerId); } catch (_) {} onDown(); audio.unlock(); };
  const up = (e) => { e.preventDefault(); el.classList.remove('down'); onUp && onUp(); };
  el.addEventListener('pointerdown', down);
  el.addEventListener('pointerup', up);
  el.addEventListener('pointercancel', up);
  el.addEventListener('lostpointercapture', up);
  el.addEventListener('contextmenu', (e) => e.preventDefault());
}
bindPad('t-left', () => { touch.left = true; }, () => { touch.left = false; });
bindPad('t-right', () => { touch.right = true; }, () => { touch.right = false; });
bindPad('t-boost', () => { touch.boost = true; }, () => { touch.boost = false; });
bindPad('t-trick', () => { trickQueued = true; });

// stop page scrolling, pinch zoom and double-tap zoom during play
document.addEventListener('touchmove', (e) => { if (!e.target.closest('.panel')) e.preventDefault(); }, { passive: false });
document.addEventListener('gesturestart', (e) => e.preventDefault());
canvas.addEventListener('touchstart', (e) => e.preventDefault(), { passive: false });

function screenAngle() {
  if (screen.orientation && typeof screen.orientation.angle === 'number') return screen.orientation.angle;
  return typeof window.orientation === 'number' ? window.orientation : 0;
}
function onOrient(e) {
  if (e.beta == null || e.gamma == null) return;
  const b = THREE.MathUtils.degToRad(e.beta), g = THREE.MathUtils.degToRad(e.gamma);
  // gravity direction in device coordinates
  const dx = Math.sin(g) * Math.cos(b), dy = -Math.sin(b);
  const a = ((screenAngle() % 360) + 360) % 360;
  let side;
  if (a === 90) side = -dy;
  else if (a === 270) side = dy;
  else if (a === 180) side = -dx;
  else side = dx;
  const dead = 0.04;
  const v = Math.abs(side) < dead ? 0 : (side - Math.sign(side) * dead) / 0.32;
  tiltSteer = clamp(v, -1, 1);
}
async function enableTilt(on) {
  tiltOn = false;
  window.removeEventListener('deviceorientation', onOrient);
  if (!on) return true;
  try {
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
      const res = await DeviceOrientationEvent.requestPermission();
      if (res !== 'granted') return false;
    }
    window.addEventListener('deviceorientation', onOrient);
    tiltOn = true;
    return true;
  } catch (e) { return false; }
}

function playerInput(dt) {
  let target = 0;
  if (keys.ArrowLeft || keys.KeyA || touch.left) target -= 1;
  if (keys.ArrowRight || keys.KeyD || touch.right) target += 1;
  if (tiltOn && target === 0) target = tiltSteer;
  const rate = target === 0 ? 7 : 5;
  player.steer = clamp(damp(player.steer, target, rate, dt), -1, 1);
  const boost = !!(keys.Space || keys.ArrowUp || keys.KeyW || touch.boost);
  const trick = trickQueued;
  trickQueued = false;
  return { steer: player.steer, boost, trick };
}

// ----------------------------------------------------------------- AI
const features = [...ramps.map((r) => ({ s: r.s, lat: r.lat, kind: 'ramp' })), ...pads.map((p) => ({ s: p.s, lat: p.lat, kind: 'pad' }))];
function timeToLand(r) {
  return (r.vy + Math.sqrt(r.vy * r.vy + 2 * GRAVITY * Math.max(0, r.y))) / GRAVITY;
}
function aiInput(r, dt) {
  const ai = r.ai;
  ai.laneTimer -= dt;
  if (ai.laneTimer <= 0) {
    ai.laneTarget = mrand(-HALF_W * 0.6, HALF_W * 0.6);
    ai.laneTimer = mrand(2.5, 5.5);
  }
  for (const f of features) {
    const d = track.wrapS(f.s - r.s);
    if (d > 8 && d < 110 && ai.seen !== f) {
      ai.seen = f;
      const want = f.kind === 'ramp' ? 0.35 + ai.skill * 0.5 : 0.3 + ai.skill * 0.6;
      if (Math.random() < want) { ai.laneTarget = f.lat + mrand(-1.2, 1.2); ai.laneTimer = d / Math.max(10, r.speed) + 0.6; }
    }
  }
  // steer around riders just ahead
  for (const o of racers) {
    if (o === r) continue;
    const ahead = o.progress - r.progress;
    if (ahead > 0 && ahead < 12 && Math.abs(o.lat - ai.lane) < 3.2) {
      ai.laneTarget = clamp(o.lat + (ai.lane >= o.lat ? 4.5 : -4.5), -HALF_W + 4, HALF_W - 4);
      ai.laneTimer = Math.max(ai.laneTimer, 1);
    }
  }
  ai.laneTarget = clamp(ai.laneTarget, -HALF_W + 4, HALF_W - 4);
  ai.lane = damp(ai.lane, ai.laneTarget, 1.1, dt);
  const look = 12 + r.speed * 0.45;
  track.at(r.s + look, ai.lane, tmp);
  const desired = Math.atan2(tmp.x - r.x, tmp.z - r.z);
  const diff = wrapAngle(desired - r.heading);
  const steer = clamp(-diff * 2.6, -1, 1);
  r.steer = damp(r.steer, steer, 8, dt);

  // pace: skill plus a gentle rubber band around the player
  if (!r.isPlayer) {
    const gap = r.progress - player.progress;
    const band = gap > 0 ? clamp(1 - gap / 1100, 0.9, 1) : clamp(1 - gap / 700, 1, 1.12);
    r.topMul = ai.skill * band;
  }
  let boost = false;
  ai.boostCd -= dt;
  if (ai.boostCd <= 0 && game.state !== 'countdown') {
    r.boostTime = Math.max(r.boostTime, mrand(1.0, 2.0));
    ai.boostCd = mrand(7, 14) / (0.6 + ai.skill * 0.4);
    if (r.isPlayer) boost = true;
  }
  let trick = false;
  if (r.airborne && !r.trick && timeToLand(r) > TRICK_TIME + 0.12 && Math.random() < dt * 6 * ai.skill) trick = true;
  return { steer: r.steer, boost, trick };
}

// ----------------------------------------------------------- simulation
const TRICKS = [
  { id: 'spin', name: '360 SPIN', pts: 400 },
  { id: 'roll', name: 'BARREL ROLL', pts: 500 },
  { id: 'flip', name: 'BACKFLIP', pts: 650 },
];

function updateRacer(r, dt, input) {
  const fx = Math.sin(r.heading), fz = Math.cos(r.heading);
  // ----- speed
  let top = MAX_SPEED * (r.isPlayer ? 1 : r.topMul);
  if (r.isPlayer && TEST) top = MAX_SPEED * 0.97;
  if (r.boostTime > 0) { top = BOOST_SPEED * (r.isPlayer ? 1 : Math.min(1, r.topMul + 0.04)); r.boostTime -= dt; }
  const off = Math.abs(r.lat) - HALF_W;
  if (off > 0) top *= Math.max(0.55, 1 - off * 0.045);
  if (r.stunned > 0) { top *= 0.3; r.stunned -= dt; }
  if (game.state === 'countdown') top = 0;
  if (!r.airborne) {
    const k = r.speed < top ? (r.boostTime > 0 ? 1.6 : 0.55) : 1.4;
    r.speed += (top - r.speed) * Math.min(1, k * dt);
  }
  // ----- steering
  const grip = r.airborne ? 0.15 : 3.4;
  const turn = 1.5 * (0.3 + 0.7 * Math.min(1, r.speed / 16)) * (r.airborne ? 0.4 : 1);
  if (game.state !== 'countdown') r.heading -= input.steer * turn * dt;
  const nfx = Math.sin(r.heading), nfz = Math.cos(r.heading);
  r.vx += (nfx * r.speed - r.vx) * Math.min(1, grip * dt);
  r.vz += (nfz * r.speed - r.vz) * Math.min(1, grip * dt);
  r.x += r.vx * dt;
  r.z += r.vz * dt;

  // ----- track position, walls
  const prevS = r.s;
  locate(r);
  const ex = Math.abs(r.lat) - WALL;
  if (ex > 0) {
    const i = r.idx, sgn = Math.sign(r.lat);
    const rx = -track.tz[i], rz = track.tx[i];
    r.x -= rx * ex * sgn; r.z -= rz * ex * sgn;
    const out = (r.vx * rx + r.vz * rz) * sgn;
    if (out > 0) { r.vx -= rx * out * sgn * 1.2; r.vz -= rz * out * sgn * 1.2; }
    r.speed *= 1 - 1.2 * dt;
    // turn the nose back along the course
    const along = Math.atan2(track.tx[i], track.tz[i]);
    const backward = Math.cos(wrapAngle(r.heading - along)) < 0;
    if (!backward) r.heading += wrapAngle(along - r.heading) * Math.min(1, 2.5 * dt);
    locate(r);
  }

  // ----- laps
  if (r.s > L * 0.4 && r.s < L * 0.6) r.passedHalf = true;
  if (prevS > L * 0.75 && r.s < L * 0.25) {
    if (r.lap === 0 || r.passedHalf) { r.lap++; r.passedHalf = false; onLap(r); }
  } else if (prevS < L * 0.25 && r.s > L * 0.75 && r.lap > 0) {
    r.lap--; r.passedHalf = true;
  }
  r.progress = r.finished ? r.progress : (r.lap - 1) * L + r.s;

  // ----- ramps and air
  let rampY = -1, onRampNow = null;
  for (const rp of ramps) {
    const ds = track.wrapS(r.s - rp.s + L / 2) - L / 2;
    if (ds >= 0 && ds <= RAMP_L && Math.abs(r.lat - rp.lat) <= RAMP_W / 2) {
      const y = -RAMP_SINK + (RAMP_H + RAMP_SINK) * ds / RAMP_L;
      if (y > rampY) { rampY = y; onRampNow = rp; }
    }
  }
  if (!r.airborne) {
    if (onRampNow && rampY > 0) {
      r.y = rampY;
      r.onRamp = onRampNow;
    } else if (r.onRamp) {
      // left the ramp: off the lip gives full launch, off the side just drops
      const lip = r.y > RAMP_H - 0.6;
      r.airborne = true;
      r.vy = lip ? r.speed * (RAMP_H + RAMP_SINK) / RAMP_L * 1.25 + 4 : 0;
      r.airTime = 0;
      r.tricksDone = [];
      r.onRamp = null;
    } else {
      r.y = 0;
    }
  }
  if (r.airborne) {
    r.airTime += dt;
    r.vy -= GRAVITY * dt;
    r.y += r.vy * dt;
    if (input.trick && !r.trick && game.state !== 'countdown') {
      const t = TRICKS[(r.tricksDone.length + (r.isPlayer ? 0 : r.i)) % TRICKS.length];
      r.trick = { ...t, t: 0, dir: Math.random() < 0.5 ? -1 : 1 };
      if (r.isPlayer) audio.boost();
    }
    if (r.trick) {
      r.trick.t += dt;
      if (r.trick.t >= TRICK_TIME) { r.tricksDone.push(r.trick); r.trick = null; }
    }
    if (r.y <= 0 && r.vy < 0) land(r);
  }

  // ----- boost pads
  for (const p of pads) {
    const ds = track.wrapS(r.s - p.s + L / 2) - L / 2;
    if (Math.abs(ds) < 5 && Math.abs(r.lat - p.lat) < 3.6 && !r.airborne) {
      if (r.boostTime < 1.2) {
        if (r.isPlayer && r.boostTime < 0.2) { audio.boost(); toast('BOOST PAD'); }
        r.boostTime = 1.4;
      }
    }
  }

  // ----- wrong way (player)
  if (r.isPlayer) {
    const along = fx * track.tx[r.idx] + fz * track.tz[r.idx];
    r.wrongWay = along < -0.3 && r.speed > 3 ? r.wrongWay + dt : 0;
  }

  // ----- visuals
  const targetBank = (r.airborne ? 0.15 : 0.32) * input.steer * Math.min(1, r.speed / 20);
  r.bank = damp(r.bank, targetBank, 6, dt);
  const targetPitch = r.airborne ? clamp(-r.vy * 0.03, -0.45, 0.5) : (r.onRamp ? -Math.atan((RAMP_H + RAMP_SINK) / RAMP_L) : -0.04 - (r.boostTime > 0 ? 0.05 : 0));
  r.pitch = damp(r.pitch, targetPitch, r.onRamp ? 14 : 5, dt);
}

const qA = new THREE.Quaternion(), qB = new THREE.Quaternion(), eul = new THREE.Euler(0, 0, 0, 'YXZ');
const AX = new THREE.Vector3(1, 0, 0), AY = new THREE.Vector3(0, 1, 0), AZ = new THREE.Vector3(0, 0, 1);
function syncRacer(r, t) {
  const bob = r.airborne || r.onRamp ? 0 : Math.sin(t * 3.1 + r.i) * 0.06 + Math.sin(t * 5.3 + r.i * 2) * 0.03;
  r.root.position.set(r.x, r.y + bob - 0.12, r.z);
  r.root.rotation.y = r.heading;
  eul.set(r.pitch + (r.airborne || r.onRamp ? 0 : Math.sin(t * 2.3 + r.i) * 0.025), 0, r.bank);
  qA.setFromEuler(eul);
  if (r.trick) {
    const p = r.trick.t / TRICK_TIME;
    const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
    const ang = e * Math.PI * 2;
    if (r.trick.id === 'spin') qB.setFromAxisAngle(AY, ang * r.trick.dir);
    else if (r.trick.id === 'roll') qB.setFromAxisAngle(AZ, ang * r.trick.dir);
    else qB.setFromAxisAngle(AX, -ang);
    qA.multiply(qB);
  }
  r.body.quaternion.copy(qA);
  r.rider.rotation.z = r.bank * 0.6;
  r.rider.rotation.x = r.boostTime > 0 ? 0.12 : 0.04;
  const h = Math.max(0, r.y);
  r.shadow.position.set(r.x, 0.04, r.z);
  r.shadow.rotation.y = r.heading;
  const ss = 1 / (1 + h * 0.12);
  r.shadow.scale.set(ss, 1, ss);
  r.shadow.material.opacity = 0.9 * ss;
  if (r.tag) {
    r.tag.position.set(r.x, r.y + 3.6, r.z);
    const d = camera.position.distanceTo(r.root.position);
    r.tag.visible = d < 90 && game.state !== 'title';
    r.tag.material.opacity = clamp((90 - d) / 30, 0, 1);
  }
}

function land(r) {
  r.y = 0;
  r.airborne = false;
  const air = r.airTime;
  r.vy = 0;
  const burst = r.isPlayer ? 46 : 22;
  for (let k = 0; k < burst; k++) {
    const a = Math.random() * Math.PI * 2, sp = mrand(3, 9);
    spray.emit(r.x, 0.3, r.z, Math.cos(a) * sp + r.vx * 0.3, mrand(4, 9), Math.sin(a) * sp + r.vz * 0.3, mrand(0.5, 1.0), mrand(1.0, 2.2));
  }
  if (r.trick) {
    // landed mid-trick: wipeout
    r.trick = null;
    r.stunned = 1.3;
    r.speed *= 0.35;
    r.vx *= 0.35; r.vz *= 0.35;
    if (r.isPlayer) {
      banner('WIPEOUT!', 'finish the trick before you land', true);
      audio.fail();
      game.heat = 0; game.mult = 1; game.heatTimer = 0;
      updateMult();
      shake = 0.6;
    }
    r.tricksDone = [];
    return;
  }
  if (r.isPlayer) {
    audio.splash(0.6 + Math.min(1, air));
    shake = 0.35;
    const done = r.tricksDone;
    if (done.length) {
      const base = done.reduce((a, t) => a + t.pts, 0) * (done.length > 1 ? 1.5 : 1);
      addHeat(3 * done.length);
      const pts = award(base, 'stunt');
      game.stats.stunts += done.length;
      game.boost = Math.min(1, game.boost + 0.22 * done.length);
      const head = done.length > 1 ? 'PERFECT COMBO!' : 'CLEAN LANDING!';
      banner(head, `+${pts.toLocaleString()} · ${done.map((t) => t.name).join(' + ')}`);
      audio.trick();
    } else if (air > 0.6) {
      const pts = award(Math.round(air * 150), 'stunt');
      toast(`BIG AIR +${pts}`);
    }
  }
  r.tricksDone = [];
}

function onLap(r) {
  const now = game.raceTime;
  if (r.lap > 1) {
    const lt = now - r.lapStart;
    if (lt < r.bestLap) r.bestLap = lt;
  }
  r.lapStart = now;
  if (r.isPlayer && r.lap > 1 && r.lap <= LAPS) {
    // fresh coins each lap
    coins.forEach((c) => { c.taken = false; });
    audio.lap();
    banner(r.lap === LAPS ? 'FINAL LAP!' : `LAP ${r.lap}`, `lap time ${fmtTime(now - (r.lapStartPrev || 0))}`);
  }
  r.lapStartPrev = now;
  if (r.lap > LAPS && !r.finished) {
    r.finished = true;
    r.finishTime = now;
    r.progress = LAPS * L + 1000 - game.finishOrder.length;
    game.finishOrder.push(r);
    if (r.isPlayer) playerFinished();
  }
}

// ----------------------------------------------------------------- scoring
function addHeat(n) {
  game.heat += n;
  game.heatTimer = 4.5;
  updateMult();
}
function updateMult() {
  const m = Math.min(8, 1 + Math.floor(game.heat / 5));
  if (m !== game.mult) {
    game.mult = m;
    const el = $('hud-mult');
    el.textContent = 'x' + m;
    el.classList.remove('pop');
    void el.offsetWidth;
    el.classList.add('pop');
    setTimeout(() => el.classList.remove('pop'), 160);
  }
}
function award(base, kind) {
  const pts = Math.round(base * game.mult);
  game.score += pts;
  if (kind === 'coin') game.stats.coinPts += pts;
  else if (kind === 'stunt') game.stats.stuntPts += pts;
  else if (kind === 'overtake') game.stats.overtakePts += pts;
  return pts;
}

// ----------------------------------------------------------------- camera
let camYaw = 0, shake = 0;
const camTarget = new THREE.Vector3();
const camLook = new THREE.Vector3();
let baseFov = 68;
function chaseCamera(dt, snap = false) {
  const r = player;
  camYaw = snap ? r.heading : camYaw + wrapAngle(r.heading - camYaw) * Math.min(1, 4.2 * dt);
  const fx = Math.sin(camYaw), fz = Math.cos(camYaw);
  const back = 9.5 + (r.boostTime > 0 ? 1.2 : 0), up = 3.9;
  camTarget.set(r.x - fx * back, Math.max(r.y * 0.65, 0) + up, r.z - fz * back);
  if (snap) camera.position.copy(camTarget);
  else camera.position.lerp(camTarget, 1 - Math.exp(-7 * dt));
  camLook.set(r.x + fx * 9, r.y * 0.8 + 1.6, r.z + fz * 9);
  if (shake > 0) {
    shake = Math.max(0, shake - dt * 1.6);
    camera.position.x += (Math.random() - 0.5) * shake * 0.6;
    camera.position.y += (Math.random() - 0.5) * shake * 0.6;
  }
  camera.lookAt(camLook);
  const speedK = clamp(r.speed / BOOST_SPEED, 0, 1);
  const want = baseFov + speedK * 10 + (r.boostTime > 0 ? 6 : 0);
  camera.fov = damp(camera.fov, want, 3, dt);
  camera.updateProjectionMatrix();
}
function orbitCamera(t) {
  const r = player;
  const a = t * 0.18 + 2.2;
  camera.position.set(r.x + Math.sin(a) * 15, 4.5 + Math.sin(t * 0.3) * 0.8, r.z + Math.cos(a) * 15);
  camera.lookAt(r.x, 1.6, r.z);
  camera.fov = baseFov;
  camera.updateProjectionMatrix();
}

// -------------------------------------------------------------- HUD / UI
const hud = {
  score: $('hud-score'), best: $('hud-best'), pos: $('hud-pos'), lap: $('hud-lap'), time: $('hud-time'),
  speed: $('hud-speed'), boost: $('hud-boost'), trickBtn: $('t-trick'),
  last: {},
};
function setText(el, key, v) { if (hud.last[key] !== v) { hud.last[key] = v; el.textContent = v; } }
function setHTML(el, key, v) { if (hud.last[key] !== v) { hud.last[key] = v; el.innerHTML = v; } }
let bannerTimer = 0, toastTimer = 0;
function banner(text, sub = '', bad = false) {
  const el = $('banner');
  el.innerHTML = `${text}${sub ? `<span class="pts">${sub}</span>` : ''}`;
  el.classList.toggle('bad', bad);
  el.classList.add('show');
  bannerTimer = 1.8;
}
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  toastTimer = 1.2;
}
function updateHud(dt) {
  setText(hud.score, 'score', String(game.score).padStart(6, '0'));
  setText(hud.best, 'best', String(Math.max(best, game.score)).padStart(6, '0'));
  const p = player.place;
  setHTML(hud.pos, 'pos', `${p}<small>${ordinal(p)}</small>`);
  setText(hud.lap, 'lap', `LAP ${clamp(player.lap, 1, LAPS)}/${LAPS}`);
  setText(hud.time, 'time', fmtTime(game.raceTime));
  setText(hud.speed, 'speed', String(Math.round(player.speed * 3.6)));
  const bw = Math.round(game.boost * 100);
  if (hud.last.boostW !== bw) { hud.last.boostW = bw; hud.boost.style.width = bw + '%'; }
  const active = game.boosting;
  if (hud.last.boostA !== active) { hud.last.boostA = active; hud.boost.classList.toggle('active', active); }
  const ready = player.airborne && !player.trick && timeToLand(player) > TRICK_TIME + 0.05;
  if (hud.last.ready !== ready) { hud.last.ready = ready; hud.trickBtn.classList.toggle('ready', ready); }
  if (bannerTimer > 0) { bannerTimer -= dt; if (bannerTimer <= 0) $('banner').classList.remove('show'); }
  if (toastTimer > 0) { toastTimer -= dt; if (toastTimer <= 0) $('toast').classList.remove('show'); }
  if (player.wrongWay > 1.2 && toastTimer <= 0.2) toast('WRONG WAY  ↺');
  drawMinimap();
}

// minimap
const mm = $('minimap'), mmc = mm.getContext('2d');
const mmBg = document.createElement('canvas');
mmBg.width = mmBg.height = 240;
let mmScale = 1, mmCx = 0, mmCz = 0;
{
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < track.N; i++) {
    minX = Math.min(minX, track.px[i]); maxX = Math.max(maxX, track.px[i]);
    minZ = Math.min(minZ, track.pz[i]); maxZ = Math.max(maxZ, track.pz[i]);
  }
  mmCx = (minX + maxX) / 2; mmCz = (minZ + maxZ) / 2;
  mmScale = 190 / Math.max(maxX - minX, maxZ - minZ);
  const c = mmBg.getContext('2d');
  c.lineJoin = 'round';
  c.beginPath();
  for (let i = 0; i <= track.N; i += 6) {
    const k = i % track.N;
    const [x, y] = mmXY(track.px[k], track.pz[k]);
    if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
  }
  c.closePath();
  c.strokeStyle = 'rgba(255,255,255,.25)'; c.lineWidth = 16; c.stroke();
  c.strokeStyle = 'rgba(255,255,255,.85)'; c.lineWidth = 5; c.stroke();
  const [sx, sy] = mmXY(track.px[0], track.pz[0]);
  c.fillStyle = '#ffd23f'; c.fillRect(sx - 6, sy - 6, 12, 12);
}
function mmXY(x, z) { return [120 + (x - mmCx) * mmScale, 120 + (z - mmCz) * mmScale]; }
function drawMinimap() {
  mmc.clearRect(0, 0, 240, 240);
  mmc.drawImage(mmBg, 0, 0);
  for (let k = racers.length - 1; k >= 0; k--) {
    const r = racers[k];
    const [x, y] = mmXY(r.x, r.z);
    mmc.beginPath();
    mmc.arc(x, y, r.isPlayer ? 11 : 8, 0, Math.PI * 2);
    mmc.fillStyle = r.color;
    mmc.fill();
    if (r.isPlayer) { mmc.lineWidth = 4; mmc.strokeStyle = '#fff'; mmc.stroke(); }
  }
}

// screens
const show = (id, on) => $(id).classList.toggle('hidden', !on);
function setScreen(name) {
  show('title', name === 'title');
  show('results', name === 'results');
  show('pause', name === 'pause');
  const racing = name === 'race';
  show('hud', racing);
  show('touch', racing && (coarse || 'ontouchstart' in window));
}
$('title-best').textContent = best.toLocaleString();

function startRace() {
  audio.unlock();
  resetRace();
  game.state = 'countdown';
  game.countdown = 3.6;
  game.lastBeep = 4;
  game.paused = false;
  setScreen('race');
  hud.last = {};
  $('hud-mult').textContent = 'x1';
  $('banner').classList.remove('show');
  audio.startEngine();
  audio.startMusic();
  audio.setMusicLevel(0.3);
}

function playerFinished() {
  game.state = 'done';
  game.doneTimer = 3.2;
  const place = game.finishOrder.indexOf(player) + 1;
  const bonus = FINISH_BONUS[place - 1] || 0;
  game.stats.finish = bonus;
  game.score += bonus;
  banner(place === 1 ? 'YOU WIN!' : `FINISHED ${place}${ordinal(place)}`, `+${bonus.toLocaleString()} finish bonus`);
  audio.finish(place === 1);
  if (player.ai == null) player.ai = { skill: 0.9, lane: player.lat, laneTarget: player.lat, laneTimer: 2, boostCd: 99, seen: null, auto: true };
}

function showResults() {
  game.state = 'results';
  audio.setMusicLevel(0.35);
  audio.stopEngine();
  const place = game.finishOrder.indexOf(player) + 1;
  $('res-place').textContent = `${place}${ordinal(place)}`;
  $('res-msg').textContent = ['Champion of the lagoon!', 'So close. Silver on the water.', 'On the podium!', 'Solid ride. Go again?', 'Keep at it: boost on the straights.', 'Tip: land tricks to fill the boost meter.'][place - 1];
  // standings: finished riders by time, others by estimated time
  const rows = racers.map((r) => {
    let t = r.finished ? r.finishTime : game.raceTime + (LAPS * L - r.progress) / Math.max(20, MAX_SPEED * (r.topMul || 0.9));
    return { r, t, est: !r.finished };
  }).sort((a, b) => a.t - b.t);
  $('res-table').innerHTML = rows.map((row, k) =>
    `<tr class="${row.r.isPlayer ? 'me' : ''}"><td>${k + 1}</td><td><span class="sw" style="background:${row.r.color}"></span>${row.r.name}</td><td style="text-align:right">${row.est ? '~' : ''}${fmtTime(row.t)}</td></tr>`).join('');
  const s = game.stats;
  if (game.score > best) {
    best = game.score; game.newBest = true;
    try { localStorage.setItem(BEST_KEY, String(best)); } catch (e) { /* ignore */ }
  }
  $('title-best').textContent = best.toLocaleString();
  $('res-breakdown').innerHTML = `
    <div><span>Coins (${s.coins})</span><b>${s.coinPts.toLocaleString()}</b></div>
    <div><span>Stunts (${s.stunts})</span><b>${s.stuntPts.toLocaleString()}</b></div>
    <div><span>Overtakes (${s.overtakes})</span><b>${s.overtakePts.toLocaleString()}</b></div>
    <div><span>Finish bonus</span><b>${s.finish.toLocaleString()}</b></div>
    <div><span>Best lap</span><b>${fmtTime(player.bestLap)}</b></div>
    <div class="total"><span>SCORE</span><b>${game.score.toLocaleString()}</b></div>
    ${game.newBest ? '<div class="newbest">NEW BEST SCORE!</div>' : ''}`;
  setScreen('results');
}

function togglePause(force) {
  if (!['countdown', 'race', 'done'].includes(game.state)) return;
  game.paused = force !== undefined ? force : !game.paused;
  show('pause', game.paused);
  if (game.paused) { audio.suspend(); } else { audio.resume(); last = performance.now(); }
}
function toMenu() {
  game.paused = false;
  game.state = 'title';
  audio.stopEngine();
  audio.setMusicLevel(0.2);
  resetRace();
  setScreen('title');
}

$('btn-play').addEventListener('click', startRace);
$('btn-again').addEventListener('click', startRace);
$('btn-resume').addEventListener('click', () => togglePause(false));
$('btn-restart').addEventListener('click', () => { show('pause', false); startRace(); });
$('btn-menu').addEventListener('click', toMenu);
$('btn-menu2').addEventListener('click', toMenu);
$('btn-pause').addEventListener('click', () => togglePause(true));
$('opt-sound').addEventListener('change', (e) => { audio.setEnabled(e.target.checked); if (e.target.checked) audio.unlock(); });
const tiltBox = $('opt-tilt');
if (!coarse || typeof DeviceOrientationEvent === 'undefined') $('tilt-row').classList.add('hidden');
tiltBox.addEventListener('change', async () => {
  const ok = await enableTilt(tiltBox.checked);
  if (!ok) tiltBox.checked = false;
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { if (['countdown', 'race', 'done'].includes(game.state)) togglePause(true); audio.suspend(); }
  else if (!game.paused) audio.resume();
});
// first tap anywhere wakes audio on iOS
// (iOS only treats touchend / click / keydown as gestures that can start audio)
const WAKE_EVENTS = ['touchend', 'click', 'keydown'];
function wakeAudio() {
  audio.unlock();
  if (game.state === 'title' && audio.ctx) { audio.startMusic(); audio.setMusicLevel(0.25); }
  if (audio.ctx && audio.ctx.state === 'running') WAKE_EVENTS.forEach((ev) => removeEventListener(ev, wakeAudio, true));
}
WAKE_EVENTS.forEach((ev) => addEventListener(ev, wakeAudio, true));

// ---------------------------------------------------------------- resize
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // keep a sensible horizontal view in portrait
  const hfov = 92;
  const vfov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(hfov / 2)) / camera.aspect));
  baseFov = clamp(vfov, 60, 92);
  camera.fov = baseFov;
  camera.updateProjectionMatrix();
  spray.mat.uniforms.uScale.value = (h * renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(baseFov / 2)));
}
addEventListener('resize', resize);
if (window.visualViewport) visualViewport.addEventListener('resize', resize);
resize();

// --------------------------------------------------------------- frame
const mtx = new THREE.Matrix4(), qq = new THREE.Quaternion(), vv = new THREE.Vector3(), ss = new THREE.Vector3();
let last = performance.now();
let perfT = 0, perfN = 0, perfChecked = false;

function step(dt) {
  game.t += dt;
  const t = game.t;
  waterUniforms.uTime.value = t;
  padTex.offset.y = -t * 1.6;

  if (game.state === 'countdown') {
    game.countdown -= dt;
    const n = Math.ceil(game.countdown);
    if (n < game.lastBeep && n <= 3 && n >= 1) { game.lastBeep = n; audio.beep(false); countdownText(String(n)); }
    if (game.countdown <= 0) {
      game.state = 'race';
      audio.beep(true);
      countdownText('GO!', true);
      audio.setMusicLevel(0.55);
      racers.forEach((r) => { r.lapStart = 0; });
    }
  }
  const racing = game.state === 'race' || game.state === 'done' || game.state === 'results' || game.state === 'countdown';
  if (racing) {
    if (game.state !== 'countdown') game.raceTime += dt;
    // player boost from the meter
    let pin = { steer: 0, boost: false, trick: false };
    if (player.ai) pin = aiInput(player, dt);
    else pin = playerInput(dt);
    game.boosting = false;
    if (!player.ai && pin.boost && game.boost > 0.01 && game.state === 'race') {
      if (player.boostTime <= 0.05) audio.boost();
      player.boostTime = Math.max(player.boostTime, 0.12);
      game.boost = Math.max(0, game.boost - 0.33 * dt);
      game.boosting = true;
    }
    for (const r of racers) updateRacer(r, dt, r.isPlayer ? pin : aiInput(r, dt));
    collide(dt);
    rank();
    if (game.state === 'race' || game.state === 'done') playerEvents(dt);
    if (game.state === 'done') { game.doneTimer -= dt; if (game.doneTimer <= 0) showResults(); }
  }

  // effects
  for (const r of racers) {
    r.wake.update(dt, r);
    if (!r.airborne && r.speed > 4) {
      const near = r.isPlayer || camera.position.distanceToSquared(r.root.position) < 140 * 140;
      if (near) {
        const fx = Math.sin(r.heading), fz = Math.cos(r.heading);
        const rate = (r.isPlayer ? 1.4 : 0.7) * r.speed * dt * (r.boostTime > 0 ? 1.8 : 1);
        let n = Math.floor(rate) + (Math.random() < rate % 1 ? 1 : 0);
        while (n-- > 0) {
          const side = Math.random() < 0.5 ? -1 : 1;
          const up = r.boostTime > 0 ? mrand(6, 11) : mrand(2.5, 6);
          spray.emit(r.x - fx * 2.1 + mrand(-0.3, 0.3), 0.4, r.z - fz * 2.1 + mrand(-0.3, 0.3),
            -fx * r.speed * 0.12 + fz * side * mrand(0.5, 2.5) * (1 + Math.abs(r.steer) * 2),
            up,
            -fz * r.speed * 0.12 - fx * side * mrand(0.5, 2.5) * (1 + Math.abs(r.steer) * 2),
            mrand(0.45, 0.9), mrand(0.5, 1.2));
        }
      }
    }
  }
  spray.update(dt);
}

function countdownText(txt, go = false) {
  const el = $('countdown');
  el.textContent = txt;
  el.classList.toggle('go', go);
  el.classList.remove('hidden');
  clearTimeout(countdownText.t);
  countdownText.t = setTimeout(() => el.classList.add('hidden'), go ? 700 : 900);
}

function collide(dt) {
  for (let a = 0; a < racers.length; a++) {
    for (let b = a + 1; b < racers.length; b++) {
      const A = racers[a], B = racers[b];
      const dx = B.x - A.x, dz = B.z - A.z;
      const d2 = dx * dx + dz * dz;
      const R = 2.5;
      if (d2 < R * R && d2 > 1e-6 && Math.abs(A.y - B.y) < 1.5) {
        const d = Math.sqrt(d2), push = (R - d) / 2, nx = dx / d, nz = dz / d;
        A.x -= nx * push; A.z -= nz * push; B.x += nx * push; B.z += nz * push;
        const rel = (B.vx - A.vx) * nx + (B.vz - A.vz) * nz;
        if (rel < 0) {
          A.vx += nx * rel * 0.5; A.vz += nz * rel * 0.5; B.vx -= nx * rel * 0.5; B.vz -= nz * rel * 0.5;
          A.speed *= 0.985; B.speed *= 0.985;
          if ((A.isPlayer || B.isPlayer) && rel < -3) { audio.bump(); shake = Math.max(shake, 0.25); }
        }
      }
    }
  }
}

function rank() {
  const order = [...racers].sort((a, b) => {
    if (a.finished && b.finished) return a.finishTime - b.finishTime;
    if (a.finished) return -1;
    if (b.finished) return 1;
    return b.progress - a.progress;
  });
  order.forEach((r, k) => { r.place = k + 1; });
}

function playerEvents(dt) {
  if (game.heatTimer > 0) {
    game.heatTimer -= dt;
    if (game.heatTimer <= 0) { game.heat = 0; updateMult(); }
  }
  if (player.finished) return;
  // coins
  const R2 = 2.9 * 2.9;
  for (const c of coins) {
    if (c.taken) continue;
    const dx = c.x - player.x, dy = c.y - (player.y + 1.2), dz = c.z - player.z;
    if (dx * dx + dy * dy + dz * dz < R2) {
      c.taken = true;
      game.stats.coins++;
      addHeat(1);
      award(50, 'coin');
      game.boost = Math.min(1, game.boost + 0.07);
      audio.coin();
      for (let k = 0; k < 8; k++) spray.emit(c.x, c.y, c.z, mrand(-3, 3), mrand(1, 5), mrand(-3, 3), 0.4, 0.8);
    }
  }
  // overtakes: once per rival per lap
  if (game.state === 'race' && game.raceTime > 4) {
    for (const o of racers) {
      if (o === player || o.finished) continue;
      const key = o.i;
      const ahead = player.progress - o.progress;
      if (!game.passed[key]) game.passed[key] = { behind: ahead < 0, lap: -1 };
      const st = game.passed[key];
      if (ahead < -2) st.behind = true;
      if (st.behind && ahead > 2) {
        st.behind = false;
        if (st.lap !== player.lap) {
          st.lap = player.lap;
          game.stats.overtakes++;
          addHeat(2);
          const pts = award(150, 'overtake');
          toast(`OVERTOOK ${o.name} +${pts}`);
          audio.overtake();
        }
      }
    }
  }
}

function render(dt) {
  const t = game.t;
  if (game.state === 'title') orbitCamera(t);
  else chaseCamera(dt, false);
  for (const r of racers) syncRacer(r, t);
  water.position.set(camera.position.x, 0, camera.position.z);
  sky.position.copy(camera.position);

  // buoys bob
  for (let k = 0; k < buoys.length; k++) {
    const b = buoys[k];
    vv.set(b.x, Math.sin(t * 2 + b.ph) * 0.12 - 0.1, b.z);
    qq.setFromAxisAngle(AZ, Math.sin(t * 1.7 + b.ph) * 0.08);
    mtx.compose(vv, qq, ss.set(1, 1, 1));
    buoyMesh.setMatrixAt(k, mtx);
  }
  buoyMesh.instanceMatrix.needsUpdate = true;
  // coins spin
  for (let k = 0; k < coins.length; k++) {
    const c = coins[k];
    vv.set(c.x, c.y + Math.sin(t * 3 + k) * 0.15, c.z);
    qq.setFromAxisAngle(AY, t * 3 + k * 0.4);
    mtx.compose(vv, qq, ss.setScalar(c.taken ? 0 : 1));
    coinMesh.setMatrixAt(k, mtx);
  }
  coinMesh.instanceMatrix.needsUpdate = true;
  for (const b of boats) { b.g.position.y = 0.3 + Math.sin(t * 1.2 + b.ph) * 0.15; b.g.rotation.z = Math.sin(t * 0.9 + b.ph) * 0.03; }

  if (game.state === 'race' || game.state === 'done' || game.state === 'countdown') {
    updateHud(dt);
    audio.updateEngine(clamp(player.speed / BOOST_SPEED, 0, 1), player.boostTime > 0, player.airborne);
  }
  renderer.render(scene, camera);
}

function frame(now) {
  requestAnimationFrame(frame);
  let real = (now - last) / 1000;
  last = now;
  if (!(real > 0)) real = 0;
  const frameDt = Math.min(0.05, real);
  if (!game.paused) {
    const total = frameDt * TIME_SCALE;
    const n = Math.max(1, Math.ceil(total / 0.034));
    for (let k = 0; k < n; k++) step(total / n);
    render(Math.min(0.5, frameDt * TIME_SCALE));
  }
  // drop the resolution once if this device is struggling
  if (!perfChecked && game.state === 'race' && !game.paused) {
    perfT += real; perfN++;
    if (perfT > 4) {
      perfChecked = true;
      const avg = perfT / perfN;
      if (avg > 1 / 40 && pixelRatio > 1.2) {
        pixelRatio = Math.max(1, pixelRatio * 0.7);
        renderer.setPixelRatio(pixelRatio);
        resize();
      }
    }
  }
}

resetRace();
setScreen('title');
$('loading').classList.add('hidden');
requestAnimationFrame((now) => { last = now; frame(now); });
if (TEST && qs.has('autostart')) startRace();
