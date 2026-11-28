// Procedural music and sound effects built on Web Audio. No audio files.
//
// iOS notes: the AudioContext is created and resumed inside a tap; a silent
// looping <audio> element and navigator.audioSession put the page in the
// "playback" audio category so sound plays even with the ringer switched off.

const SILENT_WAV =
  'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQQAAAAAAAAA';

const NOTE = (n) => 440 * Math.pow(2, (n - 69) / 12);

export class GameAudio {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.music = null;
    this.engines = [];
    this.unlockEl = null;
  }

  // Must be called from a user gesture (tap / click / key).
  unlock() {
    if (!this.enabled) return;
    try {
      if (navigator.audioSession) navigator.audioSession.type = 'playback';
    } catch (e) { /* not supported */ }
    if (!this.unlockEl) {
      const el = document.createElement('audio');
      el.src = SILENT_WAV;
      el.loop = true;
      el.setAttribute('playsinline', '');
      el.setAttribute('x-webkit-airplay', 'deny');
      el.preload = 'auto';
      this.unlockEl = el;
    }
    this.unlockEl.play().catch(() => {});
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      this.ctx = new AC();
      this._build();
    }
    if (this.ctx.state !== 'running') this.ctx.resume().catch(() => {});
    // A one-sample buffer played in the gesture fully wakes older iOS versions.
    const b = this.ctx.createBuffer(1, 1, 22050);
    const s = this.ctx.createBufferSource();
    s.buffer = b;
    s.connect(this.ctx.destination);
    s.start(0);
  }

  setEnabled(on) {
    this.enabled = on;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(t);
    this.master.gain.setTargetAtTime(on ? 0.9 : 0, t, 0.05);
    if (!on && this.unlockEl) this.unlockEl.pause();
  }

  suspend() { if (this.ctx && this.ctx.state === 'running') this.ctx.suspend(); }
  resume() { if (this.ctx && this.enabled && this.ctx.state !== 'running') this.ctx.resume(); }

  _build() {
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = this.enabled ? 0.9 : 0;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.ratio.value = 4;
    this.master.connect(comp);
    comp.connect(ctx.destination);

    this.musicBus = ctx.createGain();
    this.musicBus.gain.value = 0.0;
    this.musicBus.connect(this.master);
    this.sfxBus = ctx.createGain();
    this.sfxBus.gain.value = 0.8;
    this.sfxBus.connect(this.master);

    // white noise buffer reused by drums, splashes and the engine wash
    const len = ctx.sampleRate * 2;
    this.noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;

    // simple feedback delay for the lead
    this.delay = ctx.createDelay(1);
    this.delay.delayTime.value = 0.351;
    const fb = ctx.createGain();
    fb.gain.value = 0.28;
    const dl = ctx.createGain();
    dl.gain.value = 0.35;
    this.delay.connect(fb);
    fb.connect(this.delay);
    this.delay.connect(dl);
    dl.connect(this.musicBus);
  }

  // ---------------------------------------------------------------- music
  startMusic() {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.musicBus.gain.cancelScheduledValues(t);
    this.musicBus.gain.setTargetAtTime(0.55, t, 0.3);
    if (this.music) return;
    const bpm = 126;
    this.music = { step: 0, next: this.ctx.currentTime + 0.1, spb: 60 / bpm / 4 };
    this.music.timer = setInterval(() => this._schedule(), 25);
  }

  stopMusic(fade = 0.4) {
    if (!this.ctx || !this.music) return;
    const t = this.ctx.currentTime;
    this.musicBus.gain.cancelScheduledValues(t);
    this.musicBus.gain.setTargetAtTime(0, t, fade / 3);
    const m = this.music;
    setTimeout(() => { clearInterval(m.timer); }, fade * 1000 + 200);
    this.music = null;
  }

  setMusicLevel(v) {
    if (!this.ctx || !this.music) return;
    this.musicBus.gain.setTargetAtTime(v, this.ctx.currentTime, 0.4);
  }

  _schedule() {
    const m = this.music;
    if (!m || !this.ctx) return;
    while (m.next < this.ctx.currentTime + 0.12) {
      this._step(m.step, m.next, m.spb);
      m.step++;
      m.next += m.spb;
    }
  }

  _step(step, t, spb) {
    // A minor: Am - F - C - G, one bar each, 16 steps per bar
    const bar = Math.floor(step / 16) % 4;
    const s = step % 16;
    const roots = [57, 53, 48, 55]; // A3 F3 C3 G3
    const chords = [[0, 3, 7], [0, 4, 7], [0, 4, 7], [0, 4, 7]];
    const root = roots[bar];
    const section = Math.floor(step / 64) % 4; // every 4 bars, vary the arrangement

    // kick on the beat
    if (s % 4 === 0) this._kick(t);
    // clap on 2 and 4
    if (s === 4 || s === 12) this._clap(t);
    // hats on off-beats, 16ths in later sections
    if (s % 4 === 2) this._hat(t, 0.11, 0.05);
    else if (section >= 2 && s % 2 === 1) this._hat(t, 0.05, 0.03);

    // driving bass: off-beat eighths with octave jumps
    if (s % 2 === 0) {
      const n = root - 12 + (s % 8 === 6 ? 12 : 0);
      this._bass(t, NOTE(n), spb * 1.7);
    }

    // arpeggio lead
    if (section !== 1 || bar >= 2) {
      const pattern = [0, 1, 2, 1, 2, 0, 1, 2];
      const c = chords[bar];
      if (s % 2 === 0) {
        const idx = pattern[(s / 2) % 8];
        const oct = (s >= 8 && section >= 1) ? 24 : 12;
        this._lead(t, NOTE(root + c[idx] + oct), spb * 1.6);
      }
    }
    // pad on each bar
    if (s === 0) {
      for (const iv of chords[bar]) this._pad(t, NOTE(root + iv), spb * 16);
    }
  }

  _env(g, t, a, peak, d) {
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + a + d);
  }

  _kick(t) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(42, t + 0.12);
    this._env(g, t, 0.003, 0.9, 0.22);
    o.connect(g); g.connect(this.musicBus);
    o.start(t); o.stop(t + 0.3);
  }

  _noiseHit(t, type, freq, q, peak, dur, bus) {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    const g = ctx.createGain();
    this._env(g, t, 0.002, peak, dur);
    src.connect(f); f.connect(g); g.connect(bus || this.musicBus);
    src.start(t, Math.random() * 1.5); src.stop(t + dur + 0.05);
    return { src, f, g };
  }

  _clap(t) { this._noiseHit(t, 'bandpass', 1500, 0.8, 0.45, 0.14); }
  _hat(t, peak, dur) { this._noiseHit(t, 'highpass', 8000, 0.5, peak, dur); }

  _bass(t, f, dur) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = 'sawtooth';
    o.frequency.value = f;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.Q.value = 6;
    lp.frequency.setValueAtTime(900, t);
    lp.frequency.exponentialRampToValueAtTime(160, t + dur);
    const g = ctx.createGain();
    this._env(g, t, 0.005, 0.32, dur);
    o.connect(lp); lp.connect(g); g.connect(this.musicBus);
    o.start(t); o.stop(t + dur + 0.05);
  }

  _lead(t, f, dur) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = 'square';
    o.frequency.value = f;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 2600;
    const g = ctx.createGain();
    this._env(g, t, 0.004, 0.075, dur);
    o.connect(lp); lp.connect(g); g.connect(this.musicBus); g.connect(this.delay);
    o.start(t); o.stop(t + dur + 0.05);
  }

  _pad(t, f, dur) {
    const ctx = this.ctx;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.035, t + 0.4);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 1200;
    for (const det of [-8, 8]) {
      const o = ctx.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = f;
      o.detune.value = det;
      o.connect(lp);
      o.start(t); o.stop(t + dur + 0.05);
    }
    lp.connect(g); g.connect(this.musicBus);
  }

  // ------------------------------------------------------------- engine
  // One engine voice for the player; volume and pitch follow speed.
  startEngine() {
    if (!this.ctx || this.engine) return;
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = 'sawtooth';
    o.frequency.value = 50;
    const o2 = ctx.createOscillator();
    o2.type = 'square';
    o2.frequency.value = 25;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 500;
    lp.Q.value = 2;
    const g = ctx.createGain();
    g.gain.value = 0;
    const g2 = ctx.createGain();
    g2.gain.value = 0.35;
    o.connect(lp); o2.connect(g2); g2.connect(lp); lp.connect(g); g.connect(this.sfxBus);

    // water wash: looped noise through a bandpass
    const w = ctx.createBufferSource();
    w.buffer = this.noise;
    w.loop = true;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 900;
    bp.Q.value = 0.6;
    const wg = ctx.createGain();
    wg.gain.value = 0;
    w.connect(bp); bp.connect(wg); wg.connect(this.sfxBus);
    o.start(); o2.start(); w.start();
    this.engine = { o, o2, lp, g, w, bp, wg };
  }

  updateEngine(speed01, boosting, airborne) {
    const e = this.engine;
    if (!e) return;
    const t = this.ctx.currentTime;
    const f = 48 + speed01 * 95 + (boosting ? 25 : 0) + (airborne ? 30 : 0);
    e.o.frequency.setTargetAtTime(f, t, 0.08);
    e.o2.frequency.setTargetAtTime(f / 2, t, 0.08);
    e.lp.frequency.setTargetAtTime(380 + speed01 * 1300 + (airborne ? 500 : 0), t, 0.1);
    e.g.gain.setTargetAtTime(0.09 + speed01 * 0.08, t, 0.1);
    e.wg.gain.setTargetAtTime(airborne ? 0.0 : 0.05 + speed01 * 0.12, t, 0.08);
    e.bp.frequency.setTargetAtTime(600 + speed01 * 900, t, 0.1);
  }

  stopEngine() {
    const e = this.engine;
    if (!e) return;
    const t = this.ctx.currentTime;
    e.g.gain.setTargetAtTime(0, t, 0.1);
    e.wg.gain.setTargetAtTime(0, t, 0.1);
    setTimeout(() => { try { e.o.stop(); e.o2.stop(); e.w.stop(); } catch (_) {} }, 600);
    this.engine = null;
  }

  // ----------------------------------------------------------------- sfx
  _tone(f, dur, type = 'square', peak = 0.15, when = 0, slideTo = 0) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime + when;
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f, t);
    if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
    const g = this.ctx.createGain();
    this._env(g, t, 0.004, peak, dur);
    o.connect(g); g.connect(this.sfxBus);
    o.start(t); o.stop(t + dur + 0.05);
  }

  coin() { this._tone(NOTE(88), 0.06, 'square', 0.09); this._tone(NOTE(95), 0.16, 'square', 0.09, 0.06); }
  beep(high) { this._tone(high ? 880 : 440, high ? 0.5 : 0.18, 'square', 0.14); }
  trick() { [72, 76, 79, 84].forEach((n, i) => this._tone(NOTE(n), 0.12, 'triangle', 0.18, i * 0.06)); }
  overtake() { this._tone(NOTE(79), 0.08, 'triangle', 0.14); this._tone(NOTE(86), 0.14, 'triangle', 0.14, 0.07); }
  lap() { [67, 71, 74, 79].forEach((n, i) => this._tone(NOTE(n), 0.16, 'square', 0.1, i * 0.08)); }
  bump() { this._tone(120, 0.15, 'sawtooth', 0.18, 0, 60); }
  fail() { this._tone(300, 0.5, 'sawtooth', 0.15, 0, 70); }
  boost() {
    if (!this.ctx) return;
    const h = this._noiseHit(this.ctx.currentTime, 'bandpass', 600, 1.2, 0.35, 0.6, this.sfxBus);
    h.f.frequency.exponentialRampToValueAtTime(3000, this.ctx.currentTime + 0.5);
  }
  splash(size = 1) {
    if (!this.ctx) return;
    this._noiseHit(this.ctx.currentTime, 'lowpass', 1400, 0.7, 0.5 * size, 0.55, this.sfxBus);
  }
  finish(win) {
    const seq = win ? [72, 76, 79, 84, 88, 91] : [72, 74, 76, 79];
    seq.forEach((n, i) => this._tone(NOTE(n), 0.22, 'square', 0.12, i * 0.11));
  }
}
