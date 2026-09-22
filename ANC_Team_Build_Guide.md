# Erasure-Aware Speech Enhancement — Team-of-3 Build Guide

This version reorganizes the full implementation into **three parallel workstreams** (one per teammate), with clear **sync points** where your work merges and gets tested together. Follow it top to bottom as your actual build calendar.

---

## 0. How this guide works

You are not building this in one long sequential chain — that wastes two of your three people. Instead:

- **Person A owns the left side of the pipeline:** data, the physics simulator, and experiment tracking (MLflow).
- **Person B owns the safety chain:** detector, gate, limiter, and the real-time audio plumbing.
- **Person C owns the "brains":** the GTCRN enhancer, LPC concealment, training loop, and evaluation.

Each workstream can start on **Day 0 in parallel**, because each one only needs *synthetic placeholder inputs* to begin — nobody has to wait for anybody else until the first **Sync Point**.

```
Week:        1        2        3        4        5        6        7        8        9-12
Person A:  [Setup+Sim data]──[MLflow+more data]──────[support/iterate]────────[demo data]
Person B:  [Detector]──[Gate+Limiter]──[Realtime I/O skeleton]──[Integrate]───[Demo rig]
Person C:  [GTCRN inference]──[Training loop]──[LPC concealment]──[Integrate]─[Eval+ablation]
                              ↑ SYNC 1                            ↑ SYNC 2    ↑ SYNC 3
```

---

## 1. Day 0 — everyone together (2–3 hours, do this as a group)

1. Create the shared GitHub repo. Agree on the branch model:
   - `main` — always working, only updated at Sync Points via reviewed PRs
   - `person-a`, `person-b`, `person-c` — everyone works on their own branch day-to-day
2. Set up the shared folder structure (same as before):

```
anc-project/
├── data/{raw, simulated, real_clipped}/
├── src/{simulator, detector, gate, enhancer, concealment, pipeline, train, eval, realtime}/
├── checkpoints/
├── mlruns/                 # MLflow local tracking store (gitignored)
├── results/
└── README.md
```

3. Everyone installs the same environment:

```bash
python -m venv anc-env
source anc-env/bin/activate
pip install torch torchaudio numpy scipy librosa soundfile \
            sounddevice pyroomacoustics onnx onnxruntime \
            matplotlib pandas pesq pystoi mlflow
```

4. **Decide and write down the shared interfaces right now**, before anyone codes — this is the single most important step for a 3-person team, because it lets you build in parallel without stepping on each other:

```python
# THE CONTRACT — put this in src/interfaces.py and never break it without telling the other two

# Person A's simulator produces:
#   noisy_waveform:    np.ndarray, shape (N,), float32, range [-1, 1]
#   clean_target:      np.ndarray, shape (N,), float32
#   erasure_mask:       np.ndarray, shape (N,), {0.0, 1.0}

# Person B's detector consumes:
#   primary_block:      np.ndarray, shape (block_len,), float32   # e.g. 160 samples @16kHz = 10ms
#   reference_block:     np.ndarray or None, same shape
# and produces:
#   erased: bool   (per block)

# Person B's gate consumes:
#   x: np.ndarray, erasure_mask: np.ndarray (same shape)
# and produces:
#   gated: np.ndarray (same shape, erased samples zeroed + ramped)

# Person C's enhancer consumes:
#   gated_waveform: np.ndarray, state: dict or None
# and produces:
#   enhanced: np.ndarray, new_state: dict

# Person C's LPC concealment consumes:
#   enhanced_history: np.ndarray (last ~30ms), gap_len: int
# and produces:
#   filled: np.ndarray, shape (gap_len,)
```

Print this out or pin it in your team chat. As long as every function respects these shapes and types, the three of you can build independently and it will click together at Sync Point 1.

5. Assign a 15-minute daily standup time (even async in chat is fine): each person posts *what I finished*, *what I'm doing next*, *what I'm blocked on*.

---

## 2. Workstream A (Person A) — Data + Physics Simulator + Experiment Tracking

### Step A1 — Download the datasets (Day 1)

| Purpose | Dataset | Where |
|---|---|---|
| Clean speech | LibriSpeech train-clean-100 | openslr.org/12 |
| Lombard speech | Lombard GRID | GitHub (Alghamdi et al.) |
| Military/vehicle noise | NOISEX-92 | search mirrors |
| Gunshot recordings | UrbanSound8K / MIVIA gunshot dataset | Kaggle / MIVIA lab |
| Helicopter/siren/engine | AudioSet / Freesound | freesound.org, CC-licensed only |
| Real defence audio (bonus) | MAD (Military Audio Dataset), Nature Sci. Data 2024 | paper's data DOI |

Save everything to `data/raw/` and write `data/raw/sources.md` listing exact URLs + licenses — you'll need this for the References slide.

### Step A2 — Build the capture-chain simulator (Days 2–4)

```python
# src/simulator/capture_chain.py
import numpy as np

def db_spl_to_amplitude(db_spl, ref=20e-6):
    pa = ref * (10 ** (db_spl / 20))
    return pa / 200.0   # calibration constant — tune against a real recording

def db_to_linear(db):
    return 10 ** (db / 20)

def soft_saturate(x, threshold):
    return np.tanh(x / threshold) * threshold

def friedlander_blast(t, peak_amp, t_plus=0.001, alpha=3.0):
    p = peak_amp * (1 - t / t_plus) * np.exp(-alpha * t / t_plus)
    p[t < 0] = 0
    return p

def mix_absolute_units(speech, noise, impulse,
                        speech_db_spl=95, noise_db_spl=100,
                        impulse_peak_db_spl=150, mic_overload_db_spl=125,
                        preamp_gain_db=20, clip_type="hard"):
    speech_scaled  = speech  * db_spl_to_amplitude(speech_db_spl)
    noise_scaled   = noise   * db_spl_to_amplitude(noise_db_spl)
    impulse_scaled = impulse * db_spl_to_amplitude(impulse_peak_db_spl)

    at_mic = speech_scaled + noise_scaled + impulse_scaled
    clean_target = speech_scaled

    at_mic = soft_saturate(at_mic, threshold=db_spl_to_amplitude(mic_overload_db_spl))
    post_preamp = at_mic * db_to_linear(preamp_gain_db)

    if clip_type == "hard":
        noisy_signal = np.clip(post_preamp, -1.0, 1.0)
    else:
        noisy_signal = np.tanh(post_preamp)

    erasure_mask = (np.abs(noisy_signal) >= 0.97).astype(np.float32)
    return noisy_signal.astype(np.float32), clean_target.astype(np.float32), erasure_mask
```

**Randomize these per utterance** (verify each range against a source before quoting it in slides):

| Parameter | Range |
|---|---|
| Speech level at 2.5cm | 85–105 dB SPL |
| Continuous noise at headset | 85–115 dB SPL |
| Impulse peak at headset | 130–165 dB SPL |
| Mic overload point | 120–135 dB SPL |
| Clip type | hard / soft(tanh) — randomize |
| Room reverb (RT60) | 0.1–0.5s via `pyroomacoustics` |

### Step A3 — Generate the first data batch (Day 5)

```python
# src/simulator/generate_batch.py
import numpy as np, soundfile as sf, random, os
from capture_chain import mix_absolute_units

def generate_batch(clean_paths, noise_paths, impulse_paths, out_dir, n=500, sr=16000):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        speech, _ = sf.read(random.choice(clean_paths))
        noise, _  = sf.read(random.choice(noise_paths))
        impulse, _ = sf.read(random.choice(impulse_paths))
        # pad/trim all to same length here (omitted for brevity)
        noisy, clean, mask = mix_absolute_units(
            speech, noise, impulse,
            speech_db_spl=random.uniform(85, 105),
            noise_db_spl=random.uniform(85, 115),
            impulse_peak_db_spl=random.uniform(130, 165),
            clip_type=random.choice(["hard", "soft"]))
        sf.write(f"{out_dir}/{i:05d}_noisy.wav", noisy, sr)
        sf.write(f"{out_dir}/{i:05d}_clean.wav", clean, sr)
        np.save(f"{out_dir}/{i:05d}_mask.npy", mask)
```

Run this to produce your first ~500 training triplets in `data/simulated/`. **Hand this off to Person C as soon as you have even 50 examples** — don't wait for a perfect dataset before unblocking training.

### Step A4 — Set up MLflow (Day 5–6, can overlap with A3)

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns
```

Run this on whichever machine stays on most (or a shared cloud VM if you have one). Give the other two teammates the URL (`http://<your-ip>:5000`) so all three of you log to the same dashboard.

```python
# src/tracking/mlflow_setup.py
import mlflow

def init_tracking(tracking_uri="http://<team-server-ip>:5000", experiment="anc-ablation"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
```

Share this one file with the team — everyone imports `init_tracking()` at the top of any training/eval script so all runs land in the same place automatically.

### Step A5 onward — keep feeding the pipeline

Once the simulator works, your ongoing job is: generate bigger/better batches, add real recorded gunshot tails, start collecting real clipped clap/balloon recordings (Step needed before Sync Point 3), and support Person C with data questions.

---

## 3. Workstream B (Person B) — Detector, Gate, Limiter, Real-Time Chain

### Step B1 — Build the detector, test it standalone (Days 1–3)

```python
# src/detector/erasure_detector.py
import numpy as np

class ErasureDetector:
    def __init__(self, sr=16000, clip_thresh=0.97, onset_db_thresh=20,
                 kurtosis_thresh=10, min_hold_ms=5, recovery_db_thresh=12, max_hold_ms=80):
        self.sr = sr
        self.clip_thresh = clip_thresh
        self.onset_db_thresh = onset_db_thresh
        self.kurtosis_thresh = kurtosis_thresh
        self.min_hold = int(min_hold_ms * sr / 1000)
        self.recovery_db_thresh = recovery_db_thresh
        self.max_hold = int(max_hold_ms * sr / 1000)
        self.block_len = int(1 * sr / 1000)
        self.slow_mean = 1e-6
        self.hold_counter = 0

    def process_block(self, x_block, ref_block=None):
        clipped = np.mean(np.abs(x_block) > self.clip_thresh) > 0
        flat_top = self._has_flat_top_run(x_block)
        block_energy = np.mean(x_block ** 2) + 1e-12
        onset_db = 10 * np.log10(block_energy / (self.slow_mean + 1e-12))
        onset_trigger = onset_db > self.onset_db_thresh
        impulsive = self._kurtosis(x_block) > self.kurtosis_thresh

        external = False
        if ref_block is not None:
            p = 10 * np.log10(np.mean(x_block ** 2) + 1e-12)
            r = 10 * np.log10(np.mean(ref_block ** 2) + 1e-12)
            external = abs(p - r) < 6

        trigger = clipped or flat_top or (onset_trigger and (impulsive or external))

        if trigger:
            self.hold_counter = self.min_hold
        elif self.hold_counter > 0:
            recovering = onset_db > self.recovery_db_thresh
            if recovering and self.hold_counter < self.max_hold:
                self.hold_counter = min(self.hold_counter + self.block_len, self.max_hold)
            else:
                self.hold_counter = max(0, self.hold_counter - self.block_len)

        if self.hold_counter == 0:
            alpha = self.block_len / (0.1 * self.sr)
            self.slow_mean = (1 - alpha) * self.slow_mean + alpha * block_energy

        return self.hold_counter > 0

    @staticmethod
    def _kurtosis(x):
        x = x - np.mean(x)
        std = np.std(x) + 1e-12
        return np.mean((x / std) ** 4) - 3

    @staticmethod
    def _has_flat_top_run(x, min_run=3, thresh=0.97):
        run = 0
        for v in np.abs(x) > thresh:
            run = run + 1 if v else 0
            if run >= min_run:
                return True
        return False
```

**Unit test it yourself, don't wait for real integration:**

```python
# tests/test_detector.py
import numpy as np
from src.detector.erasure_detector import ErasureDetector

def synth_click(length=160, click_at=80, click_len=10):
    x = np.random.randn(length) * 0.01   # quiet "speech"
    x[click_at:click_at+click_len] = 0.99  # clipped click
    return x

det = ErasureDetector()
# feed 200 blocks of quiet speech first (warm up slow_mean), then a clicky one
for _ in range(100):
    det.process_block(np.random.randn(160) * 0.05)
result = det.process_block(synth_click())
print("Detected:", result)   # should print True
```

### Step B2 — Build the gate (Days 3–4)

```python
# src/gate/deterministic_gate.py
import numpy as np

def apply_gate(x, erasure_mask, ramp_ms=0.5, sr=16000):
    ramp_len = int(ramp_ms * sr / 1000)
    gated = x.copy()
    gated[erasure_mask.astype(bool)] = 0.0
    edges = np.diff(erasure_mask.astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
    for s in starts:
        lo, hi = max(0, s - ramp_len), s
        gated[lo:hi] *= (1 - ramp[:hi - lo])
    for e in ends:
        lo, hi = e, min(len(x), e + ramp_len)
        gated[lo:hi] *= ramp[:hi - lo]
    return gated
```

### Step B3 — Build the limiter (Day 4)

```python
# src/gate/limiter.py
import numpy as np

def apply_limiter(x, ceiling_dbfs=-3, sr=16000, lookahead_ms=1, release_ms=50):
    ceiling = 10 ** (ceiling_dbfs / 20)
    lookahead = int(lookahead_ms * sr / 1000)
    gain = np.ones_like(x)
    for i in range(len(x)):
        window = x[i:i + lookahead]
        peak = np.max(np.abs(window)) if len(window) else abs(x[i])
        if peak > ceiling:
            gain[i] = ceiling / (peak + 1e-9)
    release_samples = int(release_ms * sr / 1000)
    for i in range(1, len(gain)):
        if gain[i] > gain[i - 1]:
            gain[i] = gain[i - 1] + (gain[i] - gain[i - 1]) / release_samples
    return x * gain
```

Write a quick test: feed it a signal with a deliberate spike above the ceiling and confirm the output never exceeds it. **This block alone is a great "safety guarantee" demo slide** — you can show it working even before anything else in the pipeline exists.

### Step B4 — Real-time audio I/O skeleton (Days 5–7)

Build this as a **passthrough first** (mic → speaker, no processing) so you have a working live-audio rig early, then plug in the pipeline once Sync Point 1 happens.

```python
# src/realtime/passthrough_test.py
import sounddevice as sd

SR, BLOCK = 16000, 160

def callback(indata, outdata, frames, time_info, status):
    outdata[:, 0] = indata[:, 0]   # pure passthrough, proves your I/O rig works

with sd.Stream(samplerate=SR, blocksize=BLOCK, channels=2, callback=callback, dtype='float32'):
    print("Passthrough running — mic to speaker. Ctrl+C to stop.")
    input()
```

Get this working with your real mic + headphones hardware early — audio driver issues are the #1 time-sink teams hit right before a demo, so surface them in week 1, not week 11.

### Step B5 onward

Once your three pieces (detector, gate, limiter) each work standalone, you're ready for **Sync Point 1**.

---

## 4. Workstream C (Person C) — GTCRN Enhancer, LPC Concealment, Training, Evaluation

### Step C1 — Get GTCRN running on a single file (Days 1–3)

1. Clone the official GTCRN (ICASSP 2024) reference repo from GitHub.
2. Confirm its checkpoint license permits your use — note this in your references doc.
3. Check its native STFT config (FFT size, hop, sample rate) — **adopt its framing rather than forcing a custom one**, this is the lower-risk integration path.
4. Run inference on one plain noisy `.wav` file using their example script, just to prove the environment works before you touch your own data.

### Step C2 — Wrap it to match the team interface (Days 3–4)

```python
# src/enhancer/gtcrn_wrapper.py
import torch

class GTCRNWrapper:
    def __init__(self, checkpoint_path, device="cpu"):
        from gtcrn_repo.model import GTCRN   # adjust import to match the cloned repo
        self.model = GTCRN()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def enhance(self, gated_waveform, state=None):
        x = torch.from_numpy(gated_waveform).float().unsqueeze(0).to(self.device)
        enhanced, new_state = self.model.forward_streaming(x, state)
        return enhanced.squeeze(0).cpu().numpy(), new_state
```

### Step C3 — Build the LPC concealment module (Days 5–7, parallel to training setup)

```python
# src/concealment/lpc_fill.py
import numpy as np, librosa
from scipy.signal import lfilter

def lpc_fill_gap(enhanced_history, gap_len, sr=16000, order=16, decay_db_per_ms=1.0):
    a = librosa.lpc(enhanced_history, order=order)
    autocorr_peak = np.max(np.correlate(enhanced_history, enhanced_history, mode='full'))
    is_voiced = autocorr_peak > 0.35   # tune against real data

    if is_voiced:
        pitch_period = estimate_pitch_period(enhanced_history, sr)
        n_repeats = int(np.ceil(gap_len / pitch_period))
        excitation = np.tile(enhanced_history[-pitch_period:], n_repeats)[:gap_len]
    else:
        excitation = np.random.randn(gap_len) * np.std(enhanced_history[-order:])

    filled = lfilter([1], np.concatenate([[1], -a[1:]]), excitation)

    decay_start = int(20 * sr / 1000)
    if gap_len > decay_start:
        t_ms = np.arange(gap_len - decay_start) / sr * 1000
        filled[decay_start:] *= 10 ** (-decay_db_per_ms * t_ms / 20)
    return filled

def estimate_pitch_period(signal, sr, fmin=80, fmax=400):
    corr = np.correlate(signal, signal, mode='full')[len(signal)-1:]
    min_lag, max_lag = int(sr / fmax), int(sr / fmin)
    return np.argmax(corr[min_lag:max_lag]) + min_lag

def apply_horizon_cap(filled_signal, sr, cap_ms=40, fade_db_per_frame=6):
    cap_samples = int(cap_ms * sr / 1000)
    if len(filled_signal) <= cap_samples:
        return filled_signal
    out = filled_signal.copy()
    excess = len(filled_signal) - cap_samples
    fade = 10 ** (-fade_db_per_frame * np.arange(excess) / 20 / (0.01 * sr))
    out[cap_samples:] *= fade
    return out
```

### Step C4 — Training loop with MLflow logging (Week 2, once Person A's first data batch lands)

```python
# src/train/finetune_gtcrn.py
import torch, mlflow
from src.tracking.mlflow_setup import init_tracking

def si_sdr_loss(estimate, target, eps=1e-8):
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    s_target = (torch.sum(estimate * target) / (torch.sum(target ** 2) + eps)) * target
    e_noise = estimate - s_target
    return -10 * torch.log10(torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + eps) + eps)

def masked_spectral_loss(est_spec, target_spec, reliability_mask):
    diff = (torch.abs(est_spec) - torch.abs(target_spec)) ** 2
    return (diff * reliability_mask).sum() / (reliability_mask.sum() + 1e-8)

def train_run(model, dataloader, optimizer, device, config, run_name):
    init_tracking()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0.0
            for noisy, clean, reliability_mask in dataloader:
                noisy, clean, reliability_mask = noisy.to(device), clean.to(device), reliability_mask.to(device)
                optimizer.zero_grad()
                estimate = model(noisy)
                loss_time = si_sdr_loss(estimate, clean)
                est_spec = torch.stft(estimate, n_fft=512, hop_length=256, return_complex=True)
                clean_spec = torch.stft(clean, n_fft=512, hop_length=256, return_complex=True)
                loss_spec = masked_spectral_loss(est_spec, clean_spec, reliability_mask)
                loss = loss_time + 0.5 * loss_spec
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.pytorch.log_model(model, "model")
    return model
```

### Step C5 — Evaluation harness (Week 2–3, needed by Sync Point 2)

```python
# src/eval/event_locked_eval.py
import numpy as np
from pesq import pesq
from pystoi import stoi

def si_sdr_numpy(estimate, target, eps=1e-8):
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    s_target = (np.dot(estimate, target) / (np.sum(target ** 2) + eps)) * target
    e_noise = estimate - s_target
    return 10 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps))

def utterance_metrics(clean, enhanced, sr=16000):
    return {"pesq": pesq(sr, clean, enhanced, 'wb'),
            "stoi": stoi(clean, enhanced, sr, extended=False),
            "sisdr": si_sdr_numpy(enhanced, clean)}

def click_leak_rate(enhanced_window, clean_window, thresh_db=6):
    speech_peak = np.max(np.abs(clean_window)) + 1e-9
    out_peak = np.max(np.abs(enhanced_window))
    return 1 if 20 * np.log10(out_peak / speech_peak) > thresh_db else 0

def gap_duration_ms(enhanced_window, sr, floor_db=-40):
    floor = 10 ** (floor_db / 20)
    below = np.abs(enhanced_window) < floor
    max_run = cur_run = 0
    for v in below:
        cur_run = cur_run + 1 if v else 0
        max_run = max(max_run, cur_run)
    return max_run / sr * 1000

def event_locked_metrics(clean, enhanced, event_timestamps_sec, sr=16000,
                          window_before_ms=100, window_after_ms=500):
    results = []
    before, after = int(window_before_ms * sr / 1000), int(window_after_ms * sr / 1000)
    for t in event_timestamps_sec:
        center = int(t * sr)
        lo, hi = max(0, center - before), min(len(clean), center + after)
        c_win, e_win = clean[lo:hi], enhanced[lo:hi]
        if len(c_win) < sr * 0.05:
            continue
        results.append({"event_time": t, "sisdr": si_sdr_numpy(e_win, c_win),
                         "click_leak": click_leak_rate(e_win, c_win),
                         "gap_ms": gap_duration_ms(e_win, sr)})
    return results

def log_eval_to_mlflow(run_id, utt_metrics, event_results):
    import mlflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(utt_metrics)
        mlflow.log_metrics({
            "click_leak_rate": sum(r["click_leak"] for r in event_results) / len(event_results),
            "avg_gap_ms": sum(r["gap_ms"] for r in event_results) / len(event_results),
            "event_locked_sisdr": sum(r["sisdr"] for r in event_results) / len(event_results),
        })
```

---

## 5. SYNC POINT 1 — end of Week 2 (all three together, ~2 hours)

**Goal: prove the pipeline runs end to end, even badly.**

1. Person A hands over: a folder of 50+ simulated triplets.
2. Person B hands over: working `detector.py`, `gate.py`, `limiter.py` — each already unit-tested alone.
3. Person C hands over: a GTCRN wrapper that runs inference, and the LPC module.
4. Together, write `src/pipeline/full_pipeline.py` — this is a **joint session**, not a solo task, because it's where all three interfaces meet:

```python
# src/pipeline/full_pipeline.py
import numpy as np
from src.detector.erasure_detector import ErasureDetector
from src.gate.deterministic_gate import apply_gate
from src.gate.limiter import apply_limiter
from src.enhancer.gtcrn_wrapper import GTCRNWrapper
from src.concealment.lpc_fill import lpc_fill_gap, apply_horizon_cap

class ANCPipeline:
    def __init__(self, gtcrn_checkpoint, sr=16000):
        self.sr = sr
        self.detector = ErasureDetector(sr=sr)
        self.enhancer = GTCRNWrapper(gtcrn_checkpoint)
        self.enhanced_history = np.zeros(int(0.03 * sr))
        self.gru_state = None

    def process_frame(self, primary_block, reference_block=None):
        erased = self.detector.process_block(primary_block, reference_block)
        erasure_mask = np.full_like(primary_block, float(erased))
        gated = apply_gate(primary_block, erasure_mask, sr=self.sr)
        enhanced, self.gru_state = self.enhancer.enhance(gated, state=self.gru_state)

        reliability = 1.0 - np.mean(erasure_mask)
        if reliability < 1.0:
            filled = apply_horizon_cap(
                lpc_fill_gap(self.enhanced_history, len(primary_block), sr=self.sr), sr=self.sr)
        else:
            filled = enhanced

        output = reliability * enhanced + (1 - reliability) * filled
        output = apply_limiter(output, ceiling_dbfs=-3, sr=self.sr)

        self.enhanced_history = np.roll(self.enhanced_history, -len(output))
        self.enhanced_history[-len(output):] = output
        return output
```

5. Run it on one of Person A's simulated files end to end. **It doesn't need to sound good yet** — you're checking that shapes match, nothing crashes, and the erasure mask actually flags the impulse. This is your first "it's alive" milestone.
6. Merge everyone's branch into `main` after this works.

---

## 6. Weeks 3–6 — back to parallel work, now against the real pipeline

- **Person A:** generate a bigger, better dataset batch; start collecting real clipped recordings (clap/balloon at raised mic gain) for the later real-world test tier.
- **Person B:** wire the full pipeline into the real-time audio I/O skeleton (replace the passthrough with `pipeline.process_frame()`); start measuring real-time factor and latency.
- **Person C:** run the ablation matrix as separate MLflow runs (see table below), fine-tune GTCRN, tune the LPC voiced/unvoiced threshold and horizon cap against real data.

**Ablation runs — log every row to MLflow as its own run:**

| Run name | detector | calibrated data | state protection | LPC | reference mic |
|---|---|---|---|---|---|
| `row_i_baseline` | ✗ | ✗ | ✗ | ✗ | ✗ |
| `row_ii_calibrated_data` | ✗ | ✓ | ✗ | ✗ | ✗ |
| `row_iii_detector_gate` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `row_iv_state_protection` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `row_v_lpc` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `row_vi_reference_mic` | ✓ | ✓ | ✓ | ✓ | ✓ |

```python
ablation_configs = [
    {"name": "row_i_baseline",          "detector_enabled": False, "calibrated_data": False, "state_protection": False, "lpc_concealment": False, "reference_mic": False, "lr": 1e-4, "epochs": 20},
    {"name": "row_ii_calibrated_data",  "detector_enabled": False, "calibrated_data": True,  "state_protection": False, "lpc_concealment": False, "reference_mic": False, "lr": 1e-4, "epochs": 20},
    {"name": "row_iii_detector_gate",   "detector_enabled": True,  "calibrated_data": True,  "state_protection": False, "lpc_concealment": False, "reference_mic": False, "lr": 1e-4, "epochs": 20},
    {"name": "row_iv_state_protection", "detector_enabled": True,  "calibrated_data": True,  "state_protection": True,  "lpc_concealment": False, "reference_mic": False, "lr": 1e-4, "epochs": 20},
    {"name": "row_v_lpc",               "detector_enabled": True,  "calibrated_data": True,  "state_protection": True,  "lpc_concealment": True,  "reference_mic": False, "lr": 1e-4, "epochs": 20},
    {"name": "row_vi_reference_mic",    "detector_enabled": True,  "calibrated_data": True,  "state_protection": True,  "lpc_concealment": True,  "reference_mic": True,  "lr": 1e-4, "epochs": 20},
]

for cfg in ablation_configs:
    model = build_model(cfg)   # your own factory function that toggles blocks per config
    train_run(model, dataloader, optimizer, device, cfg, run_name=cfg["name"])
```

Once all six are logged, open the MLflow UI, select all six runs, and use its built-in comparison view to generate the chart/table for your evaluation slide — no manual spreadsheet needed.

---

## 7. SYNC POINT 2 — end of Week 6 (all three together, ~2–3 hours)

1. Person B's real-time rig now runs the *actual trained* model from Person C, not a placeholder.
2. Person A's real clipped recordings get run through the pipeline for the first time.
3. Together, watch the waveform of a real clap/balloon test through the pipeline and sanity-check: did the detector fire? Did the gap get filled smoothly? Did the limiter ever get triggered unexpectedly?
4. Fix whatever breaks — this is normal and expected at this checkpoint.

---

## 8. Weeks 7–9 — hardening + demo prep (mostly parallel, daily syncs)

- **Person A:** finalize the real-recording test tier; help build the "3 waveforms side by side" visualization for the demo.
- **Person B:** measure real-time factor / latency on your target device (laptop CPU is fine; Jetson/Pi only if you have one); build the live demo script with on-screen detector indicator.
- **Person C:** finalize the model checkpoint, run the small listening test (5–10 people, "did you hear a click/gap?"), run Whisper WER before/after.

```python
# src/realtime/live_demo.py
import sounddevice as sd
import numpy as np
from src.pipeline.full_pipeline import ANCPipeline

SR, BLOCK_SIZE = 16000, 160
pipeline = ANCPipeline(gtcrn_checkpoint="checkpoints/gtcrn_finetuned.pt", sr=SR)

def audio_callback(indata, outdata, frames, time_info, status):
    primary = indata[:, 0]
    reference = indata[:, 1] if indata.shape[1] > 1 else None
    outdata[:, 0] = pipeline.process_frame(primary, reference)

with sd.Stream(samplerate=SR, blocksize=BLOCK_SIZE, channels=2,
               callback=audio_callback, dtype='float32'):
    print("Running live demo. Press Ctrl+C to stop.")
    input()
```

**Demo choreography to rehearse together at least 3 times before presenting:**
1. Raise mic gain so a clap/balloon pop genuinely clips the ADC.
2. Someone reads a sentence; mid-sentence, the pop fires.
3. Show three outputs side by side: raw (click+hole) vs. plain GTCRN (leaks or holes) vs. full pipeline (clean).
4. Record this as a backup video in case live hardware misbehaves.

---

## 9. SYNC POINT 3 — Week 9–10 (all three together)

Full run-through of the pitch + live demo, timed, with one person playing "skeptical judge" and asking the hard questions:
- "Isn't this just PLC plus a denoiser?"
- "How do you know your synthetic blast matches a real one?"
- "Why not just clip the output?"

Assign each answer to whoever built that part — it reads as more credible when the person who built it explains it.

---

## 10. Weeks 10–12 — final polish

- Confirm every number on your slides is either measured (from MLflow) or explicitly labeled "target."
- Add the "known limitation" slide about sustained automatic fire before a judge can ask.
- Finalize references with real, checkable sources.
- Rehearse the full pitch end to end at least twice as a team.

---

## Quick-reference: who owns what, always

| If it breaks... | Ask |
|---|---|
| Data generation, dataset issues, MLflow server down | Person A |
| Detector false-triggers, gate artifacts, limiter clipping wrong, audio I/O glitches | Person B |
| Model quality, training crashes, LPC sounds unnatural, evaluation numbers | Person C |
| Pipeline integration / "it worked alone but not together" | All three, together — this is always a sync issue, not one person's bug |
