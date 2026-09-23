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
        self.block_len = int(1 * sr / 1000)   # 1ms worth of samples
        self.slow_mean = 1e-6                  # running "how loud is normal" estimate
        self.hold_counter = 0                  # counts down while we're "in an erasure event"
    def process_block(self, x_block, ref_block=None):
        # --- Check 1: clipping ---
        # If any sample's amplitude is above our threshold, the mic likely overloaded
        clipped = np.mean(np.abs(x_block) > self.clip_thresh) > 0

        # --- Check 2: flat-top run ---
        # Several samples in a row pinned near max volume = real clipping,
        # not just one naturally loud speech peak.
        flat_top = self._has_flat_top_run(x_block)

        # --- Check 3: sudden volume jump (onset) ---
        # Compare this block's energy to our running "normal" baseline.
        block_energy = np.mean(x_block ** 2) + 1e-12
        onset_db = 10 * np.log10(block_energy / (self.slow_mean + 1e-12))
        onset_trigger = onset_db > self.onset_db_thresh

        # --- Check 4: kurtosis (how "spiky" is this block?) ---
        # Speech is relatively smooth; a gunshot impulse is extremely peaky.
        impulsive = self._kurtosis(x_block) > self.kurtosis_thresh

        # --- Check 5 (optional): external-source ratio, needs a 2nd mic ---
        # Your own voice is much louder on the primary mic than the reference
        # mic. A gunshot from outside arrives at similar loudness on both.
        external = False
        if ref_block is not None:
            primary_level = 10 * np.log10(np.mean(x_block ** 2) + 1e-12)
            ref_level = 10 * np.log10(np.mean(ref_block ** 2) + 1e-12)
            external = abs(primary_level - ref_level) < 6

        # --- Fusion: combine all checks into one final decision ---
        trigger = clipped or flat_top or (onset_trigger and (impulsive or external))

        # --- Hold logic: once triggered, stay "erased" for a bit ---
        if trigger:
            self.hold_counter = self.min_hold
        elif self.hold_counter > 0:
            recovering = onset_db > self.recovery_db_thresh
            if recovering and self.hold_counter < self.max_hold:
                self.hold_counter = min(self.hold_counter + self.block_len, self.max_hold)
            else:
                self.hold_counter = max(0, self.hold_counter - self.block_len)

        # --- Update our "what's normal" baseline ---
        # Only update while NOT in an erasure event, so a gunshot doesn't
        # corrupt our idea of "normal background loudness."
        if self.hold_counter == 0:
            alpha = self.block_len / (0.1 * self.sr)   # ~100ms smoothing window
            self.slow_mean = (1 - alpha) * self.slow_mean + alpha * block_energy

        return self.hold_counter > 0

    @staticmethod
    def _kurtosis(x):
        # Standard kurtosis formula: measures "peakiness" of the waveform
        # shape, independent of loudness.
        x = x - np.mean(x)
        std = np.std(x) + 1e-12
        return np.mean((x / std) ** 4) - 3

    @staticmethod
    def _has_flat_top_run(x, min_run=3, thresh=0.97):
        # Counts consecutive samples above the clip threshold. 3+ in a row
        # means a real flat-top clip, not just one loud peak.
        run = 0
        for v in np.abs(x) > thresh:
            run = run + 1 if v else 0
            if run >= min_run:
                return True
        return False