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