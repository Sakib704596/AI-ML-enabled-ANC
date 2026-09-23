import numpy as np


def apply_gate(x, erasure_mask, ramp_ms=0.5, sr=16000):
    """
    Mutes (zeros out) any sample flagged as 'erased' by the detector,
    with a short smooth fade at the edges so muting doesn't create its
    own click.
    """
    ramp_len = int(ramp_ms * sr / 1000)
    gated = x.copy()

    gated[erasure_mask.astype(bool)] = 0.0

    edges = np.diff(erasure_mask.astype(int))
    starts = np.where(edges == 1)[0]   # last sample BEFORE erasure begins
    ends = np.where(edges == -1)[0]    # last sample OF the erasure

    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))

    # fade OUT: must END exactly at the boundary sample, reaching full
    # attenuation (0) right where the hard-mute takes over
    for s in starts:
        hi = s + 1                          # include the boundary sample
        lo = max(0, hi - ramp_len)
        seg_len = hi - lo
        gated[lo:hi] *= (1 - ramp[-seg_len:])  # use the END of the ramp curve

    # fade IN: must START exactly at the boundary sample, rising from 0
    for e in ends:
        lo = e + 1                          # first sample after erasure
        hi = min(len(x), lo + ramp_len)
        seg_len = hi - lo
        gated[lo:hi] *= ramp[:seg_len]      # use the START of the ramp curve

    return gated