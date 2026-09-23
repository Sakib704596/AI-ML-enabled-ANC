import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "gate"))

from deterministic_gate import apply_gate


def test_erased_region_is_silent():
    """The middle of an erased region should be exactly zero."""
    x = np.ones(160) * 0.5
    mask = np.zeros(160)
    mask[60:100] = 1.0

    gated = apply_gate(x, mask)

    assert gated[80] == 0.0, "Middle of erasure region should be silent!"


def test_no_hard_jump_at_boundaries():
    """The sample right before erasure and the sample right after should
    NOT be at full original volume — they should be partway faded, proving
    there's no instant on/off click."""
    x = np.ones(160) * 0.5
    mask = np.zeros(160)
    mask[60:100] = 1.0

    gated = apply_gate(x, mask)

    assert gated[59] < 0.5, "Sample right before erasure should be fading, not full volume!"
    assert gated[100] < 0.5, "Sample right after erasure should be fading in, not full volume!"


def test_unerased_audio_passes_through_unchanged():
    """If nothing is erased, the gate should not touch the audio at all."""
    x = np.random.randn(160) * 0.3
    mask = np.zeros(160)  # nothing erased

    gated = apply_gate(x, mask)

    assert np.allclose(gated, x), "Gate modified audio even though nothing was erased!"


def test_fade_is_monotonic_going_into_erasure():
    """Volume should smoothly DECREASE approaching the erasure, not jump
    around unpredictably."""
    x = np.ones(160) * 0.5
    mask = np.zeros(160)
    mask[60:100] = 1.0

    gated = apply_gate(x, mask)

    fade_out_region = gated[52:60]  # the 8 samples right before erasure
    diffs = np.diff(fade_out_region)

    assert np.all(diffs <= 1e-9), "Fade-out is not smoothly decreasing!"


if __name__ == "__main__":
    test_erased_region_is_silent()
    print("PASSED: test_erased_region_is_silent")

    test_no_hard_jump_at_boundaries()
    print("PASSED: test_no_hard_jump_at_boundaries")

    test_unerased_audio_passes_through_unchanged()
    print("PASSED: test_unerased_audio_passes_through_unchanged")

    test_fade_is_monotonic_going_into_erasure()
    print("PASSED: test_fade_is_monotonic_going_into_erasure")

    print("\nAll gate tests passed!")