"""Values shared by the fixture writers (no side effects on import)."""


def random_values(n=64):
    """64 float32 values in [-8, 8) from a fixed linear congruential sequence;
    integer arithmetic only, so every implementation gets the same values
    (libalice: residual::tests::random_values)."""
    x = 0x2545F491
    out = []
    for _ in range(n):
        x = (x * 1664525 + 1013904223) % 2 ** 32
        out.append((x >> 8) / 2 ** 20 - 8.0)
    return out
