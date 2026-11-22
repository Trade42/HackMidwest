"""The few statistics every table and figure uses.

The kit's convention (README.md of the kit, decoders.per_cycle_rate): shot k of X memory is
paired with shot k of Z memory, a pair fails when either fails, and with failure fraction P
over the pairs the logical error rate per cycle is 1 - (1 - P)^(1/12).
"""
import math

from scipy.stats import binomtest

CYCLES = 12
Z95 = 1.959963984540054   # the normal quantile of a two-sided 95% interval


def per_cycle(fraction, cycles=CYCLES):
    """The rate per cycle of a failure fraction: 1 - (1 - P)^(1/cycles)."""
    return 1 - (1 - fraction) ** (1 / cycles)


def per_cycle_rate(failures, shots, cycles=CYCLES):
    """(rate, standard error) per cycle, the same arithmetic as the kit's decoders.per_cycle_rate.

    The kit's error is the rate at P + sigma minus the rate at P, with sigma the binomial
    standard deviation of P. It is 0 when there is no failure, which is why the figures use
    the Wilson interval below.
    """
    total = failures / shots
    sigma = math.sqrt(total * (1 - total) / shots)
    rate = per_cycle(total, cycles)
    return rate, per_cycle(min(total + sigma, 1.0), cycles) - rate


def wilson(failures, shots, z=Z95):
    """The 95% Wilson interval of a failure fraction, as (low, high)."""
    if shots == 0:
        return (0.0, 1.0)
    fraction = failures / shots
    centre = fraction + z * z / (2 * shots)
    half = z * math.sqrt(fraction * (1 - fraction) / shots + z * z / (4 * shots * shots))
    scale = 1 + z * z / shots
    low = 0.0 if failures == 0 else max(0.0, (centre - half) / scale)
    high = 1.0 if failures == shots else min(1.0, (centre + half) / scale)
    return (low, high)


def per_cycle_interval(failures, shots, cycles=CYCLES):
    """The 95% Wilson interval of the pair failure fraction, mapped to a rate per cycle."""
    low, high = wilson(failures, shots)
    return per_cycle(low, cycles), per_cycle(high, cycles)


def paired(reference_failed, row_failed):
    """Two decoders on the same shots: how many pairs only one of them fails.

    Returns a dict with both, reference_only, row_only, and the exact two-sided binomial test
    on the discordant pairs (under "no difference" each discordant pair is a fair coin). This
    is the paired part of the team's evaluation/compare_gain.py.
    """
    reference_failed = reference_failed.astype(bool)
    row_failed = row_failed.astype(bool)
    if reference_failed.shape != row_failed.shape:
        raise ValueError("a paired count needs the same shots for both decoders")
    both = int((reference_failed & row_failed).sum())
    reference_only = int((reference_failed & ~row_failed).sum())
    row_only = int((~reference_failed & row_failed).sum())
    discordant = reference_only + row_only
    p_value = float(binomtest(row_only, discordant).pvalue) if discordant else 1.0
    return {"both": both, "reference_only": reference_only, "row_only": row_only, "p": p_value}


def sci(value, digits=2):
    """1.23e-04 style, for rates."""
    return f"{value:.{digits}e}"


def p_text(p_value):
    """A p-value for a table cell."""
    if p_value >= 0.095:
        return f"{p_value:.2f}"
    if p_value >= 0.00095:
        return f"{p_value:.3f}"
    return f"{p_value:.0e}"
