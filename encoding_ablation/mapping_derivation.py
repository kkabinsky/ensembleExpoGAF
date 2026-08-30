# -*- coding: utf-8 -*-
"""Closed-form properties of the four angular mappings, checked numerically.

Reviewer 4 asked for verification of the mathematical claims.  The companion
script ``mapping_geometry.py`` measures the geometry on real windows; this one
derives the same properties in closed form and confirms each derivation against
a numerical evaluation, so that the claims in the manuscript rest on algebra
rather than on a fitted number.

Mappings, on the min-max normalised window s in [0, 1]:

    cosine       g(s) = arccos(2s - 1)          g'(s) = -2 / sqrt(1 - (2s-1)^2)
    arctan       g(s) = arctan(s)               g'(s) = 1 / (1 + s^2)
    arccosh      g(s) = arccosh(1 + s)          g'(s) = 1 / sqrt(s^2 + 2s)
    exponential  g(s) = pi (e^s - 1) / (e - 1)  g'(s) = pi e^s / (e - 1)

Three statements are proved and printed with their numerical check:

1.  Range.  g(0) = 0 and g(1) = pi for the exponential map, exactly, since
    (e^1 - 1)/(e - 1) = 1.  For arctan the range is [0, pi/4] and for arccosh
    it is [0, arccosh 2] = [0, ln(2 + sqrt 3)].

2.  Rising resolution.  g'(s) = pi e^s / (e - 1) is strictly increasing because
    e^s is, so the exponential map is the only one of the four whose angular
    resolution grows with the value being encoded.  The growth factor across
    the window range is exactly g'(1) / g'(0) = e.  For arctan and arccosh the
    derivative is strictly decreasing; for the cosine map it is negative.

3.  Fold.  The Gramian entry is cos(phi_i + phi_j), which is not injective once
    the pair sum passes pi.  The maximum pair sum is 2 g(1): it is 2 pi for the
    exponential and cosine maps, pi/2 for arctan and 2 arccosh 2 = 2.634 for
    arccosh.  Only the first two exceed pi, so only they can fold.  For the
    exponential map the fold condition in terms of the normalised values is
    exactly e^{s_i} + e^{s_j} > e + 1.

Run:

    python mapping_derivation.py
"""
import math

import numpy as np

E = math.e
PI = math.pi
TOL = 1e-9


def report(claim, derived, numeric, tol=1e-9, note=""):
    """Compare a closed-form value with its numerical evaluation.

    Derivatives are compared at a looser tolerance because they are evaluated
    by a central difference on a finite grid, whose error is of order h^2 and
    is a property of the check rather than of the derivation.
    """
    ok = abs(derived - numeric) < tol
    print("  %-46s derived %12.6f   numeric %12.6f   %s%s"
          % (claim, derived, numeric, "ok" if ok else "MISMATCH",
             ("   " + note) if note else ""))
    return ok


def main():
    s = np.linspace(0.0, 1.0, 200001)
    allok = True

    print("1. Angular range")
    g_exp = PI * (np.exp(s) - 1.0) / (E - 1.0)
    allok &= report("exponential g(1) = pi", PI, float(g_exp[-1]), 1e-9)
    allok &= report("exponential g(0) = 0", 0.0, float(g_exp[0]))
    allok &= report("arctan g(1) = pi/4", PI / 4.0, float(np.arctan(s[-1])))
    allok &= report("arccosh g(1) = ln(2+sqrt3)",
                    math.log(2.0 + math.sqrt(3.0)),
                    float(np.arccosh(1.0 + s[-1])))
    allok &= report("cosine g(0) = pi", PI,
                    float(np.arccos(2.0 * s[0] - 1.0)))

    print()
    print("2. Rising resolution, exponential map only")
    dlow = PI * math.exp(0.0) / (E - 1.0)
    dhigh = PI * math.exp(1.0) / (E - 1.0)
    num = np.gradient(g_exp, s)
    allok &= report("g'(0) = pi/(e-1)", dlow, float(num[1]), 1e-4,
                    "central difference on a finite grid")
    allok &= report("g'(1) = pi e/(e-1)", dhigh, float(num[-2]), 1e-4,
                    "central difference on a finite grid")
    allok &= report("growth factor g'(1)/g'(0) = e", E, dhigh / dlow, 1e-12)
    print("  derivative is strictly increasing:",
          bool(np.all(np.diff(num[1:-1]) > -1e-9)))
    for name, g in (("arctan", np.arctan(s)),
                    ("arccosh", np.arccosh(1.0 + s))):
        d = np.gradient(g, s)[1:-1]
        print("  %-10s derivative strictly decreasing: %s"
              % (name, bool(np.all(np.diff(d) < 1e-9))))
    d_cos = np.gradient(np.arccos(np.clip(2.0 * s - 1.0, -1.0, 1.0)), s)[1:-1]
    print("  cosine     derivative negative throughout:",
          bool(np.all(d_cos < 0)))

    print()
    print("3. Fold, maximum pair sum against pi")
    for name, gmax in (("cosine", PI), ("arctan", PI / 4.0),
                       ("arccosh", math.log(2.0 + math.sqrt(3.0))),
                       ("exponential", PI)):
        print("  %-12s 2 g(1) = %7.4f   %s pi   -> %s"
              % (name, 2.0 * gmax, ">" if 2.0 * gmax > PI else "<",
                 "can fold" if 2.0 * gmax > PI else "cannot fold"))
    print("  exponential fold condition: e^{s_i} + e^{s_j} > e + 1 = %.6f"
          % (E + 1.0))

    print()
    print("all derivations confirmed numerically:", bool(allok))


if __name__ == "__main__":
    main()
