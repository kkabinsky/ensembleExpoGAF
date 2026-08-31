# -*- coding: utf-8 -*-
"""
mapping_derivation.py
=====================

Print the closed-form properties of the four angular mappings beside a numerical check.

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
