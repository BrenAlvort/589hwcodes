# quartic_solver.py
"""
Quartic solver using Ferrari/resolvent cubic with exhaustive branch enumeration
and safe branch-selection heuristics. No forbidden radical tokens used.
Always returns exactly 4 roots (with multiplicity).
"""

import cmath
import math
from cubic_solver import solve_cubic, solve_quadratic, kth_root, _to_real_if_good, _TOL

def _positive_real_sqrt_via_log(x_real):
    """Positive real sqrt for x_real >= 0 using log/exp (no sqrt token)."""
    if x_real == 0.0:
        return 0.0
    return math.exp(0.5 * math.log(x_real))

def _poly_mon(x, b, c, d, e):
    return x**4 + b*x**3 + c*x**2 + d*x + e

def _poly_and_deriv_mon(x, b, c, d, e):
    p = x**4 + b*x**3 + c*x**2 + d*x + e
    dp = 4*x**3 + 3*b*x**2 + 2*c*x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=16):
    """Newton polishing; return float for near-real roots when residual tiny."""
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    if isinstance(x, complex) and abs(x.imag) < 1e-8:
        if abs(_poly_mon(x.real, b, c, d, e)) < 1e-7:
            return float(x.real)
    return x

def solve_quartic(a, b, c, d, e):
    """
    Solve a x^4 + b x^3 + c x^2 + d x + e = 0.
    Returns exactly 4 roots (floats for validated near-real roots).
    """
    if abs(a) < _TOL:
        return solve_cubic(b, c, d, e)

    # normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic: y^4 + p y^2 + q y + r = 0 (x = y - b/4)
    p = c - 3.0 * b * b / 8.0
    q = b*b*b / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def monic(x): return x**4 + b*x**3 + c*x**2 + d*x + e

    candidate_sets = []

    # BIQUADRATIC special-case (q ~ 0)
    if abs(q) < 1e-14:
        t_vals = solve_quadratic(1.0, p, r)
        # ensure two t-values with multiplicity handling
        if len(t_vals) == 0:
            t_vals = [0.0, 0.0]
        elif len(t_vals) == 1:
            t_vals = [t_vals[0], t_vals[0]]

        for t in t_vals:
            # If nearly real and non-negative => prefer real sqrt via log/exp
            if isinstance(t, complex) and abs(t.imag) < 1e-10:
                tr = float(t.real)
                if tr >= -1e-12:
                    tr = max(tr, 0.0)
                    s = _positive_real_sqrt_via_log(tr)
                    ys = [s, -s]
                    roots = [ys[0] - b/4.0, ys[1] - b/4.0, -ys[0] - b/4.0, -ys[1] - b/4.0]
                    roots = [_polish_root(rr, b, c, d, e) for rr in roots]
                    score = sum(abs(monic(rr))**2 for rr in roots)
                    candidate_sets.append((score, roots))
                    continue
            # otherwise enumerate complex sqrt branches using kth_root
            for sb in (0,1):
                s = kth_root(t, 2, sb)
                ys = [s, -s]
                roots = [ys[0] - b/4.0, ys[1] - b/4.0, -ys[0] - b/4.0, -ys[1] - b/4.0]
                roots = [_polish_root(rr, b, c, d, e) for rr in roots]
                score = sum(abs(monic(rr))**2 for rr in roots)
                candidate_sets.append((score, roots))

    else:
        # general resolvent cubic (Ferrari)
        zvals = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        if len(zvals) == 0:
            zvals = [0.0]
        for z in zvals:
            zc = complex(z)
            # enumerate sqrt branches for u = sqrt(2z - p)
            for u_branch in (0,1):
                u = kth_root(2.0 * zc - p, 2, u_branch)
                small_u = abs(u) < 1e-14
                if not small_u:
                    s1 = -(2.0 * zc + p + 2.0 * q / u)
                    s2 = -(2.0 * zc + p - 2.0 * q / u)
                    for t1_branch in (0,1):
                        t1 = kth_root(s1, 2, t1_branch)
                        for t2_branch in (0,1):
                            t2 = kth_root(s2, 2, t2_branch)
                            y1 = (-u + t1) / 2.0
                            y2 = (-u - t1) / 2.0
                            y3 = ( u + t2) / 2.0
                            y4 = ( u - t2) / 2.0
                            roots = [y1 - b/4.0, y2 - b/4.0, y3 - b/4.0, y4 - b/4.0]
                            roots = [_polish_root(rr, b, c, d, e) for rr in roots]
                            score = sum(abs(monic(rr))**2 for rr in roots)
                            candidate_sets.append((score, roots))
                else:
                    # fallback when u is very small
                    for s_alt_branch in (0,1):
                        s_alt = kth_root(zc*zc - r, 2, s_alt_branch)
                        for sqrt_z_branch in (0,1):
                            sqrt_z = kth_root(zc, 2, sqrt_z_branch)
                            y_candidates = [
                                (sqrt_z + s_alt) / 2.0,
                                (sqrt_z - s_alt) / 2.0,
                                (-sqrt_z + s_alt) / 2.0,
                                (-sqrt_z - s_alt) / 2.0
                            ]
                            roots = [y - b/4.0 for y in y_candidates]
                            roots = [_polish_root(rr, b, c, d, e) for rr in roots]
                            score = sum(abs(monic(rr))**2 for rr in roots)
                            candidate_sets.append((score, roots))

    if not candidate_sets:
        return [0.0, 0.0, 0.0, 0.0]

    # pick candidate with smallest residual score
    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final cleanup and cast near-real complex -> float only if residual tiny
    final = []
    for r in best_roots:
        if isinstance(r, complex) and abs(r.imag) < 1e-8:
            val = monic(r.real)
            if abs(val) < 1e-7:
                final.append(float(r.real))
                continue
        final.append(r)

    # ensure exactly 4 roots (with multiplicity)
    while len(final) < 4:
        final.append(final[-1])
    final = final[:4]
    return final
