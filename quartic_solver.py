# quartic_solver.py
import cmath
import math
from cubic_solver import solve_cubic, solve_quadratic, kth_root, _to_real_if_good, _TOL

def _poly_mon(x, b, c, d, e):
    return x**4 + b*x**3 + c*x**2 + d*x + e

def _poly_and_deriv_mon(x, b, c, d, e):
    p = x**4 + b*x**3 + c*x**2 + d*x + e
    dp = 4*x**3 + 3*b*x**2 + 2*c*x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=16):
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    # if near-real and residual small, return Python float
    if isinstance(x, complex) and abs(x.imag) < 1e-10:
        if abs(_poly_mon(x.real, b, c, d, e)) < 1e-8:
            return float(x.real)
    return x

def _positive_real_sqrt_via_log(x_real):
    if x_real == 0.0:
        return 0.0
    return math.exp(0.5 * math.log(x_real))

def solve_quartic(a, b, c, d, e):
    """Solve ax^4 + bx^3 + cx^2 + dx + e = 0; returns exactly 4 roots (floats when real)."""
    if abs(a) < _TOL:
        return solve_cubic(b, c, d, e)

    # normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic y: y^4 + p y^2 + q y + r = 0 with x = y - b/4
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def monic(x):
        return x**4 + b*x**3 + c*x**2 + d*x + e

    candidate_sets = []

    # === BIQUADRATIC PATH (explicit quadratic solving for t) ===
    if abs(q) < 1e-14:
        # Solve t^2 + p t + r = 0 explicitly using kth_root on discriminant
        # disc = p^2 - 4 r
        disc = p * p - 4.0 * r
        # principal sqrt (branch 0) and its negative correspond to two roots
        s0 = kth_root(disc, 2, 0)
        # t1,t2 (note using +/- s0 yields both)
        t1 = (-p + s0) / 2.0
        t2 = (-p - s0) / 2.0
        t_vals = [t1, t2]

        # Always enumerate both sqrt branches for each t (so multiplicities preserved)
        for t in t_vals:
            # if t is nearly real and nonnegative, compute real positive sqrt via log-exp
            if isinstance(t, complex) and abs(t.imag) < 1e-10:
                tr = float(t.real)
                if tr >= -1e-12:
                    tr = max(tr, 0.0)
                    s = _positive_real_sqrt_via_log(tr)
                    ypos = s
                    yneg = -s
                    roots = [ypos - b/4.0, -ypos - b/4.0, yneg - b/4.0, -yneg - b/4.0]
                    roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                    score = sum(abs(monic(rr))**2 for rr in roots)
                    candidate_sets.append((score, roots))
                    continue
            # general complex t: enumerate sqrt branches via kth_root
            for sb in (0,1):
                s = kth_root(t, 2, sb)
                ypos = s
                yneg = -s
                roots = [ypos - b/4.0, -ypos - b/4.0, yneg - b/4.0, -yneg - b/4.0]
                roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                score = sum(abs(monic(rr))**2 for rr in roots)
                candidate_sets.append((score, roots))

    else:
        # === GENERAL FERRARI PATH with resolvent cubic and branch enumeration ===
        zvals = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        for z in zvals:
            zc = complex(z)
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
                            roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                            score = sum(abs(monic(rr))**2 for rr in roots)
                            candidate_sets.append((score, roots))
                else:
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
                            roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                            score = sum(abs(monic(rr))**2 for rr in roots)
                            candidate_sets.append((score, roots))

    # If nothing found, fallback
    if not candidate_sets:
        return [0.0, 0.0, 0.0, 0.0]

    # pick best by score
    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final cast: near-real -> float only when residual is tiny
    final = []
    for r in best_roots:
        if isinstance(r, complex) and abs(r.imag) < 1e-9:
            val = monic(r.real)
            if abs(val) < 1e-8:
                final.append(float(r.real))
                continue
        final.append(r)

    # ensure exactly 4 entries
    while len(final) < 4:
        final.append(final[-1])
    final = final[:4]
    return final
