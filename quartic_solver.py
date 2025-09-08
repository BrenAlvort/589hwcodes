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
    if isinstance(x, complex) and abs(x.imag) < 1e-9:
        if abs(_poly_mon(x.real, b, c, d, e)) < 1e-8:
            return float(x.real)
    return x

def _positive_real_sqrt_via_log(x_real):
    if x_real == 0.0:
        return 0.0
    return math.exp(0.5 * math.log(x_real))

def solve_quartic(a, b, c, d, e):
    """Solve ax^4 + bx^3 + cx^2 + dx + e = 0; always returns 4 roots (float when real)."""
    if abs(a) < _TOL:
        return solve_cubic(b, c, d, e)

    # normalize
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic coefficients
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def monic(x):
        return x**4 + b*x**3 + c*x**2 + d*x + e

    candidate_sets = []

    # --- biquadratic: use robust quadratic to get exactly two t-values (t = y^2) ---
    if abs(q) < 1e-14:
        tvals = solve_quadratic(1.0, p, r)  # returns two roots (floats when real)
        # ensure exactly two returned; solve_quadratic returns [] or one or two; handle robustly
        if not tvals:
            tvals = [0.0, 0.0]
        elif len(tvals) == 1:
            tvals = [tvals[0], tvals[0]]
        # For each t enumerate sqrt branches; for real nonnegative t use positive real sqrt via log-exp
        for t in tvals:
            # if t is real number
            if isinstance(t, (float, int)):
                tr = float(t)
                if tr >= -1e-12:
                    tr = max(tr, 0.0)
                    s = _positive_real_sqrt_via_log(tr)
                    ys = [s, -s]
                    roots = [ys[0] - b/4.0, -ys[0] - b/4.0, ys[1] - b/4.0, -ys[1] - b/4.0]
                    roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                    score = sum(abs(monic(rr))**2 for rr in roots)
                    candidate_sets.append((score, roots))
                    continue
            # otherwise enumerate complex sqrt branches
            for sbranch in (0,1):
                s = kth_root(t, 2, sbranch)
                ys = [s, -s]
                roots = [ys[0] - b/4.0, -ys[0] - b/4.0, ys[1] - b/4.0, -ys[1] - b/4.0]
                roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                score = sum(abs(monic(rr))**2 for rr in roots)
                candidate_sets.append((score, roots))
    else:
        # --- general Ferrari via resolvent cubic ---
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

    if not candidate_sets:
        return [0.0, 0.0, 0.0, 0.0]

    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final map: near-real complex -> float when residual small; keep complex otherwise
    final = []
    for r in best_roots:
        if isinstance(r, complex) and abs(r.imag) < 1e-9:
            val = monic(r.real)
            if abs(val) < 1e-7:
                final.append(float(r.real))
                continue
        final.append(r)

    # ensure exactly 4 roots
    while len(final) < 4:
        final.append(final[-1])
    final = final[:4]
    return final
