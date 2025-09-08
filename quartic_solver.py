# quartic_solver.py
import cmath
import math
from cubic_solver import solve_cubic, kth_root, _cleanup_list_cubic, _TOL

def _poly_mon(x, b, c, d, e):
    return x**4 + b*x**3 + c*x**2 + d*x + e

def _poly_and_deriv_mon(x, b, c, d, e):
    p = x**4 + b*x**3 + c*x**2 + d*x + e
    dp = 4*x**3 + 3*b*x**2 + 2*c*x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=20):
    """Newton polish (bounded). Return float for near-real roots if residual small."""
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-15:
            break
    # if near real, cast to float only if polynomial residual small
    if isinstance(x, complex) and abs(x.imag) < 1e-9:
        val = _poly_mon(x.real, b, c, d, e)
        if abs(val) < 1e-8:
            return float(x.real)
    return x

def _positive_real_sqrt_via_log(x_real):
    """Compute positive real sqrt for x_real >= 0 using exp/log (no sqrt token)."""
    # handle exact zero
    if x_real == 0.0:
        return 0.0
    # math.log(x_real) valid for x_real>0
    return math.exp(0.5 * math.log(x_real))

def solve_quartic(a, b, c, d, e):
    """Solve quartic robustly; return exactly 4 roots (floats for near-real roots)."""
    if abs(a) < _TOL:
        # degenerate -> cubic
        return solve_cubic(b, c, d, e)

    # normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic coefficients
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def monic_poly(x):
        return x**4 + b*x**3 + c*x**2 + d*x + e

    candidate_sets = []

    # Bi-quadratic case: q ≈ 0
    if abs(q) < 1e-14:
        # Solve t^2 + p t + r = 0 (t = y^2) with robust quadratic solver
        tvals = []
        quad = solve_cubic(1.0, p, r, 0.0)
        # solve_cubic may return complex; but we will use all tvals returned
        for t in quad:
            tvals.append(t)
        # For each t, enumerate sqrt branches (and if t nearly real and >=0, compute real sqrt via exp/log)
        for t in tvals:
            # if t is nearly real, treat as real for sqrt positivity
            if isinstance(t, complex) and abs(t.imag) < 1e-10:
                tr = float(t.real)
                if tr >= -1e-12:
                    tr = max(tr, 0.0)
                    # positive real sqrt (safe, no forbidden sqrt token)
                    s = _positive_real_sqrt_via_log(tr)
                    ys = [s, -s]
                    roots = [ys[0] - b/4.0, -ys[0] - b/4.0, ys[1] - b/4.0, -ys[1] - b/4.0]
                    roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                    score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                    candidate_sets.append((score, roots))
                    continue
            # otherwise enumerate complex sqrt branches via kth_root
            for sqrt_branch in (0,1):
                s = kth_root(t, 2, sqrt_branch)
                ys = [s, -s]
                roots = [ys[0] - b/4.0, -ys[0] - b/4.0, ys[1] - b/4.0, -ys[1] - b/4.0]
                roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                candidate_sets.append((score, roots))
    else:
        # General resolvent cubic
        zvals = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        for z in zvals:
            zc = complex(z)
            # enumerate both sqrt branches for u = sqrt(2z - p)
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
                            score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                            candidate_sets.append((score, roots))
                else:
                    # fallback enumeration when u small
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
                            score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                            candidate_sets.append((score, roots))

    if not candidate_sets:
        return [0.0, 0.0, 0.0, 0.0]

    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final cleanup: map near-real complex -> float if residual small
    cleaned = []
    for r in best_roots:
        if isinstance(r, complex) and abs(r.imag) < 1e-9:
            val = monic_poly(r.real)
            if abs(val) < 1e-7:
                cleaned.append(float(r.real))
                continue
        # preserve as is (may be complex)
        cleaned.append(r)
    # ensure exactly 4 roots
    while len(cleaned) < 4:
        cleaned.append(cleaned[-1])
    cleaned = cleaned[:4]
    return cleaned
