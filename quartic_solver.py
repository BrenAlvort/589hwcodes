# quartic_solver.py
import cmath
import math
from cubic_solver import solve_cubic, solve_quadratic, kth_root, _to_real_if_good, _TOL

def _poly_mon(x, b, c, d, e):
    return x**4 + b * x**3 + c * x**2 + d * x + e

def _poly_and_deriv_mon(x, b, c, d, e):
    p = x**4 + b * x**3 + c * x**2 + d * x + e
    dp = 4 * x**3 + 3 * b * x**2 + 2 * c * x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=12):
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    # if near-real AND residual small, return float
    if isinstance(x, complex) and abs(x.imag) < 1e-10:
        val = _poly_mon(x.real, b, c, d, e)
        if abs(val) < 1e-8:
            return float(x.real)
    return x

def _positive_real_sqrt(x_real):
    """Safe positive real sqrt via exp/log (x_real >= 0)."""
    if x_real == 0.0:
        return 0.0
    return math.exp(0.5 * math.log(x_real))

def solve_quartic(a, b, c, d, e):
    """Solve quartic and always return 4 roots (floats for true reals)."""
    if abs(a) < _TOL:
        return solve_cubic(b, c, d, e)

    # normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic: y^4 + p y^2 + q y + r = 0, x = y - b/4
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def monic_poly(x):
        return x**4 + b * x**3 + c * x**2 + d * x + e

    candidate_sets = []

    # === Bi-quadratic case (q ≈ 0) ===
    if abs(q) < 1e-14:
        # Use robust quadratic solver for t: t^2 + p t + r = 0 (t = y^2)
        tvals = solve_quadratic(1.0, p, r)  # returns exactly two roots
        # ensure we have two t values (it returns two entries for non-degenerate)
        # enumerate sqrt branches for each t
        for t in tvals:
            # if t is a float (real)
            if isinstance(t, (float, int)):
                tr = float(t)
                if tr >= -1e-12:
                    tr = max(tr, 0.0)
                    # positive real sqrt safely via exp/log
                    s = _positive_real_sqrt(tr)
                    y_pos = s
                    y_neg = -s
                    roots = [y_pos - b/4.0, -y_pos - b/4.0, y_neg - b/4.0, -y_neg - b/4.0]
                    roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                    score = sum(abs(monic_poly(rr))**2 for rr in roots)
                    candidate_sets.append((score, roots))
                    continue
            # otherwise t may be complex; enumerate both sqrt branches using kth_root
            for sbranch in (0,1):
                s = kth_root(t, 2, sbranch)
                y_pos = s
                y_neg = -s
                roots = [y_pos - b/4.0, -y_pos - b/4.0, y_neg - b/4.0, -y_neg - b/4.0]
                roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                score = sum(abs(monic_poly(rr))**2 for rr in roots)
                candidate_sets.append((score, roots))

    else:
        # === General quartic via resolvent cubic ===
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
                            score = sum(abs(monic_poly(rr))**2 for rr in roots)
                            candidate_sets.append((score, roots))
                else:
                    # fallback when u tiny: enumerate sqrt branches of z^2 - r and sqrt(z)
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
                            score = sum(abs(monic_poly(rr))**2 for rr in roots)
                            candidate_sets.append((score, roots))

    if not candidate_sets:
        return [0.0, 0.0, 0.0, 0.0]

    # pick best candidate set
    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final cast: return floats for near-real roots whose residual is tiny, keep complex otherwise
    final = []
    for r in best_roots:
        if isinstance(r, complex) and abs(r.imag) < 1e-9:
            # check residual at real part
            val = monic_poly(r.real)
            if abs(val) < 1e-7:
                final.append(float(r.real))
                continue
        final.append(r)

    # ensure exactly 4 entries
    while len(final) < 4:
        final.append(final[-1])
    final = final[:4]
    return final
