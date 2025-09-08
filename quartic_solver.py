# quartic_solver.py
import cmath
import math
from cubic_solver import solve_cubic, kth_root, _cleanup, _TOL

def _poly_and_deriv_mon(x, b, c, d, e):
    p = x**4 + b * x**3 + c * x**2 + d * x + e
    dp = 4 * x**3 + 3 * b * x**2 + 2 * c * x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=12):
    """Small Newton polishing to improve root accuracy."""
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    if abs(x.imag) < 1e-12:
        x = complex(x.real, 0.0)
    return x

def solve_quartic(a, b, c, d, e):
    """
    Solve a x^4 + b x^3 + c x^2 + d x + e = 0.
    Uses Ferrari + resolvent cubic with branch enumeration and small Newton polishing.
    Always returns 4 roots (with multiplicity).
    """
    if abs(a) < _TOL:
        # degenerate -> cubic
        return solve_cubic(b, c, d, e)

    # normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic: y^4 + p y^2 + q y + r = 0, with x = y - b/4
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b * b * c / 16.0 - b * d / 4.0 + e

    # monic quartic for scoring
    def monic_poly(x):
        return x**4 + b * x**3 + c * x**2 + d * x + e

    candidate_sets = []

    # Bi-quadratic case (q approx 0)
    if abs(q) < 1e-14:
        tvals = solve_cubic(1.0, p, r, 0.0)  # returns up to two t values (y^2)
        for t in tvals:
            if abs(t.imag) < 1e-12:
                tv = t.real
                # enumerate sqrt branches
                for sbranch in (0, 1):
                    s = kth_root(tv, 2, sbranch)
                    ys = [s, -s]
                    roots = [ys[0] - b/4.0, -ys[0] - b/4.0, ys[1] - b/4.0, -ys[1] - b/4.0]
                    # polish each root
                    roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                    score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                    candidate_sets.append((score, roots))
    else:
        # resolvent cubic: z^3 - p z^2 - 4 r z + (4 r p - q^2) = 0
        zvals = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        # enumerate candidate z and root branches
        for z in zvals:
            zc = complex(z)
            # enumerate both sqrt branches for u = sqrt(2 z - p)
            for u_branch in (0, 1):
                u = kth_root(2.0 * zc - p, 2, u_branch)
                small_u = abs(u) < 1e-14
                if not small_u:
                    # compute s1,s2 and their sqrt branches
                    s1 = -(2.0 * zc + p + 2.0 * q / u)
                    s2 = -(2.0 * zc + p - 2.0 * q / u)
                    # enumerate sqrt branches for t1 and t2
                    for t1_branch in (0, 1):
                        t1 = kth_root(s1, 2, t1_branch)
                        for t2_branch in (0, 1):
                            t2 = kth_root(s2, 2, t2_branch)
                            # produce four y roots
                            y1 = (-u + t1) / 2.0
                            y2 = (-u - t1) / 2.0
                            y3 = ( u + t2) / 2.0
                            y4 = ( u - t2) / 2.0
                            roots = [y1 - b/4.0, y2 - b/4.0, y3 - b/4.0, y4 - b/4.0]
                            # polish
                            roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                            score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                            candidate_sets.append((score, roots))
                else:
                    # fallback enumeration when u is tiny: use sqrt(z^2 - r) and sqrt(z) branches
                    for s_alt_branch in (0, 1):
                        s_alt = kth_root(zc * zc - r, 2, s_alt_branch)
                        for sqrt_z_branch in (0, 1):
                            sqrt_z = kth_root(zc, 2, sqrt_z_branch)
                            # form candidates
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
        # fallback: return four zeros
        return [complex(0.0, 0.0)] * 4

    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]
    final = _cleanup(best_roots)

    # ensure exactly 4 roots (with multiplicity)
    while len(final) < 4:
        final.append(final[-1])
    if len(final) > 4:
        final = final[:4]
    return final
