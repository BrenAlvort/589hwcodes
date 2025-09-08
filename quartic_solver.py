# quartic_solver.py
import cmath
import math
from cubic_solver import solve_cubic, kth_root, _cleanup_list, _TOL

def _poly_mon(x, b, c, d, e):
    return x**4 + b * x**3 + c * x**2 + d * x + e

def _poly_and_deriv_mon(x, b, c, d, e):
    p = x**4 + b * x**3 + c * x**2 + d * x + e
    dp = 4*x**3 + 3*b * x**2 + 2*c * x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=8):
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    if isinstance(x, complex) and abs(x.imag) < 1e-12:
        return float(x.real)
    return x

def solve_quartic(a, b, c, d, e):
    """
    Solve a x^4 + b x^3 + c x^2 + d x + e = 0.
    Uses Ferrari + resolvent cubic, enumerates branches, polishes slightly.
    Returns exactly 4 roots (floats for near-real roots, complex otherwise).
    """
    if abs(a) < _TOL:
        # degenerate -> cubic
        return solve_cubic(b, c, d, e)

    # normalize to monic coefficients
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic coefficients (y with x = y - b/4)
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def monic_poly(x):
        return x**4 + b * x**3 + c * x**2 + d * x + e

    candidate_sets = []

    # Bi-quadratic special-case: q ≈ 0 -> t = y^2 solves t^2 + p t + r = 0
    if abs(q) < 1e-14:
        tvals = solve_cubic(1.0, p, r, 0.0)  # may return up to 2 t's
        # enumerate each t value (do NOT filter out small imaginary parts here)
        for t in tvals:
            # for each t we must produce ±sqrt(t) (both sqrt branches)
            for sqrt_branch in (0, 1):
                s = kth_root(t, 2, sqrt_branch)
                y_pos = s
                y_neg = -s
                roots = [y_pos - b/4.0, -y_pos - b/4.0, y_neg - b/4.0, -y_neg - b/4.0]
                # polish roots
                roots = [_polish_root(r_, b, c, d, e) for r_ in roots]
                # score
                score = sum(abs(monic_poly(r_))**2 for r_ in roots)
                candidate_sets.append((score, roots))

    else:
        # General resolvent cubic
        zvals = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        # for robustness enumerate many z candidates
        for z in zvals:
            zc = complex(z)
            # enumerate square-root branches for u = sqrt(2z - p)
            for u_branch in (0, 1):
                u = kth_root(2.0 * zc - p, 2, u_branch)
                small_u = abs(u) < 1e-14
                if not small_u:
                    s1 = -(2.0 * zc + p + 2.0 * q / u)
                    s2 = -(2.0 * zc + p - 2.0 * q / u)
                    # enumerate sqrt branches for s1, s2
                    for t1_branch in (0, 1):
                        t1 = kth_root(s1, 2, t1_branch)
                        for t2_branch in (0, 1):
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
                    # fallback enumeration when u tiny
                    for s_alt_branch in (0, 1):
                        s_alt = kth_root(zc*zc - r, 2, s_alt_branch)
                        for sqrt_z_branch in (0, 1):
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
        # fallback: return 4 zeros
        return [0.0, 0.0, 0.0, 0.0]

    # choose best candidate set (minimal residual score)
    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final cleanup: map near-real complex → float, ensure exactly 4 elements
    final = _cleanup_list(best_roots)
    while len(final) < 4:
        final.append(final[-1])
    final = final[:4]
    return final
