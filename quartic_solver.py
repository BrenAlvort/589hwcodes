# quartic_solver.py
import cmath
from cubic_solver import solve_cubic, _sqrtz, _cleanup, _TOL

def _poly_and_deriv_mon(x, b, c, d, e):
    """Return p(x) and p'(x) for monic quartic x^4 + b x^3 + c x^2 + d x + e"""
    # p(x)
    p = x**4 + b * x**3 + c * x**2 + d * x + e
    dp = 4*x**3 + 3*b * x**2 + 2*c * x + d
    return p, dp

def _polish_root(x0, b, c, d, e, maxiter=6):
    """
    A small Newton polishing loop to improve the root accuracy.
    This is a polishing step only (1..6 iterations).
    """
    x = x0
    for _ in range(maxiter):
        p, dp = _poly_and_deriv_mon(x, b, c, d, e)
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    # cleanup tiny imag
    if abs(x.imag) < 1e-12:
        x = complex(x.real, 0.0)
    return x

def solve_quartic(a, b, c, d, e):
    """
    Solve ax^4 + bx^3 + cx^2 + dx + e = 0
    Uses Ferrari with branch enumeration, then does a small polishing step.
    Returns exactly 4 roots (with multiplicity).
    """
    if abs(a) < _TOL:
        # degenerate -> cubic
        return solve_cubic(b, c, d, e)

    # normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # depressed quartic: y^4 + p y^2 + q y + r = 0; x = y - b/4
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    def poly_mon(x):
        return x**4 + b * x**3 + c * x**2 + d * x + e

    candidate_sets = []

    # biquadratic case q ~ 0
    if abs(q) < 1e-14:
        # t^2 + p t + r = 0
        tvals = solve_cubic(1.0, p, r, 0.0)
        for t in tvals:
            if abs(t.imag) < 1e-12:
                tv = t.real
                s = _sqrtz(tv)
                ys = [s, -s]
                roots = [ys[0] - b/4.0, -ys[0] - b/4.0, ys[1] - b/4.0, -ys[1] - b/4.0]
                # polish and score
                roots = [_polish_root(r, b, c, d, e) for r in roots]
                score = sum(abs(poly_mon(r))**2 for r in roots)
                candidate_sets.append((score, roots))

    else:
        # resolvent cubic z^3 - p z^2 - 4 r z + (4 r p - q^2) = 0
        zvals = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        # try many z candidates and branch sign combos
        for z in zvals:
            zc = complex(z)
            # principal sqrt for u = sqrt(2z - p)
            su = _sqrtz(2.0 * zc - p)
            small_u = abs(su) < 1e-12
            if not small_u:
                s1 = -(2.0 * zc + p + 2.0 * q / su)
                s2 = -(2.0 * zc + p - 2.0 * q / su)
                t1p = _sqrtz(s1)
                t2p = _sqrtz(s2)
                for su_sign in (1, -1):
                    u = su_sign * su
                    for t1_sign in (1, -1):
                        t1 = t1_sign * t1p
                        for t2_sign in (1, -1):
                            t2 = t2_sign * t2p
                            y1 = (-u + t1) / 2.0
                            y2 = (-u - t1) / 2.0
                            y3 = ( u + t2) / 2.0
                            y4 = ( u - t2) / 2.0
                            roots = [y1 - b/4.0, y2 - b/4.0, y3 - b/4.0, y4 - b/4.0]
                            # polish
                            roots = [_polish_root(r, b, c, d, e) for r in roots]
                            score = sum(abs(poly_mon(r))**2 for r in roots)
                            candidate_sets.append((score, roots))
            else:
                # fallback path when u small: use sqrt(z^2 - r) stuff
                s_alt = _sqrtz(zc * zc - r)
                sqrt_z = _sqrtz(zc)
                for s1choice in (sqrt_z, -sqrt_z):
                    for s2choice in (s_alt, -s_alt):
                        y_candidates = [
                            (s1choice + s2choice)/2.0,
                            (s1choice - s2choice)/2.0,
                            (-s1choice + s2choice)/2.0,
                            (-s1choice - s2choice)/2.0
                        ]
                        roots = [y - b/4.0 for y in y_candidates]
                        roots = [_polish_root(r, b, c, d, e) for r in roots]
                        score = sum(abs(poly_mon(r))**2 for r in roots)
                        candidate_sets.append((score, roots))

    if not candidate_sets:
        # fallback: return 4 zeros
        return [complex(0.0, 0.0)] * 4

    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]
    # post-clean: ensure exactly 4 and remove tiny imag noise
    final = _cleanup(best_roots)
    # pad if needed
    while len(final) < 4:
        final.append(final[-1])
    final = final[:4]
    return final
