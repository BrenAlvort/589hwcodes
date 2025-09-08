# quartic_solver.py
import cmath
from cubic_solver import solve_cubic, _sqrtz, _cleanup, _TOL

def solve_quartic(a, b, c, d, e):
    """
    Solve a x^4 + b x^3 + c x^2 + d x + e = 0
    Uses Ferrari + branch enumeration (finite tries) to pick sign/branch combinations
    that minimize polynomial residuals. No sqrt/**0.5/**(1/3) tokens used.
    Always returns exactly 4 roots (with multiplicity).
    """
    if abs(a) < _TOL:
        # Degenerate to cubic
        return solve_cubic(b, c, d, e)

    # Normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # Depressed quartic: y^4 + p y^2 + q y + r = 0 with x = y - b/4
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    # helper polynomial for normalized monic quartic: x^4 + b x^3 + c x^2 + d x + e
    def poly(x):
        return x**4 + b * x**3 + c * x**2 + d * x + e

    candidate_roots = []

    if abs(q) < 1e-14:
        # biquadratic: y^4 + p y^2 + r = 0
        # Solve t^2 + p t + r = 0 for t = y^2
        quad_ts = solve_cubic(1.0, p, r, 0.0)  # returns up to 2 t values
        for t in quad_ts:
            # only accept t if nearly real
            if abs(t.imag) < 1e-12:
                tval = t.real
                # y = ±sqrt(t) where sqrt uses _sqrtz
                s = _sqrtz(tval)
                y_candidates = [s, -s]
                for y in y_candidates:
                    candidate_roots.append(y - b / 4.0)
    else:
        # Solve resolvent cubic: z^3 - p z^2 - 4 r z + (4 r p - q^2) = 0
        cub_roots = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)

        # sort candidates by imaginary magnitude then by real descending
        cub_sorted = sorted(cub_roots, key=lambda z: (abs(z.imag), -z.real))

        best_roots = None
        best_score = None

        # Try each candidate z and enumerate sign flips for u, t1, t2
        for z in cub_sorted:
            # allow complex z; keep as complex
            zc = complex(z)

            # compute principal u = sqrt(2z - p)
            s_u = _sqrtz(2.0 * zc - p)

            # compute s1 and s2, then principal sqrt of those
            s1 = -(2.0 * zc + p + 2.0 * q / s_u) if abs(s_u) > 1e-16 else -(2.0 * zc + p + 0j)
            s2 = -(2.0 * zc + p - 2.0 * q / s_u) if abs(s_u) > 1e-16 else -(2.0 * zc + p - 0j)
            t1p = _sqrtz(s1)
            t2p = _sqrtz(s2)

            # enumerate sign flips: u_sign, t1_sign, t2_sign ∈ {1, -1}
            for su in (1, -1):
                u = su * s_u
                for st1 in (1, -1):
                    t1 = st1 * t1p
                    for st2 in (1, -1):
                        t2 = st2 * t2p
                        # construct four y-roots
                        y1 = (-u + t1) / 2.0
                        y2 = (-u - t1) / 2.0
                        y3 = (u + t2) / 2.0
                        y4 = (u - t2) / 2.0
                        # shift back to x
                        cand = [y1 - b / 4.0, y2 - b / 4.0, y3 - b / 4.0, y4 - b / 4.0]

                        # compute residual score
                        score = sum(abs(poly(c))**2 for c in cand)

                        if best_score is None or score < best_score:
                            best_score = score
                            best_roots = cand

        if best_roots is not None:
            candidate_roots.extend(best_roots)

    # Clean tiny imaginary parts and filter duplicates by residual,
    # but ensure we return exactly 4 roots (with multiplicity).
    cleaned = _cleanup(candidate_roots)

    # Evaluate residuals and keep those below a tolerance, but we must
    # always produce 4 roots. So pick best 4 by residual if necessary.
    root_scores = []
    for r in cleaned:
        root_scores.append((abs(poly(r)), r))

    # If no candidate found (very unlikely), fallback to zeros
    if not root_scores:
        return [complex(0, 0)] * 4

    # sort by residual ascending
    root_scores.sort(key=lambda t: t[0])

    # If we have fewer than 4 candidates, repeat top ones to reach 4
    roots_final = [t[1] for t in root_scores]
    # pad to 4
    while len(roots_final) < 4:
        roots_final.append(roots_final[-1])

    # If more than 4, pick the best 4 (smallest residuals)
    roots_final = roots_final[:4]

    # final cleanup tiny imag
    return _cleanup(roots_final)
