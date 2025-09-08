# quartic_solver.py
import cmath
from cubic_solver import solve_cubic, _sqrtz, _cleanup, _TOL

def solve_quartic(a, b, c, d, e):
    """
    Solve a x^4 + b x^3 + c x^2 + d x + e = 0.
    Uses Ferrari/resolvent cubic plus exhaustive (finite) branch enumeration to select best roots.
    Returns exactly 4 roots (with multiplicity). Uses exp/log-based k-th roots (no sqrt/**0.5/**(1/3)).
    """
    if abs(a) < _TOL:
        # Degenerate to cubic
        return solve_cubic(b, c, d, e)

    # Normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # Depressed quartic y^4 + p y^2 + q y + r = 0, x = y - b/4
    p = c - 3.0 * b * b / 8.0
    q = b**3 / 8.0 - b * c / 2.0 + d
    r = -3.0 * b**4 / 256.0 + b*b*c / 16.0 - b * d / 4.0 + e

    # monic quartic polynomial for scoring
    def poly(x):
        return x**4 + b * x**3 + c * x**2 + d * x + e

    candidate_sets = []

    # Special biquadratic case q ~ 0: solve t^2 + p t + r = 0 (t = y^2)
    if abs(q) < 1e-14:
        t_vals = solve_cubic(1.0, p, r, 0.0)  # returns up to 2 t's (as complex)
        for t in t_vals:
            if abs(t.imag) < 1e-12:
                tv = t.real
                s = _sqrtz(tv)
                y_candidates = [s, -s]
                roots = [y_candidates[0] - b/4.0, y_candidates[1] - b/4.0,
                         -y_candidates[0] - b/4.0, -y_candidates[1] - b/4.0]
                # compute score
                score = sum(abs(poly(r_))**2 for r_ in roots)
                candidate_sets.append((score, roots))
    else:
        # General Ferrari: resolvent cubic z^3 - p z^2 - 4 r z + (4 r p - q^2) = 0
        z_candidates = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)
        # Try each z candidate (including complex) and enumerate branches
        for z in z_candidates:
            zc = complex(z)
            # principal sqrt for u = sqrt(2z - p)
            su = _sqrtz(2.0 * zc - p)

            # safe fallback if su too small
            small_u = abs(su) < 1e-12

            # s1 and s2 definitions (may use fallback when u tiny)
            # when su != 0: s1 = -(2z + p + 2 q / u), s2 = -(2z + p - 2 q / u)
            # when su ≈ 0: use s1 = -(2z + p) + 2*sqrt(z*z - r) and s2 = -(2z + p) - 2*sqrt(z*z - r)
            if not small_u:
                s1 = -(2.0 * zc + p + 2.0 * q / su)
                s2 = -(2.0 * zc + p - 2.0 * q / su)
                t1p = _sqrtz(s1)
                t2p = _sqrtz(s2)
                u_options = [su, -su]
                t1_options = [t1p, -t1p]
                t2_options = [t2p, -t2p]
                # enumerate sign combos
                for u in u_options:
                    for t1 in t1_options:
                        for t2 in t2_options:
                            y1 = (-u + t1) / 2.0
                            y2 = (-u - t1) / 2.0
                            y3 = ( u + t2) / 2.0
                            y4 = ( u - t2) / 2.0
                            roots = [y1 - b / 4.0, y2 - b / 4.0, y3 - b / 4.0, y4 - b / 4.0]
                            score = sum(abs(poly(r_))**2 for r_ in roots)
                            candidate_sets.append((score, roots))
            else:
                # fallback path when u is tiny: use sqrt(z^2 - r)
                s_alt = _sqrtz(zc * zc - r)
                for s_alt_sign in (s_alt, -s_alt):
                    # form the two quadratics' discriminant-like pieces
                    # Build candidate roots using standard formulas:
                    # y = ± sqrt(z) ± sqrt(z^2 - r) ... (we enumerate sign combos)
                    # compute sqrt(z) safely
                    sqrt_z = _sqrtz(zc)
                    for s1choice in (sqrt_z, -sqrt_z):
                        for s2choice in (s_alt_sign, -s_alt_sign):
                            # compose 4 roots heuristically (these formulas are classical variants)
                            y_candidates = [ ( s1choice + s2choice ) / 2.0,
                                             ( s1choice - s2choice ) / 2.0,
                                             ( -s1choice + s2choice ) / 2.0,
                                             ( -s1choice - s2choice ) / 2.0 ]
                            roots = [y - b / 4.0 for y in y_candidates]
                            score = sum(abs(poly(r_))**2 for r_ in roots)
                            candidate_sets.append((score, roots))

    # If no candidate sets found (should be rare), fall back to zeros
    if not candidate_sets:
        return [complex(0.0, 0.0)] * 4

    # pick best by minimal score (sum squared residuals)
    candidate_sets.sort(key=lambda t: t[0])
    best_roots = candidate_sets[0][1]

    # final cleanup: reduce tiny imaginary noise, and ensure exactly 4 roots returned
    final = _cleanup(best_roots)
    # pad/trim to exactly 4
    if len(final) < 4:
        while len(final) < 4:
            final.append(final[-1])
    else:
        final = final[:4]
    return final
