# quartic_solver.py
"""
Quartic solver without using explicit radical tokens.
Implements Ferrari (resolvent cubic) with branch enumeration, polishing and degeneracy handling.
Exposes solve_quartic(a, b, c, d, e).
"""

import cmath
import math
from cubic_solver import solve_cubic, kth_root, _TOL  # kth_root used for 2nd roots too

# local tiny tolerances
_IMAG_TOL = 1e-10
_RESID_TOL = 1e-8

def _polish_quartic_root(x0, b, c, d, e, maxiter=25):
    """Polish a root of monic quartic x^4 + b x^3 + c x^2 + d x + e (Newton)."""
    x = x0
    for _ in range(maxiter):
        p = x**4 + b*x**3 + c*x**2 + d*x + e
        dp = 4*x**3 + 3*b*x**2 + 2*c*x + d
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-15:
            break
    # if near-real and residual small, return float
    if isinstance(x, complex) and abs(x.imag) < 1e-9:
        val = (x.real**4 + b*x.real**3 + c*x.real**2 + d*x.real + e)
        if abs(val) < 1e-8:
            return float(x.real)
    return x

def _positive_real_sqrt_via_log(x_real):
    """Compute positive sqrt for non-negative real via exp/log (no sqrt token)."""
    if x_real == 0.0:
        return 0.0
    return math.exp(0.5 * math.log(x_real))

def _cleanup_list(roots, b, c, d, e):
    """Convert near-real complex roots to floats when residual small."""
    cleaned = []
    for r in roots:
        if isinstance(r, complex) and abs(r.imag) < _IMAG_TOL:
            val = r.real**4 + b*r.real**3 + c*r.real**2 + d*r.real + e
            if abs(val) < _RESID_TOL:
                cleaned.append(float(r.real))
                continue
        cleaned.append(r)
    return cleaned

def _ensure_four(roots):
    """Return exactly four roots (with multiplicity) by repeating last if needed."""
    roots = list(roots)
    while len(roots) < 4:
        if roots:
            roots.append(roots[-1])
        else:
            roots.append(0.0)
    return roots[:4]

def solve_quartic(a, b, c, d, e):
    """
    Solve a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
    Returns exactly 4 roots (with multiplicity). Uses kth_root for any root extraction.
    Degenerates to cubic/quadratic/linear if leading coefficients vanish.
    """
    # handle degeneracies: fall back to cubic/quadratic/linear
    if abs(a) < _TOL:
        # cubic
        from cubic_solver import solve_cubic as _solve_cubic_fallback
        return _solve_cubic_fallback(b, c, d, e)

    # normalize to monic quartic: x^4 + B x^3 + C x^2 + D x + E
    B = b / a
    C = c / a
    D = d / a
    E = e / a

    # depressed quartic: x = y - B/4 -> y^4 + p y^2 + q y + r = 0
    p = C - 3.0 * (B*B) / 8.0
    q = (B*B*B) / 8.0 - (B*C) / 2.0 + D
    r = -3.0 * (B**4) / 256.0 + (B*B*C) / 16.0 - (B*D) / 4.0 + E

    # helper monic polynomial residual
    def monic_residual(roots):
        return sum(abs(x**4 + B*x**3 + C*x**2 + D*x + E)**2 for x in roots)

    candidate_sets = []

    # special biquadratic case q ~ 0: y^4 + p y^2 + r = 0 => t^2 + p t + r = 0 with t=y^2
    if abs(q) < 1e-14:
        # Solve t^2 + p t + r = 0 using quadratic solver (use cubic solver fallback for robust behavior)
        tvals = solve_cubic(1.0, p, r, 0.0)
        # tvals may contain 0..2 values depending on degeneracies, iterate through them
        for t in tvals:
            # if nearly real and >=0, compute positive real sqrt via exp/log
            if isinstance(t, complex) and abs(t.imag) < 1e-10:
                tr = float(t.real)
                if tr >= -1e-12:
                    tr = max(tr, 0.0)
                    s = _positive_real_sqrt_via_log(tr)
                    y_roots = [s, -s]
                    # convert to x by shift
                    roots = [y - B/4.0 for y in (y_roots[0], -y_roots[0], y_roots[1], -y_roots[1])]
                    roots = [_polish_quartic_root(x, B, C, D, E) for x in roots]
                    candidate_sets.append((monic_residual(roots), roots))
                    continue
            # otherwise use 2 sqrt branches
            for branch in (0, 1):
                s = kth_root(t, 2, branch)
                y_roots = [s, -s]
                roots = [y - B/4.0 for y in (y_roots[0], -y_roots[0], y_roots[1], -y_roots[1])]
                roots = [_polish_quartic_root(x, B, C, D, E) for x in roots]
                candidate_sets.append((monic_residual(roots), roots))

    else:
        # general case -> resolvent cubic (Ferrari)
        # resolvent cubic z^3 - p z^2 - 4 r z + (4 r p - q^2) = 0
        zvals = solve_cubic(1.0, -p, -4.0*r, 4.0*r*p - q*q)
        # iterate candidate z values and sqrt branches
        for z in zvals:
            zc = complex(z)
            # pick u = sqrt(2z - p) with two branches
            for u_branch in (0, 1):
                u = kth_root(2.0*zc - p, 2, u_branch)
                # if u is near zero, handle fallback enumeration
                if abs(u) > 1e-12:
                    # s1 = -(2 z + p + 2 q / u), s2 = -(2 z + p - 2 q / u)
                    twozp = 2.0*zc + p
                    s1 = -(twozp + 2.0 * q / u)
                    s2 = -(twozp - 2.0 * q / u)
                    # enumerate sqrt branches for s1 and s2
                    for t1_branch in (0, 1):
                        t1 = kth_root(s1, 2, t1_branch)
                        for t2_branch in (0, 1):
                            t2 = kth_root(s2, 2, t2_branch)
                            y1 = (-u + t1) / 2.0
                            y2 = (-u - t1) / 2.0
                            y3 = ( u + t2) / 2.0
                            y4 = ( u - t2) / 2.0
                            roots = [y1 - B/4.0, y2 - B/4.0, y3 - B/4.0, y4 - B/4.0]
                            roots = [_polish_quartic_root(x, B, C, D, E) for x in roots]
                            candidate_sets.append((monic_residual(roots), roots))
                else:
                    # fallback branching when u ~ 0: take sqrt(z^2 - r) and sqrt(z)
                    for s_alt_branch in (0, 1):
                        s_alt = kth_root(zc*zc - r, 2, s_alt_branch)
                        for sqrt_z_branch in (0, 1):
                            sqrt_z = kth_root(zc, 2, sqrt_z_branch)
                            y_candidates = [
                                ( sqrt_z + s_alt) / 2.0,
                                ( sqrt_z - s_alt) / 2.0,
                                (-sqrt_z + s_alt) / 2.0,
                                (-sqrt_z - s_alt) / 2.0,
                            ]
                            roots = [y - B/4.0 for y in y_candidates]
                            roots = [_polish_quartic_root(x, B, C, D, E) for x in roots]
                            candidate_sets.append((monic_residual(roots), roots))

    if not candidate_sets:
        # fallback: try numeric factorization or return four zeros
        return [0.0, 0.0, 0.0, 0.0]

    # pick candidate with minimal residual
    candidate_sets.sort(key=lambda t: t[0])
    best = candidate_sets[0][1]
    # cleanup near-real and ensure four roots
    cleaned = _cleanup_list(best, B, C, D, E)
    cleaned = _ensure_four(cleaned)
    return cleaned
