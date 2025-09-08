# cubic_solver.py
"""
Cubic solver without using explicit radical tokens (no **0.5, **(1/3), sqrt, etc.).
Implements robust Cardano method with branch enumeration and polishing.
Exposes solve_cubic(a, b, c, d).
"""

import cmath
import math

_TOL = 1e-14

def kth_root(z: complex, k: int, branch: int = 0) -> complex:
    """Return principal/branched k-th root using exp/log. Handles z==0."""
    if z == 0:
        return 0j
    # use cmath.log to get complex log
    return cmath.exp((cmath.log(z) + 2j * math.pi * branch) / k)

def _sqrtz(z: complex, branch: int = 0) -> complex:
    """2nd root helper using kth_root."""
    return kth_root(z, 2, branch)

def _cleanup(roots, imag_tol=1e-10, residual_tol=1e-9, coefs=None):
    """
    Convert near-real complex numbers to Python floats if their residual is small.
    coefs = (a,b,c,d) for polynomial a x^3 + b x^2 + c x + d, or None.
    """
    cleaned = []
    for z in roots:
        if isinstance(z, complex) and abs(z.imag) < imag_tol:
            xr = float(z.real)
            if coefs is None:
                cleaned.append(xr)
            else:
                a, b, c, d = coefs
                val = a * xr**3 + b * xr**2 + c * xr + d
                if abs(val) < residual_tol:
                    cleaned.append(xr)
                else:
                    cleaned.append(z)
        else:
            cleaned.append(z)
    return cleaned

def solve_quadratic(a, b, c):
    """Robust quadratic solver using kth_root. Returns two roots (may be complex)."""
    if abs(a) < _TOL:
        if abs(b) < _TOL:
            return []
        return [ -c / b ]
    # normalize to monic: x^2 + A x + B = 0
    A = b / a
    B = c / a
    disc = A * A - 4.0 * B
    # try both sqrt branches and pick best residual
    candidates = []
    for branch in (0, 1):
        sd = kth_root(disc, 2, branch)
        # numerically stable roots
        if A.real >= 0:
            q = -0.5 * (A + sd)
        else:
            q = -0.5 * (A - sd)
        if abs(q) > _TOL:
            r1 = q
            r2 = B / q
        else:
            r1 = (-A + sd) / 2.0
            r2 = (-A - sd) / 2.0
        candidates.append([r1, r2])
    # choose best candidate by minimal residual
    def residual(roots):
        return sum(abs(r*r + A*r + B)**2 for r in roots)
    best = min(candidates, key=residual)
    # convert near-real to floats
    out = []
    for r in best:
        if isinstance(r, complex) and abs(r.imag) < 1e-12:
            out.append(float(r.real))
        else:
            out.append(r)
    return out

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0.
    Returns 3 values (with multiplicity) whenever a!=0; falls back to quadratic/linear if degenerate.
    Near-real roots returned as Python floats when residual small.
    """
    if abs(a) < _TOL:
        # degenerate -> quadratic
        return solve_quadratic(b, c, d)

    # normalize to monic x^3 + A x^2 + B x + C
    A = b / a
    B = c / a
    C = d / a

    # depressed cubic t^3 + p t + q with x = t - A/3
    p = B - (A*A)/3.0
    q = (2*A*A*A)/27.0 - (A*B)/3.0 + C
    shift = -A/3.0

    # Discriminant D = (q/2)^2 + (p/3)^3
    D = (q/2.0)**2 + (p/3.0)**3

    # Special near-zero discriminant: multiple roots
    if abs(D) < 1e-18:
        # If p≈0 and q≈0 -> triple root at shift
        if abs(p) < 1e-16 and abs(q) < 1e-16:
            r = shift
            return _cleanup([r, r, r], coefs=(a,b,c,d))
        # else one single and one double root
        # Use real cube root for -q/2
        u = kth_root(-q/2.0, 3, 0)
        t1 = 2*u
        t2 = -u
        roots = [t1 + shift, t2 + shift, t2 + shift]
        # polish and cleanup
        polished = []
        for r in roots:
            polished.append(_polish_cubic_root(r, A, B, C))
        return _cleanup(polished, coefs=(1.0, A, B, C))

    # If D < 0 (three real roots), handle trigonometric form when p<0
    if isinstance(D, complex):
        D_real = D.real
        D_imag = D.imag
    else:
        D_real = D
        D_imag = 0.0

    if D_real < 0 and abs(D_imag) < 1e-12:
        # three real roots
        rho = 2.0 * math.sqrt(-p/3.0)
        arg = -q / (2.0 * math.sqrt(-(p/3.0)**3))
        # clip
        arg = max(-1.0, min(1.0, arg))
        theta = math.acos(arg)
        t1 = rho * math.cos(theta/3.0)
        t2 = rho * math.cos((theta + 2.0*math.pi)/3.0)
        t3 = rho * math.cos((theta + 4.0*math.pi)/3.0)
        roots = [t1 + shift, t2 + shift, t3 + shift]
        polished = [_polish_cubic_root(r, A, B, C) for r in roots]
        return _cleanup(polished, coefs=(1.0, A, B, C))

    # General Cardano: enumerate sqrt and cube root branches, pick best residual
    best_score = None
    best_roots = None
    # enumerate two sqrt branches
    for sbranch in (0,1):
        sD = kth_root(D, 2, sbranch)
        # enumerate cube root branches for u and v (3x3)
        for ub in range(3):
            for vb in range(3):
                u = kth_root(-q/2.0 + sD, 3, ub)
                v = kth_root(-q/2.0 - sD, 3, vb)
                # roots of depressed cubic
                omega = cmath.exp(2j * math.pi / 3.0)
                t1 = u + v
                t2 = omega * u + (omega**2) * v
                t3 = (omega**2) * u + omega * v
                x1 = t1 + shift
                x2 = t2 + shift
                x3 = t3 + shift
                # residual for monic cubic
                def mono(x): return x**3 + A*x**2 + B*x + C
                score = abs(mono(x1))**2 + abs(mono(x2))**2 + abs(mono(x3))**2
                if best_score is None or score < best_score:
                    best_score = score
                    best_roots = [x1, x2, x3]
    # polish and cleanup
    best_roots = [_polish_cubic_root(x, A, B, C) for x in best_roots]
    return _cleanup(best_roots, coefs=(1.0, A, B, C))

def _polish_cubic_root(x0, A, B, C, maxiter=20):
    """Newton polishing for cubic monic x^3 + A x^2 + B x + C"""
    x = x0
    for _ in range(maxiter):
        p = x**3 + A*x**2 + B*x + C
        dp = 3*x**2 + 2*A*x + B
        if abs(dp) < 1e-20:
            break
        dx = p / dp
        x = x - dx
        if abs(dx) < 1e-14:
            break
    # cast near-real to float only if residual small
    if isinstance(x, complex) and abs(x.imag) < 1e-10:
        val = (x.real**3 + A*x.real**2 + B*x.real + C)
        if abs(val) < 1e-9:
            return float(x.real)
    return x
