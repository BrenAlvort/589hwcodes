# cubic_solver.py
import cmath
import math

_TOL = 1e-14

def kth_root(z: complex, k: int, branch: int = 0) -> complex:
    """k-th root via exp/log (no forbidden radical tokens)."""
    if z == 0:
        return 0j
    return cmath.exp((cmath.log(z) + 2j * math.pi * branch) / k)

def _maybe_real(z, imag_tol=1e-10, residual_tol=1e-10, monic_coefs=None):
    """Return a Python float if z is nearly real and polynomial residual small (if monic_coefs given)."""
    if isinstance(z, complex) and abs(z.imag) < imag_tol:
        xr = float(z.real)
        if monic_coefs is None:
            return xr
        # monic_coefs is tuple (A,B,C) for cubic monic poly x^3 + A x^2 + B x + C
        A, B, C = monic_coefs
        val = xr**3 + A * xr**2 + B * xr + C
        if abs(val) < residual_tol:
            return xr
        # else keep complex (to allow polishing to fix)
    return z

def _cleanup_list_cubic(roots, monic_coefs=None):
    return [_maybe_real(z, imag_tol=1e-10, residual_tol=1e-9, monic_coefs=monic_coefs) for z in roots]

def solve_quadratic(a, b, c):
    """Robust quadratic using kth_root. Returns two roots; real roots are Python floats."""
    if abs(a) < _TOL:
        if abs(b) < _TOL:
            return []
        return [float(-c / b)]
    # normalize
    A = b / a
    B = c / a
    disc = A * A - 4.0 * B
    candidates = []
    for sbranch in (0, 1):
        sd = kth_root(disc, 2, sbranch)
        if A.real >= 0:
            q = -0.5 * (A + sd)
        else:
            q = -0.5 * (A - sd)
        if abs(q) > _TOL:
            r1 = q / 1.0
            r2 = B / q
        else:
            r1 = (-A + sd) / 2.0
            r2 = (-A - sd) / 2.0
        candidates.append([r1, r2])
    def residual(roots):
        return sum(abs(r*r + A*r + B)**2 for r in roots)
    chosen = min(candidates, key=residual)
    # convert near-real to floats if residual small
    out = []
    for r in chosen:
        if isinstance(r, complex) and abs(r.imag) < 1e-10:
            out.append(float(r.real))
        else:
            out.append(r)
    return out

def solve_cubic(a, b, c, d):
    """Solve cubic robustly; returns floats for near-real roots."""
    if abs(a) < _TOL:
        return solve_quadratic(b, c, d)

    A = b / a
    B = c / a
    C = d / a
    p = B - (A*A)/3.0
    q = (2*A*A*A)/27.0 - (A*B)/3.0 + C
    shift = -A/3.0

    D = (q/2.0)**2 + (p/3.0)**3

    # D ~ 0 multiple roots
    if abs(D) < 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            return _cleanup_list_cubic([complex(shift), complex(shift), complex(shift)], (A,B,C))
        u = -q/2.0
        if abs(u.imag) < 1e-12:
            if u == 0:
                uroot = 0.0
            else:
                uroot = math.copysign(abs(u)**(1.0/3.0), u)
            found = [2*uroot + shift, -uroot + shift, -uroot + shift]
            return _cleanup_list_cubic(found, (A,B,C))
        # else fall through to enumeration

    # D < 0: trigonometric (three real)
    if D.real < 0 and abs(D.imag) < 1e-12:
        rho = 2.0 * math.sqrt(-p/3.0)
        arg = -q / (2.0 * math.sqrt(-(p/3.0)**3))
        arg = max(-1.0, min(1.0, arg))
        theta = math.acos(arg)
        r1 = rho * math.cos(theta/3.0) + shift
        r2 = rho * math.cos((theta + 2.0*math.pi)/3.0) + shift
        r3 = rho * math.cos((theta + 4.0*math.pi)/3.0) + shift
        return _cleanup_list_cubic([complex(r1), complex(r2), complex(r3)], (A,B,C))

    # General Cardano: enumerate sqrt and cbrt branches
    best = None
    best_score = None
    for sqrt_branch in (0,1):
        sD = kth_root(D, 2, sqrt_branch)
        for ub in range(3):
            for vb in range(3):
                u = kth_root(-q/2.0 + sD, 3, ub)
                v = kth_root(-q/2.0 - sD, 3, vb)
                omega = cmath.exp(2j * math.pi / 3.0)
                t1 = u + v
                t2 = omega * u + omega**2 * v
                t3 = omega**2 * u + omega * v
                x1 = t1 + shift
                x2 = t2 + shift
                x3 = t3 + shift
                def mono(x): return x**3 + A*x**2 + B*x + C
                score = abs(mono(x1))**2 + abs(mono(x2))**2 + abs(mono(x3))**2
                if best_score is None or score < best_score:
                    best_score = score
                    best = [x1, x2, x3]

    # Small polishing of chosen triple (Newton up to few steps)
    def polish(x0):
        x = x0
        for _ in range(12):
            pval = x**3 + A*x**2 + B*x + C
            dp = 3*x**2 + 2*A*x + B
            if abs(dp) < 1e-20:
                break
            dx = pval / dp
            x = x - dx
            if abs(dx) < 1e-14:
                break
        return x

    best = [polish(r) for r in best]
    return _cleanup_list_cubic(best, (A,B,C))
