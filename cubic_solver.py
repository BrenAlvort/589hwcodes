# cubic_solver.py
import cmath
import math

_TOL = 1e-14

def kth_root(z: complex, k: int, branch: int = 0) -> complex:
    """Return k-th root of z with branch index (0..k-1), using exp/log (no sqrt/**0.5)."""
    if z == 0:
        return 0j
    return cmath.exp((cmath.log(z) + 2j * math.pi * branch) / k)

def _cleanup_as_maybe_real(z):
    """If z is nearly real, return Python float; otherwise return complex."""
    if isinstance(z, complex) and abs(z.imag) < 1e-12:
        return float(z.real)
    return z

def _cleanup_list(roots):
    """Apply _cleanup_as_maybe_real to each root and return list."""
    return [_cleanup_as_maybe_real(z) for z in roots]

def solve_quadratic(a, b, c):
    """Solve a x^2 + b x + c = 0 using kth_root enumeration for sqrt; returns two roots (float or complex)."""
    if abs(a) < _TOL:
        if abs(b) < _TOL:
            return []
        return [float(-c / b)]
    # normalize
    A = b / a
    B = c / a
    disc = A * A - 4.0 * B
    candidates = []
    # enumerate two square-root branches of disc
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
    # pick smaller residual set
    def quad_res(roots):
        Acoef = A
        Bcoef = B
        return sum(abs(r*r + Acoef*r + Bcoef)**2 for r in roots)
    chosen = min(candidates, key=quad_res)
    return _cleanup_list(chosen)

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0.
    Uses depressed cubic + Cardano with branch enumeration and a small Newton polish.
    Returns 1..3 roots (floats when real, complex otherwise).
    """
    if abs(a) < _TOL:
        # degenerate -> quadratic
        return solve_quadratic(b, c, d)

    # normalize to monic
    A = b / a
    B = c / a
    C = d / a

    # depressed cubic t^3 + p t + q = 0 with x = t - A/3
    p = B - (A*A)/3.0
    q = (2.0*A*A*A)/27.0 - (A*B)/3.0 + C
    shift = -A/3.0

    # discriminant
    D = (q/2.0)**2 + (p/3.0)**3

    # D ~ 0 -> multiple roots
    if abs(D) < 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            return _cleanup_list([complex(shift), complex(shift), complex(shift)])
        u = -q/2.0
        if abs(u.imag) < 1e-12:
            if u == 0:
                uroot = 0.0
            else:
                uroot = math.copysign(abs(u)**(1.0/3.0), u)
            return _cleanup_list([2*uroot + shift, -uroot + shift, -uroot + shift])
        # else continue to general enumeration

    # D < 0 and effectively real -> three real roots (trig method)
    if D.real < 0 and abs(D.imag) < 1e-12:
        rho = 2.0 * math.sqrt(-p/3.0)
        arg = -q / (2.0 * math.sqrt(-(p/3.0)**3))
        arg = max(-1.0, min(1.0, arg))
        theta = math.acos(arg)
        r1 = rho * math.cos(theta/3.0) + shift
        r2 = rho * math.cos((theta + 2.0*math.pi)/3.0) + shift
        r3 = rho * math.cos((theta + 4.0*math.pi)/3.0) + shift
        return _cleanup_list([r1, r2, r3])

    # General Cardano with enumeration of sqrt and cube branches
    best = None
    best_score = None
    # enumerate sqrt branch (2) and cube branches (3x3)
    for sqrt_branch in (0, 1):
        sD = kth_root(D, 2, sqrt_branch)
        for u_branch in range(3):
            for v_branch in range(3):
                u = kth_root(-q/2.0 + sD, 3, u_branch)
                v = kth_root(-q/2.0 - sD, 3, v_branch)
                omega = cmath.exp(2j * math.pi / 3.0)
                t1 = u + v
                t2 = omega * u + omega**2 * v
                t3 = omega**2 * u + omega * v
                x1 = t1 + shift
                x2 = t2 + shift
                x3 = t3 + shift
                # residual on monic cubic
                def monic_poly(x):
                    return x**3 + A*x**2 + B*x + C
                score = abs(monic_poly(x1))**2 + abs(monic_poly(x2))**2 + abs(monic_poly(x3))**2
                if best_score is None or score < best_score:
                    best_score = score
                    best = [x1, x2, x3]

    # tiny Newton polish to improve residuals (bounded iterations)
    def polish_root(x0):
        x = x0
        for _ in range(6):
            pval = x**3 + A*x**2 + B*x + C
            dp = 3*x**2 + 2*A*x + B
            if abs(dp) < 1e-20:
                break
            dx = pval / dp
            x = x - dx
            if abs(dx) < 1e-14:
                break
        return x

    best = [polish_root(r) for r in best]
    return _cleanup_list(best)
