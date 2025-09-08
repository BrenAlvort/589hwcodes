# cubic_solver.py
import cmath
import math

_TOL = 1e-14

def kth_root(z: complex, k: int, branch: int = 0) -> complex:
    """k-th root via exp/log (avoids sqrt/**0.5/**(1/3))."""
    if z == 0:
        return 0j
    return cmath.exp((cmath.log(z) + 2j * math.pi * branch) / k)

def _to_real_if_good(z, poly_func=None, imag_tol=1e-9, resid_tol=1e-8):
    """If z nearly real and residual small, return float; else return as-is."""
    if isinstance(z, complex) and abs(z.imag) < imag_tol:
        xr = float(z.real)
        if poly_func is None:
            return xr
        if abs(poly_func(xr)) < resid_tol:
            return xr
    return z

def _cleanup_list_cubic(roots, poly_func=None):
    return [_to_real_if_good(r, poly_func) for r in roots]

def solve_quadratic(a, b, c):
    """Solve a x^2 + b x + c = 0 robustly, enumerating sqrt branches. Return two roots (floats when real)."""
    if abs(a) < _TOL:
        if abs(b) < _TOL:
            return []
        return [float(-c / b)]
    # normalize to monic x^2 + A x + B = 0
    A = b / a
    B = c / a
    def mon(x): return x*x + A*x + B

    disc = A * A - 4.0 * B
    candidates = []
    for sbranch in (0, 1):
        sd = kth_root(disc, 2, sbranch)
        # numerically stable split
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
    # choose best candidate set by residual
    chosen = min(candidates, key=lambda roots: sum(abs(mon(r))**2 for r in roots))
    out = []
    for r in chosen:
        out.append(_to_real_if_good(r, mon))
    return out

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0 using Cardano with branch enumeration.
    Returns list of 1..3 roots; near-real roots are Python float.
    """
    if abs(a) < _TOL:
        return solve_quadratic(b, c, d)

    # normalize to monic x^3 + A x^2 + B x + C
    A = b / a
    B = c / a
    C = d / a
    def mon(x): return x**3 + A*x**2 + B*x + C

    # depressed cubic t^3 + p t + q, x = t - A/3
    p = B - (A*A)/3.0
    q = (2*A*A*A)/27.0 - (A*B)/3.0 + C
    shift = -A/3.0

    D = (q/2.0)**2 + (p/3.0)**3

    # near-multiple roots
    if abs(D) < 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            return _cleanup_list_cubic([complex(shift), complex(shift), complex(shift)], mon)
        u = -q/2.0
        if abs(u.imag) < 1e-12:
            if u == 0:
                uroot = 0.0
            else:
                uroot = math.copysign(abs(u)**(1.0/3.0), u)
            roots = [2*uroot + shift, -uroot + shift, -uroot + shift]
            return _cleanup_list_cubic([complex(r) for r in roots], mon)

    # three real roots (casus irreducibilis)
    if D.real < 0 and abs(D.imag) < 1e-12:
        rho = 2.0 * math.sqrt(-p/3.0)
        arg = -q / (2.0 * math.sqrt(-(p/3.0)**3))
        arg = max(-1.0, min(1.0, arg))
        theta = math.acos(arg)
        r1 = rho * math.cos(theta/3.0) + shift
        r2 = rho * math.cos((theta + 2.0*math.pi)/3.0) + shift
        r3 = rho * math.cos((theta + 4.0*math.pi)/3.0) + shift
        return _cleanup_list_cubic([complex(r1), complex(r2), complex(r3)], mon)

    # general Cardano with branch enumeration
    best = None
    best_score = None
    omega = cmath.exp(2j * math.pi / 3.0)
    for sqrt_branch in (0,1):
        sD = kth_root(D, 2, sqrt_branch)
        for ub in (0,1,2):
            for vb in (0,1,2):
                u = kth_root(-q/2.0 + sD, 3, ub)
                v = kth_root(-q/2.0 - sD, 3, vb)
                t1 = u + v
                t2 = omega * u + omega**2 * v
                t3 = omega**2 * u + omega * v
                x1 = t1 + shift
                x2 = t2 + shift
                x3 = t3 + shift
                score = abs(mon(x1))**2 + abs(mon(x2))**2 + abs(mon(x3))**2
                if best_score is None or score < best_score:
                    best_score = score
                    best = [x1, x2, x3]

    # small Newton polish (bounded)
    def polish(x0):
        x = x0
        for _ in range(12):
            pval = mon(x)
            dp = 3*x*x + 2*A*x + B
            if abs(dp) < 1e-20:
                break
            dx = pval / dp
            x = x - dx
            if abs(dx) < 1e-14:
                break
        return x

    best = [polish(r) for r in best]
    return _cleanup_list_cubic(best, mon)
