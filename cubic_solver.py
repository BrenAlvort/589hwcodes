# cubic_solver.py
import cmath
import math

_TOL = 1e-14

def kth_root(z: complex, k: int, branch: int = 0) -> complex:
    """Return k-th root of z with branch index (0..k-1), using exp/log (no sqrt/**0.5)."""
    if z == 0:
        return 0j
    # principal log + 2π i * branch
    return cmath.exp((cmath.log(z) + 2j * math.pi * branch) / k)

def _cleanup(roots):
    """Zero tiny imaginary parts and return list."""
    out = []
    for z in roots:
        if abs(z.imag) < 1e-12:
            out.append(complex(z.real, 0.0))
        else:
            out.append(z)
    return out

def solve_quadratic(a, b, c):
    """Solve a x^2 + b x + c = 0 using k-th root helper (returns two roots if a!=0)."""
    if abs(a) < _TOL:
        if abs(b) < _TOL:
            return []
        # linear
        return [complex(-c / b)]
    # normalize to monic: x^2 + A x + B = 0
    A = b / a
    B = c / a
    disc = A * A - 4.0 * B
    # enumerate both square-root branches for disc
    candidates = []
    for sbranch in (0, 1):
        sd = kth_root(disc, 2, sbranch)
        # stable quadratic split
        if (A.real >= 0):
            q = -0.5 * (A + sd)
        else:
            q = -0.5 * (A - sd)
        if abs(q) > _TOL:
            r1 = q / 1.0
            r2 = B / q
        else:
            r1 = (-A + sd) / 2.0
            r2 = (-A - sd) / 2.0
        candidates.append([complex(r1), complex(r2)])
    # pick candidate set with smaller residual
    def monic_quad_res(roots):
        Acoef = A
        Bcoef = B
        return sum(abs(r*r + Acoef*r + Bcoef)**2 for r in roots)
    chosen = min(candidates, key=monic_quad_res)
    return _cleanup(chosen)

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0.
    Uses depressed cubic + Cardano with branch enumeration and small polishing.
    Returns up to 3 complex roots (with multiplicity).
    """
    if abs(a) < _TOL:
        # degenerate -> quadratic
        return solve_quadratic(b, c, d)

    # normalize to monic: x^3 + A x^2 + B x + C
    A = b / a
    B = c / a
    C = d / a

    # depressed cubic t^3 + p t + q = 0 with x = t - A/3
    p = B - (A * A) / 3.0
    q = (2.0 * A * A * A) / 27.0 - (A * B) / 3.0 + C
    shift = -A / 3.0

    # discriminant
    D = (q / 2.0) ** 2 + (p / 3.0) ** 3

    # case D ~ 0
    if abs(D) < 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            return _cleanup([complex(shift), complex(shift), complex(shift)])
        # double root case
        u = -q / 2.0
        # real cube root when u is real
        if abs(u.imag) < 1e-12:
            if u == 0:
                uroot = 0.0
            else:
                uroot = math.copysign(abs(u)**(1.0/3.0), u)
            return _cleanup([complex(2*uroot + shift), complex(-uroot + shift), complex(-uroot + shift)])
        # else fallthrough to general enumeration

    # if D < 0 => three real roots (casus irreducibilis) — use trig formula
    if D.real < 0 and abs(D.imag) < 1e-12:
        # safe trig solution
        rho = 2.0 * math.sqrt(-p / 3.0)
        arg = -q / (2.0 * math.sqrt(-(p / 3.0) ** 3))
        # clamp
        arg = max(-1.0, min(1.0, arg))
        theta = math.acos(arg)
        r1 = rho * math.cos(theta / 3.0) + shift
        r2 = rho * math.cos((theta + 2.0 * math.pi) / 3.0) + shift
        r3 = rho * math.cos((theta + 4.0 * math.pi) / 3.0) + shift
        return _cleanup([complex(r1), complex(r2), complex(r3)])

    # General Cardano + branch enumeration
    # enumerate sqrt branches (2) and cube branches (3x3) => at most 18 combos
    best_roots = None
    best_score = None

    for sqrt_branch in (0, 1):
        sD = kth_root(D, 2, sqrt_branch)
        for ubranch in range(3):
            for vbranch in range(3):
                u = kth_root(-q / 2.0 + sD, 3, ubranch)
                v = kth_root(-q / 2.0 - sD, 3, vbranch)
                # construct roots from u,v (with cube roots of unity)
                omega = cmath.exp(2j * math.pi / 3.0)
                t1 = u + v
                t2 = omega * u + omega**2 * v
                t3 = omega**2 * u + omega * v
                x1 = t1 + shift
                x2 = t2 + shift
                x3 = t3 + shift
                # polish lightly with Newton to reduce residual
                def monic_poly(x):
                    return x**3 + A * x**2 + B * x + C
                # compute residual score
                score = abs(monic_poly(x1))**2 + abs(monic_poly(x2))**2 + abs(monic_poly(x3))**2
                if best_score is None or score < best_score:
                    best_score = score
                    best_roots = [x1, x2, x3]

    # polish chosen roots with tiny Newton (1-6 iterations) to improve accuracy
    def newton_polish_root(x0):
        x = x0
        for _ in range(6):
            # monic cubic and derivative
            p = x**3 + A * x**2 + B * x + C
            dp = 3 * x**2 + 2 * A * x + B
            if abs(dp) < 1e-20:
                break
            dx = p / dp
            x = x - dx
            if abs(dx) < 1e-14:
                break
        if abs(x.imag) < 1e-12:
            x = complex(x.real, 0.0)
        return x

    best_roots = [newton_polish_root(r) for r in best_roots]
    return _cleanup(best_roots)
