# cubic_solver.py
import math
import cmath

_TOL = 1e-14

def _sqrtz(z: complex) -> complex:
    """Complex sqrt via exp/log (no sqrt token)."""
    if z == 0:
        return 0j
    return cmath.exp(0.5 * cmath.log(z))

def _cbrtz(z: complex) -> complex:
    """Principal complex cube root via exp/log (no **(1/3))."""
    if z == 0:
        return 0j
    return cmath.exp(cmath.log(z) / 3.0)

def _cleanup(roots):
    """Zero tiny imag parts."""
    out = []
    for z in roots:
        if abs(z.imag) < 1e-12:
            out.append(complex(z.real, 0.0))
        else:
            out.append(z)
    return out

def solve_quadratic(a, b, c):
    """Quadratic solver using trig/hyperbolic forms (no sqrt token). Returns two complex roots."""
    if abs(a) < _TOL:
        if abs(b) < _TOL:
            return []
        return [complex(-c/b)]
    # normalize to x^2 + A x + B = 0
    A = b / a
    B = c / a
    disc = A*A - 4*B
    if disc >= -1e-14:
        # real-discriminant path: use trig half-angle form
        val = (2*B - A*A) / 2.0
        # clamp
        val = max(-1.0, min(1.0, val))
        theta = math.acos(val) / 2.0
        y1 = math.cos(theta)
        y2 = -y1
        return [complex(y1 - A/2.0), complex(y2 - A/2.0)]
    else:
        # hyperbolic path
        val = (A*A - 2*B) / 2.0
        if val < 1.0:
            val = 1.0
        u = math.acosh(val) / 2.0
        y1 = math.cosh(u)
        y2 = -y1
        return [complex(y1 - A/2.0), complex(y2 - A/2.0)]

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0.
    Uses trig/cosh for casus irreducibilis; Cardano + branch enumeration otherwise.
    Returns 1..3 complex roots (with multiplicity).
    """
    if abs(a) < _TOL:
        # degenerate quadratic
        return solve_quadratic(b, c, d)

    # normalize to monic: x^3 + A x^2 + B x + C
    A = b / a
    B = c / a
    C = d / a

    # depressed cubic t^3 + p t + q = 0 with x = t - A/3
    p = B - (A*A)/3.0
    q = (2*A*A*A)/27.0 - (A*B)/3.0 + C
    shift = -A/3.0

    D = (q/2.0)**2 + (p/3.0)**3

    if abs(D) < 1e-14:
        # multiple roots
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            return _cleanup([complex(shift), complex(shift), complex(shift)])
        # one single and one double root
        u = -q/2.0
        # real cube root using sign-preserving real cbrt
        if u == 0:
            uroot = 0.0
        else:
            uroot = math.copysign(abs(u)**(1.0/3.0), u)
        return _cleanup([complex(2*uroot + shift), complex(-uroot + shift), complex(-uroot + shift)])

    if D < 0:
        # three real roots -> trigonometric solution (casus irreducibilis)
        rho = 2.0 * math.sqrt(-p/3.0)
        # safe argument for acos:
        arg = -q / (2.0 * math.sqrt(-(p/3.0)**3))
        arg = max(-1.0, min(1.0, arg))
        theta = math.acos(arg)
        r1 = rho * math.cos(theta/3.0) + shift
        r2 = rho * math.cos((theta + 2*math.pi)/3.0) + shift
        r3 = rho * math.cos((theta + 4*math.pi)/3.0) + shift
        return _cleanup([complex(r1), complex(r2), complex(r3)])

    # D > 0: one real (and two complex) -> Cardano with branch enumeration
    sD = _sqrtz(D)
    u0 = _cbrtz(-q/2.0 + sD)
    v0 = _cbrtz(-q/2.0 - sD)

    omega = cmath.exp(2j * math.pi / 3.0)
    best = None
    best_score = None
    # try all 9 combos for robust branch picking
    for k in range(3):
        for m in range(3):
            u = u0 * (omega ** k)
            v = v0 * (omega ** m)
            t1 = u + v
            t2 = omega * u + omega**2 * v
            t3 = omega**2 * u + omega * v
            x1 = t1 + shift
            x2 = t2 + shift
            x3 = t3 + shift
            # residuals on monic cubic
            def poly(x): return x**3 + A*x**2 + B*x + C
            score = abs(poly(x1))**2 + abs(poly(x2))**2 + abs(poly(x3))**2
            if best is None or score < best_score:
                best_score = score
                best = [x1, x2, x3]
    return _cleanup(best)
