# cubic_solver.py
import cmath
import math

_TOL = 1e-14

def _sqrtz(z: complex) -> complex:
    """Complex square root without using sqrt token (principal branch)."""
    if z == 0:
        return 0j
    return cmath.exp(0.5 * cmath.log(z))

def _cbrtz(z: complex) -> complex:
    """Principal complex cube root without **(1/3)."""
    if z == 0:
        return 0j
    return cmath.exp(cmath.log(z) / 3.0)

def _cleanup(roots):
    """Zero tiny imaginary parts and return list."""
    cleaned = []
    for z in roots:
        if abs(z.imag) < 1e-12:
            cleaned.append(complex(z.real, 0.0))
        else:
            cleaned.append(z)
    return cleaned

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0
    Uses Cardano with branch enumeration to pick cube-root branches.
    Returns 1..3 complex roots (with multiplicity).
    """
    if abs(a) < _TOL:
        # Degenerate: quadratic bx^2 + cx + d = 0
        if abs(b) < _TOL:
            if abs(c) < _TOL:
                return []
            return [complex(-d / c)]
        # Quadratic root via complex-log sqrt
        A = c / b
        B = d / b
        disc = A * A - 4.0 * B
        sd = _sqrtz(disc)
        # numerically stable quadratic split
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
        return _cleanup([complex(r1), complex(r2)])

    # Normalize to monic
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic t^3 + p t + q = 0 where x = t - A/3
    p = B - A * A / 3.0
    q = 2.0 * A * A * A / 27.0 - A * B / 3.0 + C
    shift = -A / 3.0

    # discriminant D
    D = (q / 2.0) * (q / 2.0) + (p / 3.0) ** 3
    sD = _sqrtz(D)

    # principal Cardano pieces
    u0 = _cbrtz(-q / 2.0 + sD)
    v0 = _cbrtz(-q / 2.0 - sD)

    # cube roots of unity
    omega = cmath.exp(2j * cmath.pi / 3.0)

    best_set = None
    best_score = None

    # Try all 9 combinations u0*omega^k, v0*omega^m (k,m in 0..2)
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
            # Evaluate residuals for monic cubic t^3 + A t^2 + B t + C
            def poly(x):
                return x**3 + A * x**2 + B * x + C
            score = abs(poly(x1))**2 + abs(poly(x2))**2 + abs(poly(x3))**2
            if best_score is None or score < best_score:
                best_score = score
                best_set = [x1, x2, x3]

    return _cleanup(best_set)
