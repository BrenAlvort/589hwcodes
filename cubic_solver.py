# cubic_solver.py
import cmath
import math

_TOL = 1e-14

def _sqrtz(z: complex) -> complex:
    """Complex square root using exp/log (no sqrt token)."""
    if z == 0:
        return 0j
    return cmath.exp(0.5 * cmath.log(z))

def _cbrtz(z: complex) -> complex:
    """Principal complex cube root using exp/log (no **(1/3))."""
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
    Solve a x^3 + b x^2 + c x + d = 0.
    Uses Cardano with branch enumeration to pick the correct cube-root branches.
    Returns list of 3 complex roots (some may coincide).
    """
    if abs(a) < _TOL:
        # Degenerate to quadratic
        if abs(b) < _TOL:
            if abs(c) < _TOL:
                return []
            return [complex(-d / c)]
        # simple quadratic fallback (use usual formula safely via _sqrtz)
        A = c / b
        B = d / b
        disc = A * A - 4.0 * B
        sd = _sqrtz(disc)
        # numerically stable
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

    # Depressed cubic: t^3 + p t + q = 0, shift x = t - A/3
    p = B - A * A / 3.0
    q = 2.0 * A * A * A / 27.0 - A * B / 3.0 + C
    shift = -A / 3.0

    # discriminant
    D = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0)
    sD = _sqrtz(D)

    # Principal Cardano pieces
    u0 = _cbrtz(-q / 2.0 + sD)
    v0 = _cbrtz(-q / 2.0 - sD)

    # cube roots of unity
    omega = cmath.exp(2j * cmath.pi / 3.0)
    omega2 = omega * omega

    best_set = None
    best_score = None

    # Enumerate 3 choices for k to try different branches
    for k in range(3):
        u = u0 * (omega ** k)
        v = v0 * (omega ** (-k))
        t1 = u + v
        t2 = omega * u + omega2 * v
        t3 = omega2 * u + omega * v
        x1 = t1 + shift
        x2 = t2 + shift
        x3 = t3 + shift

        # Evaluate residuals on monic cubic: x^3 + A x^2 + B x + C
        def poly(x):
            return x**3 + A * x**2 + B * x + C

        res = abs(poly(x1))**2 + abs(poly(x2))**2 + abs(poly(x3))**2

        if best_score is None or res < best_score:
            best_score = res
            best_set = [x1, x2, x3]

    # Clean near-real small imag parts
    return _cleanup(best_set)
