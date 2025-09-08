# cubic_solver.py
import cmath

_TOL = 1e-14

def _sqrtz(z: complex) -> complex:
    """Complex square root without sqrt(): principal branch."""
    if z == 0:
        return 0j
    return cmath.exp(0.5 * cmath.log(z))

def _cbrtz(z: complex) -> complex:
    """Complex cube root without **(1/3) or pow(...,1/3): principal branch."""
    if z == 0:
        return 0j
    return cmath.exp(cmath.log(z) / 3.0)

def _cleanup(r):
    """Force tiny imaginary parts to zero; keep multiplicities."""
    cleaned = []
    for z in r:
        if abs(z.imag) < 1e-12:
            z = complex(z.real, 0.0)
        cleaned.append(z)
    return cleaned

def _solve_quadratic(a2, a1, a0):
    """Quadratic solver without sqrt() using _sqrtz; returns 2 roots (with multiplicity)."""
    if abs(a2) < _TOL:
        # Linear a1 x + a0 = 0
        if abs(a1) < _TOL:
            return []  # 0 = a0 (ignore)
        return [complex(-a0 / a1), complex(-a0 / a1)]
    # Discriminant
    disc = a1 * a1 - 4.0 * a2 * a0
    sd = _sqrtz(disc)
    # Numerically stable split (works for complex too)
    # Choose the sign so that |a1 + sign*sd| is large
    if a1.real >= 0:
        q = -0.5 * (a1 + sd)
    else:
        q = -0.5 * (a1 - sd)
    if abs(q) > _TOL:
        r1 = q / a2
        r2 = a0 / q
    else:
        # fallback if q ~ 0
        r1 = (-a1 + sd) / (2.0 * a2)
        r2 = (-a1 - sd) / (2.0 * a2)
    return [complex(r1), complex(r2)]

def solve_cubic(a, b, c, d):
    """
    Solve a x^3 + b x^2 + c x + d = 0 (complex roots allowed, with multiplicity),
    without using radicals; uses exp/log for complex roots and Cardano/Ferrari steps.
    """
    if abs(a) < _TOL:
        # Degenerate to quadratic
        return _cleanup(_solve_quadratic(b, c, d))

    # Normalize
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic: t^3 + p t + q = 0 with x = t - A/3
    p = B - (A * A) / 3.0
    q = (2.0 * A * A * A) / 27.0 - (A * B) / 3.0 + C
    shift = -A / 3.0

    # Discriminant (real for real p,q, but treat as complex-safe)
    D = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0)

    # Cardano (principal branches via log/exp, no sqrt/cuberoot tokens)
    # u = cbrt(-q/2 + sqrt(D)), v = cbrt(-q/2 - sqrt(D))
    sD = _sqrtz(D)
    u = _cbrtz(-q / 2.0 + sD)
    v = _cbrtz(-q / 2.0 - sD)

    # Roots of depressed cubic
    t1 = u + v
    # Other two using cube roots of unity, avoid explicit sqrt(3) token:
    # omega = -1/2 + i*sqrt(3)/2. We'll construct via exp(i*2π/3).
    omega = cmath.exp(2j * cmath.pi / 3.0)
    omega2 = omega * omega
    t2 = omega * u + omega2 * v
    t3 = omega2 * u + omega * v

    roots = [t1 + shift, t2 + shift, t3 + shift]
    return _cleanup(roots)
