import math
import cmath


def solve_cubic(a, b, c, d):
    if abs(a) < 1e-14:
        # Quadratic
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [complex(-d / c)] * 1
        return solve_quadratic(b, c, d)

    # Normalize
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic: x³ + px + q = 0
    p = B - A ** 2 / 3
    q = (2 * A ** 3) / 27 - (A * B) / 3 + C

    shift = -A / 3

    discriminant = (q / 2) ** 2 + (p / 3) ** 3

    if discriminant > 0:
        # One real root
        sqrt_disc = trig_sqrt(discriminant)
        u = trig_cbrt(-q / 2 + sqrt_disc)
        v = trig_cbrt(-q / 2 - sqrt_disc)
        root = u + v + shift
        return [root] * 1 + [root.conjugate()] * 2 if abs(root.imag) > 1e-8 else [root]

    elif abs(discriminant) < 1e-12:
        # Triple root or double + single
        if abs(q) < 1e-12:
            root = shift
            return [root] * 3
        else:
            u = trig_cbrt(-q / 2)
            return [2 * u + shift, -u + shift, -u + shift]

    else:
        # Three real roots (discriminant < 0)
        r = math.sqrt(-p ** 3 / 27)
        phi = math.acos(-q / (2 * r))
        t = 2 * math.sqrt(-p / 3)
        roots = [
            complex(t * math.cos(phi / 3) + shift),
            complex(t * math.cos((phi + 2 * math.pi) / 3) + shift),
            complex(t * math.cos((phi + 4 * math.pi) / 3) + shift),
        ]
        return roots


def solve_quadratic(a, b, c):
    if abs(a) < 1e-14:
        return [-c / b] if abs(b) > 1e-14 else []
    discriminant = b ** 2 - 4 * a * c
    sqrt_disc = trig_sqrt(discriminant)
    return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]


def trig_cbrt(x):
    # Cube root using trig/hyperbolic
    if x >= 0:
        return complex(math.exp(math.log(x) / 3))
    else:
        return complex(-math.exp(math.log(-x) / 3))


def trig_sqrt(x):
    if x >= 0:
        # Use cosh identity: cosh(2u) = 2cosh²(u) - 1
        cosh_2u = 1 + 2 * x  # reverse of identity
        u = math.acosh(cosh_2u) / 2
        return complex(math.cosh(u))
    else:
        # Use cos identity: cos(2θ) = 2cos²θ - 1
        cos_2theta = 1 + 2 * x  # x < 0
        theta = math.acos(cos_2theta) / 2
        return complex(0, math.sin(theta))  # purely imaginary
