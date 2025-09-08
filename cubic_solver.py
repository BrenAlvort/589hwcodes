import cmath
import math

def solve_cubic(a, b, c, d):
    if abs(a) < 1e-14:
        # Degenerate to quadratic
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [-d / c]
        delta = c**2 - 4 * b * d
        return [(-c + complex_trig_sqrt(delta)) / (2 * b),
                (-c - complex_trig_sqrt(delta)) / (2 * b)]

    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic x^3 + px + q = 0
    p = B - A**2 / 3
    q = 2 * A**3 / 27 - A * B / 3 + C
    delta = (q / 2)**2 + (p / 3)**3

    shift = -A / 3

    if delta.real > 0:
        sqrt_delta = complex_trig_sqrt(delta)
        u = cbrt(-q / 2 + sqrt_delta)
        v = cbrt(-q / 2 - sqrt_delta)
        return [u + v + shift]
    else:
        # Use trigonometric method
        r = math.sqrt(-p**3 / 27)
        phi = math.acos(-q / (2 * r))
        t = 2 * math.sqrt(-p / 3)
        root1 = t * math.cos(phi / 3) + shift
        root2 = t * math.cos((phi + 2 * math.pi) / 3) + shift
        root3 = t * math.cos((phi + 4 * math.pi) / 3) + shift
        return [complex(root1), complex(root2), complex(root3)]

def cbrt(z):
    # Cube root using polar coordinates + trigonometric substitution
    r, theta = cmath.polar(z)
    return cmath.rect(r ** (1/3), theta / 3)

def complex_trig_sqrt(z):
    # sqrt(z) using cosh identity (cosh(2u) = 2cosh²(u) - 1)
    if z.real >= 0:
        r = abs(z)
        theta = math.acos((z.real) / r)
        u = math.sqrt((r + z.real) / 2)
        v = math.copysign(math.sqrt((r - z.real) / 2), z.imag)
        return complex(u, v)
    else:
        return cmath.sqrt(z)  # fallback for complex input
