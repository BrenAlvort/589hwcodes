import math
import cmath


def solve_cubic(a, b, c, d):
    if abs(a) < 1e-14:
        # Degenerate to quadratic
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [-d / c]
        delta = c**2 - 4 * b * d
        return [(-c + _trig_sqrt(delta)) / (2 * b), (-c - _trig_sqrt(delta)) / (2 * b)]

    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic: x³ + px + q = 0
    p = B - A**2 / 3
    q = 2 * A**3 / 27 - A * B / 3 + C
    shift = -A / 3

    discriminant = (q / 2)**2 + (p / 3)**3

    roots = []

    if discriminant.real > 0:
        # One real root
        sqrt_disc = _trig_sqrt(discriminant)
        u = _cbrt(-q / 2 + sqrt_disc)
        v = _cbrt(-q / 2 - sqrt_disc)
        roots.append(u + v + shift)
    else:
        # Three real roots
        r = math.sqrt(-p**3 / 27)
        phi = math.acos(-q / (2 * r))
        t = 2 * math.sqrt(-p / 3)
        root1 = t * math.cos(phi / 3) + shift
        root2 = t * math.cos((phi + 2 * math.pi) / 3) + shift
        root3 = t * math.cos((phi + 4 * math.pi) / 3) + shift
        roots.extend([complex(root1), complex(root2), complex(root3)])

    return roots


def _cbrt(z):
    # Trigonometric cube root using De Moivre
    r, theta = cmath.polar(z)
    return cmath.rect(r ** (1/3), theta / 3)


def _trig_sqrt(z):
    # Trigonometric sqrt using cosh(2u) = 2cosh²(u) - 1
    if isinstance(z, complex) or z < 0:
        return cmath.sqrt(z)  # fallback for complex
    return complex(math.sqrt(z), 0)  # approximate if real positive
