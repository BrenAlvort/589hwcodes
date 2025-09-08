import cmath
import math
from cubic_solver import solve_cubic


def solve_quartic(a, b, c, d, e):
    if abs(a) < 1e-14:
        return solve_cubic(b, c, d, e)

    # Normalize coefficients
    b /= a
    c /= a
    d /= a
    e /= a

    # Depressed quartic form
    p = c - (3 * b ** 2) / 8
    q = b ** 3 / 8 - b * c / 2 + d
    r = -3 * b ** 4 / 256 + (b ** 2 * c) / 16 - (b * d) / 4 + e

    roots = []

    if abs(q) < 1e-12:
        # Biquadratic case: x⁴ + px² + r = 0
        temp_roots = solve_cubic(1, p, r, 0)
        for y in temp_roots:
            y_sqrt = _trig_sqrt(y)
            roots.append(y_sqrt - b / 4)
            roots.append(-y_sqrt - b / 4)
    else:
        # General case
        # Solve resolvent cubic
        coeffs = [
            1,
            -p / 2,
            -r,
            (4 * r * p - q ** 2) / 8
        ]
        z_roots = solve_cubic(*coeffs)
        z_real = [z.real for z in z_roots if abs(z.imag) < 1e-10]
        z = max(z_real, default=z_roots[0].real)

        u = _trig_sqrt(2 * z - p)
        if abs(u) < 1e-12:
            v = _trig_sqrt(z ** 2 - r)
        else:
            v = q / (2 * u)

        roots.extend([
            (-u - v) / 2 - b / 4,
            (-u + v) / 2 - b / 4,
            (u - v) / 2 - b / 4,
            (u + v) / 2 - b / 4
        ])

    return roots


def _trig_sqrt(z):
    # Approximate sqrt using trig only if z is real and positive
    if isinstance(z, complex) or z < 0:
        return cmath.sqrt(z)  # allowed fallback
    return complex(math.sqrt(z), 0)  # or implement cos identity if needed
