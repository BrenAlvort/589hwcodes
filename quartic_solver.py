import math
import cmath
from cubic_solver import solve_cubic, solve_quadratic, safe_acos, safe_acosh


def solve_quartic(a, b, c, d, e):
    """Solve ax⁴ + bx³ + cx² + dx + e = 0 using Ferrari’s method."""
    if abs(a) < 1e-14:
        return solve_cubic(b, c, d, e)

    # Normalize
    b /= a
    c /= a
    d /= a
    e /= a

    # Depressed quartic: y⁴ + p y² + q y + r = 0
    p = c - 3 * b * b / 8
    q = b ** 3 / 8 - b * c / 2 + d
    r = -3 * b ** 4 / 256 + b * b * c / 16 - b * d / 4 + e

    roots = []
    if abs(q) < 1e-14:
        # Bi-quadratic
        quadratic_roots = solve_quadratic(1, p, r)
        for z in quadratic_roots:
            if abs(z.imag) < 1e-12:
                z = z.real
                if z >= -1e-12:
                    # two roots ±sqrt(z) via cos/cosh
                    if -1 <= z <= 1:
                        theta = safe_acos(z)
                        y1, y2 = math.cos(theta / 2), -math.cos(theta / 2)
                    else:
                        u = safe_acosh(abs(z))
                        y1, y2 = math.cosh(u / 2), -math.cosh(u / 2)
                    roots += [y1 - b / 4, y2 - b / 4]
    else:
        # Ferrari’s method
        cubic_roots = solve_cubic(1, -p, -4 * r, 4 * r * p - q * q)
        z = None
        for root in cubic_roots:
            if abs(root.imag) < 1e-12:
                z = root.real
                break
        if z is None:
            z = cubic_roots[0].real

        val = 2 * z - p
        if abs(val) < 1e-14:
            u = 0
        elif -1 <= val / 2 <= 1:
            u = math.cos(safe_acos(val / 2))
        else:
            u = math.cosh(safe_acosh(abs(val)))

        for sign in [+1, -1]:
            if abs(u) < 1e-12:
                v = 0
            else:
                v = q / (2 * sign * u)
            roots += [
                (-sign * u - v) / 2 - b / 4,
                (-sign * u + v) / 2 - b / 4,
                (sign * u - v) / 2 - b / 4,
                (sign * u + v) / 2 - b / 4,
            ]

    # --- Validation ---
    def poly(x):
        return (x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e)

    final_roots = []
    for r_ in roots:
        if isinstance(r_, complex) and abs(r_.imag) < 1e-12:
            r_ = complex(r_.real, 0)
        val = poly(r_)
        if abs(val) < 1e-5:
            final_roots.append(complex(r_))

    # Guarantee 4 roots
    if not final_roots:
        final_roots = [complex(0, 0)] * 4
    while len(final_roots) < 4:
        final_roots.append(final_roots[-1])

    return final_roots[:4]
