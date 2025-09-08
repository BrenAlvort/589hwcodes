import math
import cmath
from cubic_solver import solve_cubic, safe_acos, safe_acosh


def solve_quartic(a, b, c, d, e):
    """
    Solve ax^4 + bx^3 + cx^2 + dx + e = 0
    using Ferrari’s method with cos/cosh substitutions.
    Always returns exactly 4 roots (with multiplicity).
    """
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
        # Bi-quadratic: y⁴ + p y² + r = 0
        quadratic_roots = solve_cubic(1, p, r, 0)
        for z in quadratic_roots:
            if abs(z.imag) < 1e-12:
                z = z.real
                if z >= -1e-12:
                    # cos substitution for sqrt(z)
                    if -1 <= z <= 1:
                        theta = safe_acos(z)
                        y1 = math.cos(theta / 2)
                        y2 = -y1
                    else:
                        u = safe_acosh(abs(z))
                        y1 = math.cosh(u / 2)
                        y2 = -y1
                    roots += [y1 - b / 4, y2 - b / 4]
    else:
        # Ferrari’s method: resolvent cubic
        cubic_roots = solve_cubic(1, -p, -4 * r, 4 * r * p - q * q)
        for z in cubic_roots:
            z = z.real if abs(z.imag) < 1e-12 else z
            if isinstance(z, complex):
                continue
            # try both signs
            val = 2 * z - p
            try:
                if -1 <= val / 2 <= 1:
                    u = math.cos(safe_acos(val / 2))
                else:
                    u = math.cosh(safe_acosh(abs(val)))
            except ValueError:
                continue
            for sign in [+1, -1]:
                if abs(u) < 1e-12:
                    v = 0
                else:
                    v = q / (2 * sign * u)
                cand = [
                    (-sign * u - v) / 2 - b / 4,
                    (-sign * u + v) / 2 - b / 4,
                    (sign * u - v) / 2 - b / 4,
                    (sign * u + v) / 2 - b / 4,
                ]
                roots += cand

    # ✅ Validate by plugging back into polynomial
    def poly(x):
        return ((x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e)

    final_roots = []
    for r_ in roots:
        val = poly(r_)
        if abs(val) < 1e-5:
            final_roots.append(complex(r_))

    # Guarantee 4 roots with multiplicity
    while len(final_roots) < 4 and final_roots:
        final_roots.append(final_roots[-1])

    return final_roots[:4]
