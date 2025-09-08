import cmath
import math
from cubic_solver import solve_cubic

def solve_quartic(a, b, c, d, e):
    if abs(a) < 1e-14:
        return solve_cubic(b, c, d, e)

    # Normalize
    b /= a
    c /= a
    d /= a
    e /= a

    # Depressed quartic coefficients
    p = c - 3 * b**2 / 8
    q = b**3 / 8 - b * c / 2 + d
    r = -3 * b**4 / 256 + b**2 * c / 16 - b * d / 4 + e
    roots = []

    if abs(q) < 1e-14:
        discriminant = p**2 - 4 * r
        sqrt_disc = cmath.sqrt(discriminant)
        y1 = (-p + sqrt_disc) / 2
        y2 = (-p - sqrt_disc) / 2
        for y in [y1, y2]:
            sqrt_y = cmath.sqrt(y)
            roots.append(sqrt_y - b / 4)
            roots.append(-sqrt_y - b / 4)
    else:
        # Solve resolvent cubic
        cubic_roots = solve_cubic(8, -4 * p, -8 * r, 4 * r * p - q**2)

        # Pick the best root
        real_roots = [z for z in cubic_roots if abs(z.imag) < 1e-10]
        z = max(real_roots, key=lambda x: x.real) if real_roots else max(cubic_roots, key=lambda x: x.real)
        z = z.real

        u = cmath.sqrt(2 * z - p)
        if abs(u) < 1e-14:
            v = cmath.sqrt(z**2 - r)
        else:
            v = q / (2 * u)

        roots.append((-u - v) / 2 - b / 4)
        roots.append((-u + v) / 2 - b / 4)
        roots.append((u - v) / 2 - b / 4)
        roots.append((u + v) / 2 - b / 4)

    return roots  # Keep complex parts if needed
