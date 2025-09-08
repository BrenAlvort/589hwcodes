import cmath
import math
from cubic_solver import solve_cubic

def solve_quartic(a, b, c, d, e):
    if abs(a) < 1e-14:
        return solve_cubic(b, c, d, e)

    b /= a
    c /= a
    d /= a
    e /= a

    # Depressed quartic: y⁴ + p*y² + q*y + r
    p = c - 3 * b**2 / 8
    q = b**3 / 8 - b * c / 2 + d
    r = -3 * b**4 / 256 + b**2 * c / 16 - b * d / 4 + e

    roots = []

    if abs(q) < 1e-12:
        # biquadratic equation: y⁴ + py² + r = 0 ⇒ solve as quadratic in y²
        quad_roots = solve_cubic(1, p, r, 0)  # treat as degenerate cubic
        for y2 in quad_roots:
            y = safe_complex_sqrt(y2)
            roots.append(y - b / 4)
            roots.append(-y - b / 4)
    else:
        # Resolvent cubic: z³ - (p/2)z² - r*z + (4r*p - q²)/8 = 0
        cubic_roots = solve_cubic(1, -p/2, -r, (4*r*p - q**2)/8)
        z = max(cubic_roots, key=lambda x: x.real).real

        u = safe_complex_sqrt(2*z - p)
        if abs(u) < 1e-14:
            v = safe_complex_sqrt(z**2 - r)
        else:
            v = q / (2*u)

        roots.append((-u - v) / 2 - b / 4)
        roots.append((-u + v) / 2 - b / 4)
        roots.append((u - v) / 2 - b / 4)
        roots.append((u + v) / 2 - b / 4)

    return roots

def safe_complex_sqrt(z):
    # Use trigonometric identity to compute sqrt
    if isinstance(z, complex) or z < 0:
        return cmath.sqrt(z)
    else:
        return complex(math.cos(math.acos((2 * z - 1)) / 2))  # From cos(2θ)=2cos²θ−1
