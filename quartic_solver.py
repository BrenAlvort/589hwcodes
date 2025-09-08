# quartic_solver.py
import math
import cmath
from cubic_solver import solve_cubic

def solve_quartic(a, b, c, d, e):
    if abs(a) < 1e-14:
        return solve_cubic(b, c, d, e)

    # Normalize
    b /= a; c /= a; d /= a; e /= a

    # Depressed quartic: y⁴ + p y² + q y + r = 0
    p = c - 3*b*b/8
    q = b**3/8 - b*c/2 + d
    r = -3*b**4/256 + b*b*c/16 - b*d/4 + e

    roots = []
    if abs(q) < 1e-14:
        # Bi-quadratic: y⁴ + p y² + r = 0
        cubic_roots = solve_cubic(1, p, r, 0)  # treat as quadratic in y²
        for z in cubic_roots:
            if z.real >= 0:
                alpha = math.sqrt(z.real)  # replace with cos substitution
                roots += [alpha - b/4, -alpha - b/4]
    else:
        # Ferrari’s method with resolvent cubic
        cubic_roots = solve_cubic(1, -p, -4*r, 4*r*p - q*q)
        z = max(cubic_roots, key=lambda x: x.real).real

        U = math.sqrt(2*z - p)
        V = q / (2*U)
        roots += [(-U - V)/2 - b/4, (-U + V)/2 - b/4,
                  (U - V)/2 - b/4, (U + V)/2 - b/4]

    return [complex(r, 0) if abs(r.imag) < 1e-10 else r for r in roots]
