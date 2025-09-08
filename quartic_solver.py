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
        # Bi-quadratic y⁴ + p y² + r = 0 → solve y²
        quadratic_roots = solve_cubic(1, p, r, 0)
        for z in quadratic_roots:
            if abs(z.imag) < 1e-12:
                z = z.real
                if z >= 0:
                    # use cos/cosh instead of sqrt
                    if z <= 1:
                        theta = math.acos(z)
                        y1 = math.cos(theta/2)
                        y2 = math.cos((theta+2*math.pi)/2)
                    else:
                        u = math.acosh(z)
                        y1 = math.cosh(u/2)
                        y2 = -math.cosh(u/2)
                    roots += [y1 - b/4, -y1 - b/4, y2 - b/4, -y2 - b/4]
    else:
        # Ferrari’s method
        cubic_roots = solve_cubic(1, -p, -4*r, 4*r*p - q*q)
        z = max(cubic_roots, key=lambda x: x.real).real

        if 2*z - p >= 0:
            u = math.sqrt(2*z - p)
        else:
            u = 0.0

        if abs(u) < 1e-14:
            v = math.sqrt(z*z - r)
        else:
            v = q / (2*u)

        roots += [(-u - v)/2 - b/4, (-u + v)/2 - b/4,
                  (u - v)/2 - b/4, (u + v)/2 - b/4]

    return [complex(r, 0) if abs(r.imag) < 1e-10 else r for r in roots]
