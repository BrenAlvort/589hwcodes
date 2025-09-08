# cubic_solver.py
import math
import cmath

def solve_cubic(a, b, c, d):
    if abs(a) < 1e-14:  # Degenerate: quadratic
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [-d / c]
        # Quadratic bx² + cx + d = 0
        A = c / b
        B = d / b
        # Depressed form: y² = (A² - 4B)/4
        disc = A*A - 4*B
        roots = []
        if disc >= 0:
            # cos substitution
            theta = math.acos((2*B - A*A)/2.0) / 2
            y1 = math.cos(theta)
            y2 = math.cos(theta + math.pi)
            roots = [y1 - A/2, y2 - A/2]
        else:
            # cosh substitution
            u = math.acosh((A*A - 2*B)/2.0) / 2
            y1 = math.cosh(u)
            y2 = -math.cosh(u)
            roots = [y1 - A/2, y2 - A/2]
        return [complex(r, 0) for r in roots]

    # Normalize
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic: y³ + p y + q = 0
    p = B - A**2 / 3
    q = 2 * A**3 / 27 - A * B / 3 + C
    shift = -A / 3

    D = (q/2)**2 + (p/3)**3
    roots = []

    if abs(D) < 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            roots = [shift, shift, shift]
        else:
            u = -q/2
            u = math.copysign(abs(u)**(1/3), u)
            roots = [2*u + shift, -u + shift, -u + shift]
    elif D > 0:
        # one real root via cosh
        sqrtD = math.sqrt(D)
        alpha = math.acosh(-q/(2*math.sqrt(-(p/3)**3))) / 3
        root = 2*math.sqrt(-p/3) * math.cosh(alpha) + shift
        roots = [root]
    else:
        # three real roots via cos
        rho = 2 * math.sqrt(-p/3)
        theta = math.acos(-q/(2*math.sqrt(-(p/3)**3)))
        roots = [
            rho*math.cos(theta/3) + shift,
            rho*math.cos((theta+2*math.pi)/3) + shift,
            rho*math.cos((theta+4*math.pi)/3) + shift,
        ]
    return [complex(r, 0) for r in roots]
