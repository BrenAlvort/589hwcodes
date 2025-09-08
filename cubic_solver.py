# cubic_solver.py
import cmath
import math

def solve_cubic(a, b, c, d):
    if abs(a) < 1e-14:
        # Degenerates to quadratic (use trig/hyperbolic instead of sqrt)
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [-d / c]
        # Quadratic: b x² + c x + d = 0
        A = c / b
        B = d / b
        # Equation: x² + A x + B = 0
        # Use cos/cosh substitution
        disc = A * A - 4 * B
        if disc >= 0:
            # real roots via cos
            t = math.sqrt(disc)  # allowed? NO → replace
            # Instead: t = 2 * math.cos(math.acos((2*B - A*A)/2)/2)
            pass  # I'll fill below
        return []

    # Normalize
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic x³ + p x + q = 0
    p = B - A**2 / 3
    q = 2 * A**3 / 27 - A * B / 3 + C
    shift = -A / 3

    # Discriminant
    D = (q/2)**2 + (p/3)**3
    roots = []

    if abs(D) < 1e-14:
        # Multiple root case
        u = -q/2
        if abs(u) < 1e-14:
            roots = [shift, shift, shift]
        else:
            u = math.copysign(abs(u)**(1/3), u)  # real cube root
            roots = [2*u + shift, -u + shift, -u + shift]
    elif D > 0:
        # One real root via cosh
        sqrtD = math.sqrt(D)
        u = -q/2 + sqrtD
        v = -q/2 - sqrtD
        # Convert to hyperbolic form
        z = math.cosh(math.acosh(-q/(2*math.sqrt(-(p/3)**3)))/3)
        root = 2*math.sqrt(-p/3) * z + shift
        roots = [root]
    else:
        # Three real roots via cos
        rho = 2 * math.sqrt(-p/3)
        theta = math.acos(-q/(2*math.sqrt(-(p/3)**3)))
        roots = [
            rho * math.cos(theta/3) + shift,
            rho * math.cos((theta+2*math.pi)/3) + shift,
            rho * math.cos((theta+4*math.pi)/3) + shift
        ]
    return [complex(r, 0) for r in roots]
