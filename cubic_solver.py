import math
import cmath

# --- Safe wrappers ---
def safe_acos(x):
    return math.acos(max(-1.0, min(1.0, x)))

def safe_acosh(x):
    return math.acosh(max(1.0, x))


def solve_quadratic(a, b, c):
    """Solve ax² + bx + c = 0 using cos/cosh identities."""
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return []
        return [-c / b]

    # Normalize
    A = b / a
    B = c / a

    disc = A * A - 4 * B
    roots = []
    if disc >= -1e-14:  # real roots
        # cos substitution: cos(2θ) = 2cos²θ - 1
        val = (2 * B - A * A) / 2.0
        theta = safe_acos(max(-1, min(1, val)))
        y1 = math.cos(theta / 2)
        y2 = -y1
        roots = [y1 - A / 2, y2 - A / 2]
    else:
        # cosh substitution
        val = (A * A - 2 * B) / 2.0
        u = safe_acosh(val)
        y1 = math.cosh(u / 2)
        y2 = -y1
        roots = [y1 - A / 2, y2 - A / 2]

    return [complex(r, 0) for r in roots]


def solve_cubic(a, b, c, d):
    """Solve ax³ + bx² + cx + d = 0 (cos/cosh method)."""
    if abs(a) < 1e-14:  # Degenerate → quadratic
        return solve_quadratic(b, c, d)

    # Normalize
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic: y³ + p y + q = 0
    p = B - A * A / 3
    q = 2 * A ** 3 / 27 - A * B / 3 + C
    shift = -A / 3

    D = (q / 2) ** 2 + (p / 3) ** 3
    roots = []

    if abs(D) < 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            roots = [shift, shift, shift]
        else:
            u = -q / 2
            u = math.copysign(abs(u) ** (1 / 3), u)
            roots = [2 * u + shift, -u + shift, -u + shift]
    elif D > 0:
        # one real root
        alpha = safe_acosh(-q / (2 * math.sqrt(-(p / 3) ** 3))) / 3
        root = 2 * math.sqrt(-p / 3) * math.cosh(alpha) + shift
        roots = [root]
    else:
        # three real roots
        rho = 2 * math.sqrt(-p / 3)
        theta = safe_acos(-q / (2 * math.sqrt(-(p / 3) ** 3)))
        roots = [
            rho * math.cos(theta / 3) + shift,
            rho * math.cos((theta + 2 * math.pi) / 3) + shift,
            rho * math.cos((theta + 4 * math.pi) / 3) + shift,
        ]
    return [complex(r, 0) for r in roots]
