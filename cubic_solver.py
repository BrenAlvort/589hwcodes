import math
import cmath

# --- Safe wrappers to avoid domain errors ---
def safe_acos(x):
    return math.acos(max(-1, min(1, x)))

def safe_acosh(x):
    return math.acosh(max(1, x))


def solve_cubic(a, b, c, d):
    """
    Solve ax^3 + bx^2 + cx + d = 0
    using only cos/cosh substitutions (no radicals).
    Returns list of 1–3 complex roots.
    """
    if abs(a) < 1e-14:  # Degenerate: quadratic
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [-d / c]
        # Quadratic bx² + cx + d = 0
        A = c / b
        B = d / b
        disc = A * A - 4 * B
        roots = []
        if disc >= 0:
            # cos substitution
            val = (2 * B - A * A) / 2.0
            theta = safe_acos(val) / 2
            y1 = math.cos(theta)
            y2 = math.cos(theta + math.pi)
            roots = [y1 - A / 2, y2 - A / 2]
        else:
            # cosh substitution
            val = (A * A - 2 * B) / 2.0
            u = safe_acosh(val) / 2
            y1 = math.cosh(u)
            y2 = -math.cosh(u)
            roots = [y1 - A / 2, y2 - A / 2]
        return [complex(r, 0) for r in roots]

    # Normalize
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic: y³ + p y + q = 0
    p = B - A ** 2 / 3
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
        # one real root via cosh
        alpha = safe_acosh(-q / (2 * math.sqrt(-(p / 3) ** 3))) / 3
        root = 2 * math.sqrt(-p / 3) * math.cosh(alpha) + shift
        roots = [root]
    else:
        # three real roots via cos
        rho = 2 * math.sqrt(-p / 3)
        theta = safe_acos(-q / (2 * math.sqrt(-(p / 3) ** 3)))
        roots = [
            rho * math.cos(theta / 3) + shift,
            rho * math.cos((theta + 2 * math.pi) / 3) + shift,
            rho * math.cos((theta + 4 * math.pi) / 3) + shift,
        ]
    return [complex(r, 0) for r in roots]
