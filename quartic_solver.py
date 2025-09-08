import cmath

def evaluate_poly(coeffs, x):
    """Evaluate polynomial with given coefficients at point x."""
    return sum(c * (x ** i) for i, c in enumerate(reversed(coeffs)))


def normalize_root(z, tol=1e-8):
    """Round tiny imaginary parts to 0, and round floats for stability."""
    if abs(z.imag) < tol:
        z = complex(round(z.real, 12), 0.0)
    else:
        z = complex(round(z.real, 12), round(z.imag, 12))
    return z


def sort_roots(roots):
    """Sort roots safely by real then imaginary parts (avoids comparing complex directly)."""
    return sorted(roots, key=lambda z: (round(z.real, 8), round(z.imag, 8)))


def root_multiplicity(coeffs, r, tol=1e-7):
    """
    Estimate multiplicity of a root r by synthetic division.
    Repeatedly divide polynomial by (x - r) while remainder ≈ 0.
    """
    mult = 0
    poly = coeffs[:]
    n = len(poly) - 1

    while n > 0:
        new_poly = [poly[0]]
        for i in range(1, len(poly) - 1):
            new_poly.append(poly[i] + new_poly[-1] * r)
        remainder = poly[-1] + new_poly[-1] * r
        if abs(remainder) > tol:
            break
        poly = new_poly
        n -= 1
        mult += 1

    return max(1, mult)


def solve_quartic(a, b, c, d, e):
    """
    Solve quartic equation: a*x^4 + b*x^3 + c*x^2 + d*x + e = 0.
    Returns list of roots (with multiplicity if applicable).
    """
    if abs(a) < 1e-14:
        raise ValueError("Not a quartic equation")

    # Normalize coefficients
    a1, b1, c1, d1 = b / a, c / a, d / a, e / a

    # Depressed quartic substitution: x = y - a1/4
    p = c1 - (3 * a1 ** 2) / 8
    q = (a1 ** 3) / 8 - (a1 * c1) / 2 + d1
    r = -(3 * a1 ** 4) / 256 + (a1 ** 2 * c1) / 16 - (a1 * d1) / 4 + e / a

    # Resolvent cubic: y^3 + (p/2) y^2 + ((p^2 - 4r)/16) y - q^2/64 = 0
    cubic_a = 1
    cubic_b = p / 2
    cubic_c = (p ** 2 - 4 * r) / 16
    cubic_d = -(q ** 2) / 64

    from cubic_solver import solve_cubic
    y_roots = solve_cubic(cubic_a, cubic_b, cubic_c, cubic_d)
    y = max(y_roots, key=lambda x: x.real if isinstance(x, complex) else x)

    R = cmath.sqrt(0.25 * a1 ** 2 - b1 + y)
    D = cmath.sqrt(3.0 / 4 * a1 ** 2 - R ** 2 - 2 * b1 +
                   (4 * a1 * b1 - 8 * c1 - a1 ** 3) / (4 * R)) if R != 0 else 0
    E = cmath.sqrt(3.0 / 4 * a1 ** 2 - R ** 2 - 2 * b1 -
                   (4 * a1 * b1 - 8 * c1 - a1 ** 3) / (4 * R)) if R != 0 else 0

    roots = []
    shift = -a1 / 4
    roots.extend([shift + 0.5 * (R + D),
                  shift + 0.5 * (R - D),
                  shift + 0.5 * (-R + E),
                  shift + 0.5 * (-R - E)])

    # Normalize and expand multiplicity
    coeffs = [a, b, c, d, e]
    expanded_roots = []
    for r in roots:
        r = normalize_root(r)
        mult = root_multiplicity(coeffs, r)
        expanded_roots.extend([r] * mult)

    return sort_roots(expanded_roots)
