import cmath
import math

def solve_cubic(a, b, c, d):
    if abs(a) < 1e-14:
        # Degenerate to quadratic
        if abs(b) < 1e-14:
            if abs(c) < 1e-14:
                return []
            return [-d / c]
        delta = c**2 - 4 * b * d
        sqrt_delta = cmath.sqrt(delta)
        return [(-c + sqrt_delta) / (2 * b), (-c - sqrt_delta) / (2 * b)]

    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic form: x³ + px + q = 0
    p = B - A**2 / 3
    q = 2 * A**3 / 27 - A * B / 3 + C
    delta = (q / 2)**2 + (p / 3)**3
    roots = []
    shift = -A / 3

    def cbrt(z):
        if z == 0:
            return 0
        r, theta = cmath.polar(z)
        return cmath.rect(r**(1/3), theta / 3)

    if delta > 1e-14:
        sqrt_delta = cmath.sqrt(delta)
        u = cbrt(-q / 2 + sqrt_delta)
        v = cbrt(-q / 2 - sqrt_delta)
        root1 = u + v + shift
        root2 = -(u + v)/2 + shift + (u - v)*cmath.sqrt(3)/2j
        root3 = -(u + v)/2 + shift - (u - v)*cmath.sqrt(3)/2j
        roots = [root1, root2, root3]
    elif abs(delta) <= 1e-14:
        if abs(q) < 1e-14 and abs(p) < 1e-14:
            roots = [shift, shift, shift]
        else:
            u = cbrt(-q / 2)
            roots = [2 * u + shift, -u + shift, -u + shift]
    else:
        rho = math.sqrt(abs(p)**3 / 27)
        theta = math.acos(max(min(-q / (2 * rho), 1), -1))
        t = 2 * math.sqrt(abs(p) / 3)
        root1 = t * math.cos(theta / 3) + shift
        root2 = t * math.cos((theta + 2 * math.pi) / 3) + shift
        root3 = t * math.cos((theta + 4 * math.pi) / 3) + shift
        roots = [root1, root2, root3]

    return roots  # Keep as complex, don't strip imaginary parts
