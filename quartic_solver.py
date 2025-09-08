from cubic_solver import solve_cubic, trig_sqrt


def solve_quartic(a, b, c, d, e):
    if abs(a) < 1e-14:
        return solve_cubic(b, c, d, e)

    # Normalize
    b /= a
    c /= a
    d /= a
    e /= a

    # Depressed quartic: y⁴ + py² + qy + r = 0
    p = c - 3 * b ** 2 / 8
    q = b ** 3 / 8 - b * c / 2 + d
    r = -3 * b ** 4 / 256 + b ** 2 * c / 16 - b * d / 4 + e

    roots = []

    if abs(q) < 1e-12:
        # Biquadratic
        y_roots = solve_cubic(1, p, r, 0)
        for y in y_roots:
            s = trig_sqrt(y.real)
            roots.append(-s - b / 4)
            roots.append(s - b / 4)
    else:
        # Solve resolvent cubic
        coeffs = [1, -p / 2, -r, (4 * r * p - q ** 2) / 8]
        z_roots = solve_cubic(*coeffs)
        z = max((z.real for z in z_roots if abs(z.imag) < 1e-6), default=z_roots[0].real)

        u = trig_sqrt(2 * z - p)
        if abs(u) < 1e-12:
            v = trig_sqrt(z ** 2 - r)
        else:
            v = q / (2 * u)

        roots.append((-u - v) / 2 - b / 4)
        roots.append((-u + v) / 2 - b / 4)
        roots.append((u - v) / 2 - b / 4)
        roots.append((u + v) / 2 - b / 4)

    # Ensure we return 4 roots with multiplicity
    while len(roots) < 4:
        roots.append(roots[-1])

    return roots
