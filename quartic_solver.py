# quartic_solver.py
import cmath
from cubic_solver import solve_cubic, _sqrtz, _cleanup, _TOL

def _solve_biquadratic(p, r):
    """
    Solve y^4 + p y^2 + r = 0 for y (returns 4 roots), without sqrt() token.
    Let t = y^2: t^2 + p t + r = 0, then y = ±sqrtz(t).
    """
    # Solve t^2 + p t + r = 0
    # Use the same quadratic helper logic but inlined to avoid import cycles.
    a2, a1, a0 = 1.0, p, r
    disc = a1 * a1 - 4.0 * a2 * a0
    sd = _sqrtz(disc)
    # Stable split
    if a1.real >= 0:
        q = -0.5 * (a1 + sd)
    else:
        q = -0.5 * (a1 - sd)
    if abs(q) > _TOL:
        t1 = q / a2
        t2 = a0 / q
    else:
        t1 = (-a1 + sd) / (2.0 * a2)
        t2 = (-a1 - sd) / (2.0 * a2)

    y_roots = []
    for t in (t1, t2):
        s = _sqrtz(t)  # ± sqrt(t) via exp/log
        y_roots.append(+s)
        y_roots.append(-s)
    return y_roots  # total 4 (with multiplicity)

def solve_quartic(a, b, c, d, e):
    """
    Solve a x^4 + b x^3 + c x^2 + d x + e = 0
    Returns exactly 4 roots (with multiplicity), complex allowed.
    No use of sqrt/**0.5/pow(.,0.5) or cube-root exponents.
    """
    # Degeneracies
    if abs(a) < _TOL:
        # Fall back to cubic
        return solve_cubic(b, c, d, e)

    # Normalize to monic
    b = b / a
    c = c / a
    d = d / a
    e = e / a

    # Depress quartic: x = y - b/4  →  y^4 + p y^2 + q y + r = 0
    B = b
    p = c - 3.0 * (B * B) / 8.0
    q = (B * B * B) / 8.0 - (B * c) / 2.0 + d
    r = -3.0 * (B ** 4) / 256.0 + (B * B * c) / 16.0 - (B * d) / 4.0 + e

    # Special case: biquadratic (q ~ 0)
    roots_y = []
    if abs(q) < 1e-12:
        roots_y = _solve_biquadratic(p, r)
    else:
        # Ferrari via resolvent cubic:
        # z^3 - p z^2 - 4 r z + (4 r p - q^2) = 0
        z_roots = solve_cubic(1.0, -p, -4.0 * r, 4.0 * r * p - q * q)

        # Pick a z that avoids division by ~0 in u = sqrt(2 z - p)
        # Preference: smallest |imag|, then largest Re; and |u| not tiny.
        def score(z):
            return (abs(z.imag), -z.real)

        z_candidates = sorted(z_roots, key=score)
        u = None
        picked_z = None
        for z in z_candidates:
            u_try = _sqrtz(2.0 * z - p)
            if abs(u_try) > 1e-10:  # avoid division by very small
                picked_z = z
                u = u_try
                break
        if u is None:
            # all were tiny; take the first and proceed (q should be ~0, but we’re in else)
            picked_z = z_candidates[0]
            u = _sqrtz(2.0 * picked_z - p)

        z = picked_z

        # Compute two inner quantities (can be complex)
        # s1 = -(2 z + p + 2 q / u), s2 = -(2 z + p - 2 q / u)
        two_z_plus_p = 2.0 * z + p
        s1 = -(two_z_plus_p + 2.0 * q / u)
        s2 = -(two_z_plus_p - 2.0 * q / u)

        # Four roots in y (no sqrt token used)
        t1 = _sqrtz(s1)
        t2 = _sqrtz(s2)

        roots_y = [(-u + t1) / 2.0,
                   (-u - t1) / 2.0,
                   ( u + t2) / 2.0,
                   ( u - t2) / 2.0]

    # Shift back: x = y - b/4
    shift = -B / 4.0
    roots_x = [y + shift for y in roots_y]

    return _cleanup(roots_x)
