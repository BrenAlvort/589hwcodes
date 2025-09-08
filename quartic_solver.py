import cmath

def solve_quartic(a, b, c, d, e):
    # Normalize to monic form
    if a == 0:
        return solve_cubic(b, c, d, e)
    b, c, d, e = b/a, c/a, d/a, e/a

    # Special case: biquadratic (x^4 + c*x^2 + e = 0)
    if abs(b) < 1e-12 and abs(d) < 1e-12:
        y_roots = solve_quadratic(1, c, e)
        roots = []
        for y in y_roots:
            sq = cmath.sqrt(y)
            roots += [sq, -sq]
        return polish_and_dedup(roots, [1, b, c, d, e])

    # Ferrari’s method
    p = c - (3*b*b)/8
    q = d - (b*c)/2 + (b**3)/8
    r = e - (b*d)/4 + (b*b*c)/16 - (3*b**4)/256

    cubic_roots = solve_cubic(1, -p/2, -r, (4*r*p - q*q)/8)
    y = max(cubic_roots, key=lambda z: z.real).real

    R = cmath.sqrt(max(0, y) + 0j)
    if abs(R) < 1e-12:
        D = cmath.sqrt(y*y - 4*r)
        E = cmath.sqrt(y*y - 4*r)
        roots = [ -b/4 + (D+E)/2, -b/4 + (D-E)/2,
                  -b/4 + (-D+E)/2, -b/4 + (-D-E)/2 ]
    else:
        D = cmath.sqrt(2*y - p + (q/R))
        E = cmath.sqrt(2*y - p - (q/R))
        roots = [ -b/4 + (R+D)/2, -b/4 + (R-D)/2,
                  -b/4 + (-R+E)/2, -b/4 + (-R-E)/2 ]

    return polish_and_dedup(roots, [1, b, c, d, e])

def solve_cubic(a, b, c, d):
    if abs(a) < 1e-12:
        return solve_quadratic(b, c, d)
    b, c, d = b/a, c/a, d/a
    p = c - b*b/3
    q = (2*b*b*b)/27 - (b*c)/3 + d
    Δ = (q/2)**2 + (p/3)**3

    if Δ >= 0:
        u = cbrt(-q/2 + cmath.sqrt(Δ))
        v = cbrt(-q/2 - cmath.sqrt(Δ))
        roots = [u+v - b/3]
    else:
        r = cmath.sqrt(-(p**3)/27)
        phi = cmath.acos(-q/(2*r))
        m = 2*cmath.sqrt(-p/3)
        roots = [m*cmath.cos(phi/3) - b/3,
                 m*cmath.cos((phi+2*cmath.pi)/3) - b/3,
                 m*cmath.cos((phi+4*cmath.pi)/3) - b/3]
    return roots

def solve_quadratic(a, b, c):
    if abs(a) < 1e-12:
        return [-c/b] if abs(b) > 1e-12 else []
    Δ = cmath.sqrt(b*b - 4*a*c)
    return [(-b+Δ)/(2*a), (-b-Δ)/(2*a)]

def cbrt(z):
    return abs(z)**(1/3) * cmath.exp(1j*cmath.phase(z)/3) if z else 0

def polish_and_dedup(roots, coeffs, tol=1e-7):
    # Newton correction
    def f(x):
        return coeffs[0]*x**4 + coeffs[1]*x**3 + coeffs[2]*x**2 + coeffs[3]*x + coeffs[4]
    def df(x):
        return 4*coeffs[0]*x**3 + 3*coeffs[1]*x**2 + 2*coeffs[2]*x + coeffs[3]

    polished = []
    for r in roots:
        for _ in range(2):  # 2 Newton steps
            denom = df(r)
            if abs(denom) < 1e-12: break
            r -= f(r)/denom
        # Clean small imaginary parts
        if abs(r.imag) < tol: r = r.real
        polished.append(r)

    # Deduplicate within tolerance
    unique = []
    for r in polished:
        if not any(abs(r - u) < tol for u in unique):
            unique.append(r)
    # Ensure exactly 4 roots (with multiplicity if needed)
    while len(unique) < 4:
        unique.append(unique[-1])
    return unique[:4]
