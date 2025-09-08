from quartic_solver import solve_quartic
from cubic_solver import solve_cubic


def is_close(a, b, tol=1e-6):
    return abs(a - b) < tol


def poly_eval(coeffs, x):
    result = 0
    for coeff in coeffs:
        result = result * x + coeff
    return result


def test_ladder_problem():
    coeffs = [1, -5000, -135040000, -14400000000, 8294400000000]
    roots = solve_quartic(*coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    assert any(abs(r - 18.0) < 1e-3 for r in real_roots)


def test_quartic_evaluation_accuracy():
    coeffs = [1, 0, -10, 0, 9]
    roots = solve_quartic(*coeffs)
    assert len(roots) == 4
    for r in roots:
        val = poly_eval(coeffs, r)
        assert abs(val) < 1e-5, f"Root {r} failed evaluation: {val}"
