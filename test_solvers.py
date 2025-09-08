from quartic_solver import solve_quartic
from cubic_solver import solve_cubic
import cmath

def is_close_complex(a, b, tol=1e-6):
    return abs(a - b) < tol

def test_quadratic():
    # x² - 3x + 2 = 0 → roots: 1, 2
    roots = solve_quartic(0, 0, 1, -3, 2)
    expected = [1.0, 2.0]
    assert len(roots) == len(expected)
    for r in expected:
        assert any(is_close_complex(r, x) for x in roots)

def test_cubic_real():
    # x³ - 6x² + 11x - 6 = 0 → roots: 1, 2, 3
    roots = solve_cubic(1, -6, 11, -6)
    expected = [1.0, 2.0, 3.0]
    assert len(roots) == len(expected)
    for r in expected:
        assert any(is_close_complex(r, x) for x in roots)

def test_cubic_complex():
    # x³ + x + 1 = 0 → 1 real, 2 complex
    roots = solve_cubic(1, 0, 1, 1)
    assert len(roots) == 3
    assert sum(1 for r in roots if abs(r.imag) < 1e-6) == 1
    assert sum(1 for r in roots if abs(r.imag) >= 1e-6) == 2

def test_quartic_real_roots():
    # x⁴ - 10x² + 9 = 0 → roots: ±1, ±3
    roots = solve_quartic(1, 0, -10, 0, 9)
    expected = [-3.0, -1.0, 1.0, 3.0]
    assert len(roots) == len(expected)
    for r in expected:
        assert any(is_close_complex(r, x) for x in roots)

def test_quartic_complex_roots():
    # x⁴ + 2x² + 2 = 0 → all complex
    roots = solve_quartic(1, 0, 2, 0, 2)
    assert len(roots) == 4
    assert all(abs(r.imag) > 1e-6 for r in roots)

def test_ladder_problem_quartic():
    # From the ladder problem
    roots = solve_quartic(1, -5000, -135040000, -14400000000, 8294400000000)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    # Should find a root close to 18
    assert any(abs(r - 18.0) < 1e-3 for r in real_roots)
