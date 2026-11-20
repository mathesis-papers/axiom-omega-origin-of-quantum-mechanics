"""Checks for module n11 (Appendix A: independence models, Axiom A_0, quadratic consequences,
interpretations).  Path-free; NumPy and SymPy only; deterministic seeds.

Run:  python3 check_independence.py
"""
import itertools
from fractions import Fraction

import numpy as np
import sympy as sp

TOL = 1e-11
RNG = np.random.default_rng(20261120)


def close(a, b, tol=TOL):
    return np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), atol=tol, rtol=0)


# ----------------------------------------------------------------------------- helpers
def quat(v):
    """unit quaternion a+bi+cj+dk as the 2x2 complex matrix [[a+bi, c+di],[-c+di, a-bi]]"""
    a, b, c, d = v
    return np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]])


def rand_unit(n):
    v = RNG.normal(size=n)
    return v / np.linalg.norm(v)


def qconj(q):
    return q.conj().T


def qre(q):
    return np.real(np.trace(q)) / 2


def clifford_gammas(n):
    """Hermitian matrices g_1..g_n with g_i g_j + g_j g_i = 2 delta_ij (Jordan--Wigner)."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    m = (n + 1) // 2
    gs = []
    for k in range(m):
        for P in (X, Y):
            ops = [Z] * k + [P] + [I] * (m - k - 1)
            M = ops[0]
            for o in ops[1:]:
                M = np.kron(M, o)
            gs.append(M)
    return gs[:n]


def vec(x, gs):
    return sum(xi * g for xi, g in zip(x, gs))


def shannon(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def hbin(p):
    return shannon([p, 1 - p])


def vn_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    return shannon(np.clip(ev, 0, None))


def rand_density(d, rank=None):
    rank = rank or d
    G = RNG.normal(size=(d, rank)) + 1j * RNG.normal(size=(d, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def proj_r(r):
    return (I2 + r[0] * SX + r[1] * SY + r[2] * SZ) / 2


# ----------------------------------------------------------------------------- groups
def group_a0_realisation():
    """Omega gives A_0 with (IV): u(x,y)=x conj(y), symmetric part = c(x,y) 1."""
    for _ in range(200):
        x, y, z = (quat(rand_unit(4)) for _ in range(3))
        u = lambda a, b: a @ qconj(b)
        assert close(u(y, x), qconj(u(x, y)))                       # (II)
        assert close(u(x, y) @ u(y, x), np.eye(2))                   # (III)
        assert close(u(x, y) @ u(y, z), u(x, z))                     # (IV)
        up = (u(x, y) + qconj(u(x, y))) / 2
        c = qre(qconj(x) @ y)                                        # c = Re(conj(x) y)
        assert close(up, c * np.eye(2))                              # u_+ = c 1
        um = (u(x, y) - qconj(u(x, y))) / 2
        assert close(up @ up - um @ um, np.eye(2))                   # u_+^2 - u_-^2 = 1
        assert close(up @ um, um @ up)
    # (I): equal rows at z = x force y = x
    x = quat(rand_unit(4))
    y = quat(rand_unit(4))
    assert not close(x @ qconj(x), y @ qconj(x))


def group_converses():
    """S^1 with k = kappa^2: (Omega1),(Omega2) hold, (Omega3) fails; recoded unit comparison."""
    kap = lambda x, y: (1 + np.real(x * np.conj(y))) / 2
    k = lambda x, y: kap(x, y) ** 2
    th = np.linspace(0, 2 * np.pi, 721)[:-1]
    pts = np.exp(1j * th)
    K = k(pts[:, None], pts[None, :])
    assert np.allclose(K, K.T)                                       # (Omega2a)
    assert np.all((K >= 0) & (K <= 1 + 1e-15))
    off = ~np.eye(len(pts), dtype=bool)
    assert np.all(K[off] < 1)                                        # (Omega2b)
    for i in range(0, len(pts), 37):                                 # (Omega2d): unique zero -x
        zeros = np.where(np.abs(K[i]) < 1e-14)[0]
        assert len(zeros) == 1 and close(pts[zeros[0]], -pts[i])
    # (Omega2c): heap [x,u,v] = x conj(u) v; invariance in both slots
    for _ in range(100):
        x, y, a, b = np.exp(1j * RNG.uniform(0, 2 * np.pi, 4))
        assert close(k(x * np.conj(a) * b, y * np.conj(a) * b), k(x, y))
        assert close(k(a * np.conj(b) * x, a * np.conj(b) * y), k(x, y))
    # (Omega3) fails at w = 1, x = i
    w, x = 1 + 0j, 1j
    s = x * np.conj(w) * x
    assert close(s, -1)
    assert close(k(w, s), 0) and close((2 * k(w, x) - 1) ** 2, 0.25)
    assert close(k(w, x), 0.25)
    # recoded unit comparison
    def u(x, y):
        sv = 2 * k(x, y) - 1
        return sv + 1j * np.sign(np.imag(x * np.conj(y))) * np.sqrt(np.clip(1 - sv ** 2, 0, None))
    for _ in range(300):
        x, y = np.exp(1j * RNG.uniform(0, 2 * np.pi, 2))
        assert close(abs(u(x, y)), 1)                                # (III) with u* = conj
        assert close(u(y, x), np.conj(u(x, y)))                      # (II)
        assert close(np.real(u(x, y)), 2 * k(x, y) - 1)              # u_+ = 2k - 1
    # continuity across Im(x conj y) = 0: at x conj y = +-1 the imaginary part vanishes
    for base in (1.0, -1.0):
        for eps in (1e-4, -1e-4, 1e-7, -1e-7):
            z = base * np.exp(1j * eps)
            assert abs(u(z, 1) - (1.0 if base > 0 else -1.0)) < 1e-3
    # u(x,z)=1 iff x=z
    for i in range(1, len(pts), 13):
        assert abs(u(pts[i], pts[0]) - 1) > 1e-6
    assert close(u(pts[5], pts[5]), 1)


def group_clauses():
    """Independence of the clauses of A_0 and of invertibility of 2."""
    # two points, constant u = 1: (II),(III), rows equal
    U = np.ones((2, 2))
    assert np.allclose(U, U.T) and np.allclose(U * U.T, 1)
    assert np.allclose(U[0], U[1])
    # real values 2, 1/2 with identity involution: (III) holds, (II) fails
    U = np.array([[1, 2], [0.5, 1]])
    assert np.allclose(U * U.T, 1) and not np.allclose(U, U.T)
    assert not np.allclose(U[0], U[1])
    # values 2, 2: (II) holds, (III) fails
    U = np.array([[1, 2], [2, 1]])
    assert np.allclose(U, U.T) and not np.allclose(U * U.T, 1)
    # F_2: 2 = 0, both parity equations 0 = 0
    assert (1 + 1) % 2 == 0 and (1 - 1) % 2 == 0
    # Z/6Z: characteristic 6 (not 2), 2*3 = 0, 2 has no inverse
    assert all((6 * a) % 6 == 0 for a in range(6)) and (2 * 3) % 6 == 0
    assert all((2 * b) % 6 != 1 for b in range(6))
    # Paper I Prop. 0.1 on random unitary matrices (fair elements)
    for d in (2, 3, 4):
        Q, _ = np.linalg.qr(RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d)))
        up, um = (Q + Q.conj().T) / 2, (Q - Q.conj().T) / 2
        v, D = np.eye(d) - up, Q - np.eye(d)
        assert close(up @ um, um @ up) and close(up @ up - um @ um, np.eye(d))
        assert close(um @ um, -v @ (2 * np.eye(d) - v)) and close(D.conj().T @ D, 2 * v)


def group_positivity():
    """Reciprocal gains without positivity (Ex. n11:h2)."""
    # negative singleton
    assert (-1) * (-1) == 1 and (1 + (-1)) / 2 == 0
    # dual numbers a + b eps as [[a,b],[0,a]], involution eps -> -eps
    dual = lambda a, b: np.array([[a, b], [0, a]], dtype=float)
    u, us = dual(1, 1), dual(1, -1)
    assert np.allclose(u @ us, np.eye(2)) and np.allclose(dual(0, 1) @ dual(0, 1), 0)
    # split complex a + b j, j^2 = 1, as [[a,b],[b,a]], involution j -> -j
    spl = lambda a, b: np.array([[a, b], [b, a]], dtype=float)
    u, us = spl(5 / 4, 3 / 4), spl(5 / 4, -3 / 4)
    assert np.allclose(u @ us, np.eye(2))
    assert np.allclose(spl(0, 3 / 4) @ spl(0, 3 / 4), (9 / 16) * np.eye(2))
    # Cl_{4,0} with reversion = adjoint in a Hermitian representation
    g = clifford_gammas(4)
    E = np.eye(g[0].shape[0])
    for i in range(4):
        for j in range(4):
            assert close(g[i] @ g[j] + g[j] @ g[i], 2 * (i == j) * E)
    e12, e34 = g[0] @ g[1], g[2] @ g[3]
    B = e12 + e34
    assert close(B.conj().T, -B)
    assert close(B @ B, -2 * E + 2 * g[0] @ g[1] @ g[2] @ g[3])
    assert not close(B @ B, (np.trace(B @ B) / E.shape[0]) * E)        # not scalar
    f = (E + e12) @ (E + e34) / 2
    assert close(f @ f.conj().T, E) and close(f.conj().T @ f, E)        # fair
    assert close((f - f.conj().T) / 2, B / 2)                           # odd part B/2
    # three-point circle gains (Paper I Ex. 0.27)
    s3 = np.sqrt(3) / 2
    G = np.array([[1, s3, s3], [s3, 1, -1], [s3, -1, 1]])
    assert close(np.linalg.det(G), -3)
    assert close(sorted(np.linalg.eigvalsh(G)), sorted([2, (1 + np.sqrt(7)) / 2, (1 - np.sqrt(7)) / 2]))
    chord = lambda cs: np.sqrt(2 - 2 * cs)
    assert close(2 * chord(s3), np.sqrt(6) - np.sqrt(2)) and 2 * chord(s3) < chord(-1) == 2
    # the gains themselves: e^{i pi/6}, e^{i pi/6}, -1 have these real parts; loop product -1
    g01 = g02 = np.exp(1j * np.pi / 6)
    g12 = -1
    assert close(np.real([g01, g02, g12]), [s3, s3, -1])
    assert close(g01 * g12 * np.conj(g02), -1)
    # two-point unit-diagonal tables with |s|<=1 are PSD
    for s in np.linspace(-1, 1, 41):
        assert np.all(np.linalg.eigvalsh(np.array([[1, s], [s, 1]])) >= -1e-14)


def group_clifford():
    """Unit spheres of every R^n with Clifford gains satisfy A_0, (IV) and the round-trip identity."""
    for n in range(2, 8):
        g = clifford_gammas(n)
        E = np.eye(g[0].shape[0])
        for _ in range(30):
            x, y, z, w = (rand_unit(n) for _ in range(4))
            X, Y, Z, W = (vec(a, g) for a in (x, y, z, w))
            u = lambda A, B: A @ B
            assert close(u(Y, X), u(X, Y).conj().T)                    # (II): reversion
            assert close(u(X, Y) @ u(Y, X), E)                         # (III)
            assert close(u(X, Y) @ u(Y, Z), u(X, Z))                   # (IV)
            up = (u(X, Y) + u(Y, X)) / 2
            um = (u(X, Y) - u(Y, X)) / 2
            assert close(up, np.dot(x, y) * E)
            assert close(um @ um, -(1 - np.dot(x, y) ** 2) * E)
            assert close(X @ W @ X, vec(2 * np.dot(x, w) * x - w, g))  # s_x(w) = x w x
            sxw = 2 * np.dot(x, w) * x - w
            kap = lambda a, b: (1 + np.dot(a, b)) / 2
            assert close(kap(w, sxw), (2 * kap(w, x) - 1) ** 2)        # round-trip identity
            assert close(u(X, Y) @ u(Y, Z) @ u(Z, X), E)               # loop product trivial
    # the projected loop (Bargmann invariant) is not trivial: Pauli axes
    P = [proj_r(np.eye(3)[i]) for i in range(3)]
    assert close(np.trace(P[0] @ P[1] @ P[2]), (1 + 1j) / 4)
    assert close(np.trace(P[0] @ P[2] @ P[1]), (1 - 1j) / 4)
    assert all(close(np.trace(P[i] @ P[j]), 0.5) for i in range(3) for j in range(3) if i != j)


def group_rational_circle():
    """Rational points of S^1 satisfy all algebraic clauses of Omega; the set is not a manifold."""
    def pt(t):
        t = Fraction(t)
        return ((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t))
    mul = lambda p, q: (p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0])
    conj = lambda p: (p[0], -p[1])
    ts = [Fraction(a, b) for a in range(-7, 8) for b in (1, 2, 3, 5)]
    P = [pt(t) for t in ts]
    for p in P:
        assert p[0] ** 2 + p[1] ** 2 == 1
    for p, q in itertools.product(P[:20], P[:20]):
        r = mul(p, conj(q))
        assert r[0] ** 2 + r[1] ** 2 == 1                               # closed under x conj(y)
        kap = (1 + (p[0] * q[0] + p[1] * q[1])) / 2                     # rational comparison
        s = mul(mul(q, conj(p)), q)                                     # s_q(p) = q conj(p) q
        kap_s = (1 + (p[0] * s[0] + p[1] * s[1])) / 2
        assert kap_s == (2 * kap - 1) ** 2                              # (Omega3) exactly
    # the antipode -x is rational
    for t in ts:
        if t != 0:
            assert pt(-1 / t) == (-pt(t)[0], -pt(t)[1])
    # dyadic truncations t_n of sqrt 2 give rational points converging to (-1/3, 2 sqrt2/3)
    lim = np.array([-1 / 3, 2 * np.sqrt(2) / 3])
    for n in range(1, 40):
        tn = Fraction(int(np.floor(2 ** n * np.sqrt(2))), 2 ** n)
        p = pt(tn)
        assert p[0] ** 2 + p[1] ** 2 == 1
        dist = np.linalg.norm(np.array([float(p[0]), float(p[1])]) - lim)
        assert dist < 8 * 2.0 ** (-n)
    assert (2 * sp.sqrt(2) / 3).is_rational is False                    # limit is irrational
    assert sp.Rational(-1, 3) ** 2 + (2 * sp.sqrt(2) / 3) ** 2 == 1


def group_complex_structure():
    """No J^2=-I in odd real dimension; the kernel space of kappa on S^3 has dimension 5."""
    for n in range(1, 8):
        assert np.isclose(np.linalg.det(-np.eye(n)), (-1) ** n)
    J0 = np.array([[0, -1], [1, 0]])
    assert np.allclose(J0 @ J0, -np.eye(2))
    pts = np.array([rand_unit(4) for _ in range(40)])
    Kc = pts @ pts.T                                                    # c = <x,y>
    Kk = (1 + Kc) / 2                                                   # kappa
    sv_c = np.linalg.svd(Kc, compute_uv=False)
    sv_k = np.linalg.svd(Kk, compute_uv=False)
    assert np.sum(sv_c > 1e-9) == 4 and np.sum(sv_k > 1e-9) == 5
    # beta on S^2 (rays) has rank 4
    rs = np.array([rand_unit(3) for _ in range(40)])
    Kb = (1 + rs @ rs.T) / 2
    assert np.sum(np.linalg.svd(Kb, compute_uv=False) > 1e-9) == 4


def group_born_weights():
    """Cubic weight: [W],[B],[N] hold, [Comp] fails; trine total 5/4; Busch route."""
    t = sp.symbols('t')
    f = (1 + (2 * t - 1) ** 3) / 2
    assert sp.expand(f - (4 * t ** 3 - 6 * t ** 2 + 3 * t)) == 0
    assert sp.simplify(f + f.subs(t, 1 - t) - 1) == 0                  # [N] on a qubit
    assert f.subs(t, 0) == 0 and f.subs(t, 1) == 1
    tt = np.linspace(0, 1, 2001)
    fv = (1 + (2 * tt - 1) ** 3) / 2
    assert np.all((fv >= 0) & (fv <= 1)) and np.all(np.diff(fv) >= -1e-15)
    assert np.all(fv[1:] > 0)                                           # F^{-1}(0) = {0}
    s = sp.symbols('s', positive=True)
    F = f.subs(t, s ** 2)
    assert sp.simplify(F.subs(s, sp.Rational(1, 2)) * F.subs(s, sp.Rational(1, 3))
                       - F.subs(s, sp.Rational(1, 6))) != 0             # not multiplicative
    # value on projections against the state e_z: (1 + r_z^3)/2
    for _ in range(50):
        r = rand_unit(3)
        beta = (1 + r[2]) / 2
        assert close((1 + (2 * beta - 1) ** 3) / 2, (1 + r[2] ** 3) / 2)
    # trine
    r1 = np.array([0, 0, 1.0])
    r2 = np.array([np.sqrt(3) / 2, 0, -0.5])
    r3 = np.array([-np.sqrt(3) / 2, 0, -0.5])
    Es = [2 / 3 * proj_r(r) for r in (r1, r2, r3)]
    assert close(sum(Es), I2) and close(r1 + r2 + r3, 0)
    assert all(np.all(np.linalg.eigvalsh(Ek) >= -1e-14) for Ek in Es)
    fz = lambda rz: sp.Rational(1, 2) * (1 + rz ** 3)
    total = sp.Rational(2, 3) * (fz(1) + 2 * fz(sp.Rational(-1, 2)))
    assert total == sp.Rational(5, 4)
    assert fz(sp.Rational(-1, 2)) == sp.Rational(7, 16)
    # discontinuous antipodal indicator: first nonzero coordinate positive
    ind = lambda r: 1 if r[np.flatnonzero(np.abs(r) > 1e-12)[0]] > 0 else 0
    assert ind(r1) + ind(-r1) == 1 and ind(r2) + ind(-r2) == 1
    assert Fraction(2, 3) * (ind(r1) + ind(r2) + ind(r3)) == Fraction(4, 3)
    # the trace rule is additive and gives 1
    rho = proj_r(r1)
    assert close(sum(np.trace(rho @ Ek) for Ek in Es), 1)
    # two-outcome rank-one tests with sum I are complementary projections
    for _ in range(50):
        r = rand_unit(3)
        E1 = proj_r(r)
        E2 = I2 - E1
        assert np.linalg.matrix_rank(E2, tol=1e-9) == 1 and close(E2, proj_r(-r))
    # Busch: an additive normalised p on qubit effects is tr(rho E); with p(P_x)=1, rho = P_x
    rx = rand_unit(3)
    Px = proj_r(rx)
    for _ in range(20):
        r = rand_unit(3)
        assert close(np.trace(Px @ proj_r(r)).real, (1 + rx @ r) / 2)


def group_instruments():
    """Same effects, different instruments; Lueders selection; time premises."""
    k0 = np.array([1, 0], dtype=complex)
    k1 = np.array([0, 1], dtype=complex)
    plus = (k0 + k1) / np.sqrt(2)
    P0, P1 = np.outer(k0, k0), np.outer(k1, k1)
    Kl = [P0, P1]
    Kr = [np.outer(plus, k0), np.outer(plus, k1)]
    for K in (Kl, Kr):
        assert close(sum(A.conj().T @ A for A in K), I2)
    for a in range(2):
        assert close(Kl[a].conj().T @ Kl[a], Kr[a].conj().T @ Kr[a])    # equal effects
    for _ in range(30):
        rho = rand_density(2)
        p0 = np.trace(P0 @ rho).real
        assert p0 > 0
        for K, expect in ((Kl, 0.0), (Kr, 0.5)):
            post = K[0] @ rho @ K[0].conj().T / p0
            assert close(np.trace(P1 @ post).real, expect)
    # rank-two repeatable branch that moves rays inside its range
    P = np.diag([1, 1, 0]).astype(complex)
    U = np.diag([1, -1, 1]).astype(complex)
    K1, K2 = U @ P, np.eye(3) - P
    assert close(K1.conj().T @ K1 + K2.conj().T @ K2, np.eye(3))
    for _ in range(20):
        G = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))
        rho = G @ G.conj().T
        rho /= np.trace(rho)
        post = K1 @ rho @ K1.conj().T
        post /= np.trace(post)
        assert close(np.trace(P @ post), 1)                             # repeatable
    a = np.array([1, 1, 0]) / np.sqrt(2)
    b = np.array([1, -1, 0]) / np.sqrt(2)
    assert close(K1 @ a, b)
    # Lueders selection: any Kraus family with sum K*K = P has K = K P
    for _ in range(20):
        V = RNG.normal(size=(6, 3)) + 1j * RNG.normal(size=(6, 3))
        Q, _ = np.linalg.qr(V)                                          # 6x3 isometry
        Ks = [Q[2 * i:2 * i + 2, :] @ P for i in range(3)]              # rows of an isometry, times P
        S = sum(A.conj().T @ A for A in Ks)
        assert close(S, P)
        for A in Ks:
            assert close(A, A @ P)
            assert close(A @ np.array([0, 0, 1]), 0)
    # unilateral shift on a window
    n = 12
    Ssh = np.zeros((n + 1, n))
    for i in range(n):
        Ssh[i + 1, i] = 1
    assert np.allclose(Ssh.T @ Ssh, np.eye(n))
    SSt = Ssh @ Ssh.T
    e0 = np.zeros(n + 1)
    e0[0] = 1
    assert np.allclose(SSt[:n, :n], (np.eye(n + 1) - np.outer(e0, e0))[:n, :n])
    # finite-dimensional isometries are unitary
    Q, _ = np.linalg.qr(RNG.normal(size=(5, 5)))
    assert np.allclose(Q @ Q.T, np.eye(5))
    # Hamel homomorphism on Q + Q sqrt2: h(a + b sqrt2) = b pi; g(t) = diag(e^{-ih/2}, e^{ih/2})
    def g(a, b):
        h = float(b) * np.pi
        return np.diag([np.exp(-0.5j * h), np.exp(0.5j * h)])
    samples = [(Fraction(RNG.integers(-9, 10), RNG.integers(1, 6)),
                Fraction(RNG.integers(-9, 10), RNG.integers(1, 6))) for _ in range(40)]
    for (a1, b1), (a2, b2) in itertools.combinations(samples[:15], 2):
        assert close(g(a1 + a2, b1 + b2), g(a1, b1) @ g(a2, b2))         # homomorphism
        x0 = quat(rand_unit(4))
        xt, xs = g(a1, b1) @ x0, g(a2, b2) @ x0
        assert close(g(a1 + a2, b1 + b2) @ x0, xt @ qconj(x0) @ xs)      # x(t+s) = [x(t),x(0),x(s)]
    for a in samples:
        assert close(g(a[0], 0), np.eye(2))                              # identity at rational times
    gs2 = g(0, 1)
    assert close(gs2, -1j * SZ)
    assert close(abs(np.vdot(plus, gs2 @ plus)) ** 2, 0)                 # |+> to the orthogonal ray
    # Hamel values arbitrarily close to t=0: g(b sqrt2 - a) with |b sqrt2 - a| small, b odd
    t_small = 169 * np.sqrt(2) - 239                                     # b = 169, a = -239
    assert 0 < t_small < 0.0022 and close(g(-239, 169), -1j * SZ)
    assert close(abs(np.vdot(plus, g(-239, 169) @ plus)) ** 2, 0)
    # two Hamiltonians: H = 0 and H = hbar omega sigma_z at t = pi/(2 omega)
    U0 = np.eye(2)
    U1 = np.cos(np.pi / 2) * I2 - 1j * np.sin(np.pi / 2) * SZ
    assert close(U1, -1j * SZ)
    assert close(abs(np.vdot(U0 @ plus, U1 @ plus)) ** 2, 0)


def group_composites():
    """Real composite, dark summand, exchange sectors, C1(a) model."""
    J0 = np.array([[0, -1], [1, 0]], dtype=float)
    Y = np.kron(J0, J0)
    assert np.allclose(Y, Y.T) and np.allclose(Y @ Y, np.eye(4)) and np.isclose(np.trace(Y), 0)
    rp, rm = (np.eye(4) + Y) / 4, (np.eye(4) - Y) / 4
    for r in (rp, rm):
        assert np.all(np.linalg.eigvalsh(r) >= -1e-14) and np.isclose(np.trace(r), 1)
    assert np.allclose(rp @ rm, 0)                                       # orthogonal supports
    sym2 = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    prods = [np.kron(A, B) for A in sym2 for B in sym2]
    M = np.array([p.ravel() for p in prods])
    assert np.linalg.matrix_rank(M) == 9                                 # 9 of 10 symmetric dims
    for p in prods:
        assert np.isclose(np.trace(p @ Y), 0)
        assert np.isclose(np.trace(p @ rp), np.trace(p @ rm))
    for _ in range(20):
        A = RNG.normal(size=(2, 2))
        A = A + A.T
        assert np.isclose(np.trace(J0 @ A), 0)
    # complex product: Hermitian product basis spans 16 real dimensions
    herm = [I2, SX, SY, SZ]
    Mc = np.array([np.concatenate([np.kron(A, B).real.ravel(), np.kron(A, B).imag.ravel()])
                   for A in herm for B in herm])
    assert np.linalg.matrix_rank(Mc) == 16
    # dark summand: H = C^2 (x) C^2, K = C^2
    d1 = d2 = 2
    dK = 2
    def ext(E, F):
        top = np.kron(E, F)
        out = np.zeros((d1 * d2 + dK, d1 * d2 + dK), dtype=complex)
        out[:d1 * d2, :d1 * d2] = top
        out[d1 * d2:, d1 * d2:] = (np.trace(E) / d1) * (np.trace(F) / d2) * np.eye(dK)
        return out
    for _ in range(10):
        a, b = rand_unit(3), rand_unit(3)
        Es = [proj_r(a), proj_r(-a)]
        Fs = [proj_r(b), proj_r(-b)]
        assert close(sum(ext(E, F) for E in Es for F in Fs), np.eye(d1 * d2 + dK))
        k1 = np.zeros(d1 * d2 + dK); k1[4] = 1
        k2 = np.zeros(d1 * d2 + dK); k2[5] = 1
        mixed = np.zeros((6, 6)); mixed[:4, :4] = np.eye(4) / 4
        for E in Es:
            for F in Fs:
                X = ext(E, F)
                assert close(k1 @ X @ k1, k2 @ X @ k2)
                assert close(k1 @ X @ k1, np.trace(X @ mixed))
    # exchange sectors: v (x) v symmetric unit; antisymmetrisation of v (x) v vanishes
    v = rand_unit(2) + 1j * rand_unit(2)
    v /= np.linalg.norm(v)
    vv = np.kron(v, v)
    SW = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            SW[2 * j + i, 2 * i + j] = 1
    assert close(SW @ vv, vv) and close(np.linalg.norm(vv), 1)
    assert close((vv - SW @ vv) / 2, 0)
    # C1(a) model: ([a],[b]) -> [a (x) F b], F = diag(1,2)
    Fm = np.diag([1.0, 2.0])
    b, bp = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
    assert np.isclose(b @ bp, 0)
    nb, nbp = Fm @ b / np.linalg.norm(Fm @ b), Fm @ bp / np.linalg.norm(Fm @ bp)
    assert np.isclose(abs(nb @ nbp) ** 2, 9 / 25)


def group_events_connections_entropy():
    """Null cones, two connections, reset, pinching."""
    # Euclidean form t^2 + x^2 has only the zero null vector; Lorentz form has (1,+-1)
    for _ in range(100):
        v = RNG.normal(size=2)
        assert v @ v > 0
    assert (1 ** 2 - 1 ** 2) == 0 and (1 ** 2 - (-1) ** 2) == 0
    # boost of (0,2)
    for beta in (0.3, -0.6, 0.9):
        gam = 1 / np.sqrt(1 - beta ** 2)
        tp, xp = gam * (0 - beta * 2), gam * (2 - beta * 0)
        assert np.isclose(tp, -2 * gam * beta) and np.isclose(xp, 2 * gam)
        assert np.isclose(tp ** 2 - xp ** 2, -4)
        assert np.sign(tp) == -np.sign(beta)
    # two connections on a triangle with identical fibres: loop products 1 and -1
    triv = [1, 1, 1]
    twisted = [-1, 1, 1]
    assert np.prod(triv) == 1 and np.prod(twisted) == -1
    # reversal by the inverse
    assert (-1) * (1 / -1) == 1
    # reset channel
    K = [np.outer([1, 0], [1, 0]), np.outer([1, 0], [0, 1])]
    assert close(sum(A.conj().T @ A for A in K), I2)
    out = sum(A @ (I2 / 2) @ A.conj().T for A in K)
    assert close(out, np.diag([1, 0]))
    assert np.isclose(vn_entropy(out) - vn_entropy(I2 / 2), -np.log(2))
    # pinching: S(Pi rho) - S(rho) = D(rho || Pi rho); qubit Z-pinching formula
    for _ in range(200):
        r = rand_unit(3) * RNG.uniform(0, 1)
        rho = (I2 + r[0] * SX + r[1] * SY + r[2] * SZ) / 2
        pin = np.diag(np.diag(rho))
        dS = vn_entropy(pin) - vn_entropy(rho)
        nr = np.linalg.norm(r)
        assert np.isclose(dS, hbin((1 + r[2]) / 2) - hbin((1 + nr) / 2))
        assert dS >= -1e-12
        ev, U = np.linalg.eigh(rho)
        logrho = U @ np.diag(np.log(np.clip(ev, 1e-300, None))) @ U.conj().T
        D = np.trace(rho @ (logrho - np.diag(np.log(np.diag(pin).real)))).real
        assert np.isclose(dS, D, atol=1e-9)
    rho = (I2 + 0.7 * SZ) / 2
    assert np.isclose(vn_entropy(np.diag(np.diag(rho))) - vn_entropy(rho), 0)
    # monotonicity of h((1+t)/2) on [0,1]
    tt = np.linspace(0, 1, 1001)
    hv = np.array([hbin((1 + x) / 2) for x in tt])
    assert np.all(np.diff(hv) < 0)


def group_sorkin():
    """Third-order interference vanishes for q = |a|^2; quartic and classical comparisons."""
    for _ in range(200):
        v = RNG.normal(size=7) + 1j * RNG.normal(size=7)
        sets = [set(), {0}, {1, 2}, {3, 4, 5}]
        for A, B, C in itertools.permutations(sets, 3):
            if A & B or A & C or B & C:
                continue
            q = lambda E: abs(sum(v[e] for e in E)) ** 2
            I3 = q(A | B | C) - q(A | B) - q(A | C) - q(B | C) + q(A) + q(B) + q(C)
            assert abs(I3) < 1e-9
        # sesquilinear version z^* H z with H Hermitian (Prop. bdpurity)
        H = RNG.normal(size=(7, 7)) + 1j * RNG.normal(size=(7, 7))
        H = H + H.conj().T
        def qH(E):
            z = np.zeros(7, dtype=complex)
            for e in E:
                z[e] = v[e]
            return (z.conj() @ H @ z).real
        A, B, C = {0, 1}, {2}, {3, 4}
        I3 = qH(A | B | C) - qH(A | B) - qH(A | C) - qH(B | C) + qH(A) + qH(B) + qH(C)
        assert abs(I3) < 1e-9
    # symbolic over a commutative ring
    a, b, c, ab, bb, cb = sp.symbols('a b c abar bbar cbar')
    q = lambda x, xb: x * xb
    I3 = (q(a + b + c, ab + bb + cb) - q(a + b, ab + bb) - q(a + c, ab + cb) - q(b + c, bb + cb)
          + q(a, ab) + q(b, bb) + q(c, cb))
    assert sp.expand(I3) == 0
    # unit amplitudes: quadratic 9-12+3=0, quartic 81-48+3=36
    assert 3 ** 2 - 3 * 2 ** 2 + 3 == 0
    assert 3 ** 4 - 3 * 2 ** 4 + 3 == 36
    # classical additive probability: I3 = 0, I2 = 0; coherent equal amplitudes: I2 = 2
    p = [1 / 3, 1 / 3, 1 / 3]
    P = lambda E: sum(p[e] for e in E)
    assert np.isclose(P({0, 1, 2}) - P({0, 1}) - P({0, 2}) - P({1, 2}) + 1, 0)
    assert np.isclose(P({0, 1}) - P({0}) - P({1}), 0)
    amp = np.ones(3) / np.sqrt(3)
    Q = lambda E: abs(sum(amp[e] for e in E)) ** 2
    assert np.isclose(Q({0, 1}) - Q({0}) - Q({1}), 2 / 3)


def mubs(d):
    """complete set of d+1 MUBs for d = 2 or an odd prime, as lists of column-vector matrices"""
    bases = [np.eye(d, dtype=complex)]
    if d == 2:
        bases.append(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        bases.append(np.array([[1, 1], [1j, -1j]]) / np.sqrt(2))
        return bases
    w = np.exp(2j * np.pi / d)
    for a in range(d):
        B = np.array([[w ** ((a * j * j + bb * j) % d) for bb in range(d)] for j in range(d)]) / np.sqrt(d)
        bases.append(B)
    return bases


def group_entropy():
    """Collision-sum identity, entropy bounds, equality cases, Pauli contexts."""
    for d in (2, 3, 5):
        Bs = mubs(d)
        for B1, B2 in itertools.combinations(Bs, 2):
            assert close(np.abs(B1.conj().T @ B2) ** 2, np.full((d, d), 1 / d))
        for m in range(1, d + 2):
            fam = Bs[:m]
            # orthonormal basis of the span L of traceless diagonals
            gens = []
            for B in fam:
                for j in range(d - 1):
                    gens.append(np.outer(B[:, j], B[:, j].conj()) - np.eye(d) / d)
            Gm = np.array([np.concatenate([g.real.ravel(), g.imag.ravel()]) for g in gens])
            Qo, _ = np.linalg.qr(Gm.T)
            assert np.linalg.matrix_rank(Gm) == m * (d - 1)
            for _ in range(15):
                rho = rand_density(d, rank=RNG.integers(1, d + 1))
                D = rho - np.eye(d) / d
                Dv = np.concatenate([D.real.ravel(), D.imag.ravel()])
                res = Dv - Qo @ (Qo.T @ Dv)
                R = res @ res
                ps = [np.clip(np.real(np.einsum('ij,jk,ki->i', B.conj().T, rho, B)), 0, None) for B in fam]
                cs = [float((pb ** 2).sum()) for pb in ps]
                pur = np.trace(rho @ rho).real
                assert np.isclose(sum(cs), pur + (m - 1) / d - R, atol=1e-10)
                if m == d + 1:
                    assert abs(R) < 1e-10
                Hs = sum(shannon(pb) for pb in ps)
                H2 = sum(-np.log(cb) for cb in cs)
                bound = m * np.log(m / (pur + (m - 1) / d - R))
                assert Hs >= H2 - 1e-10 and H2 >= bound - 1e-10
                assert bound >= m * np.log(m / (pur + (m - 1) / d)) - 1e-12
    # equality: maximally mixed state attains all three
    for d in (2, 3):
        rho = np.eye(d) / d
        Bs = mubs(d)
        ps = [np.real(np.diag(B.conj().T @ rho @ B)) for B in Bs]
        Hs = sum(shannon(p) for p in ps)
        assert np.isclose(Hs, (d + 1) * np.log((d + 1) / (1 / d + 1)))
    # Pauli contexts
    g = lambda x: hbin((1 + np.sqrt(x)) / 2)
    xs = np.linspace(1e-6, 1 - 1e-6, 4001)
    gv = np.array([g(x) for x in xs])
    assert np.all(np.diff(gv, 2) < 1e-12)                                # concave
    xsym = sp.symbols('x', positive=True)
    pp = (1 + sp.sqrt(xsym)) / 2
    gsym = -pp * sp.log(pp) - (1 - pp) * sp.log(1 - pp)
    dgsym = sp.diff(gsym, xsym)
    for xv in (sp.Rational(1, 10), sp.Rational(1, 2), sp.Rational(9, 10)):
        val = dgsym.subs(xsym, xv) + sp.atanh(sp.sqrt(xv)) / (2 * sp.sqrt(xv))
        assert abs(float(sp.N(val, 30))) < 1e-25
    tt = np.linspace(1e-4, 1 - 1e-4, 2000)
    assert np.all(np.diff(np.arctanh(tt) / tt) > 0)
    for _ in range(3000):
        r = rand_unit(3) * RNG.uniform(0, 1) ** 0.5
        nr = np.linalg.norm(r)
        Hsum = sum(hbin((1 + rb) / 2) for rb in r)
        assert Hsum >= 2 * np.log(2) + hbin((1 + nr) / 2) - 1e-12
        assert Hsum >= 3 * np.log(6 / (3 + nr ** 2)) - 1e-12
        assert 2 * np.log(2) + hbin((1 + nr) / 2) >= 3 * np.log(6 / (3 + nr ** 2)) - 1e-12
    for nr in (0.3, 0.8, 1.0):
        r = np.array([0, 0, nr])
        assert np.isclose(sum(hbin((1 + rb) / 2) for rb in r), 2 * np.log(2) + hbin((1 + nr) / 2))
        r = np.array([nr, nr, 0]) / np.sqrt(2)
        assert sum(hbin((1 + rb) / 2) for rb in r) > 2 * np.log(2) + hbin((1 + nr) / 2) + 1e-6
    assert np.isclose(sum(hbin(0.5) for _ in range(3)), 3 * np.log(6 / 3))   # r = 0 equality
    assert 4 > 1.5 ** 3 and 2 * np.log(2) > 3 * np.log(1.5)
    # P = (1 + |r|^2)/2 for a qubit
    r = rand_unit(3) * 0.6
    rho = (I2 + r[0] * SX + r[1] * SY + r[2] * SZ) / 2
    assert np.isclose(np.trace(rho @ rho).real, (1 + r @ r) / 2)


def group_interpretations():
    """Retained record versus mixture; heralding on a Bell pair."""
    p = 3 / 4
    assert np.isclose(p ** 2 + (1 - p) ** 2, 5 / 8) and np.isclose(2 * p * (1 - p), 1 - 5 / 8)
    # explicit: Psi = sqrt p |0>|0> + sqrt(1-p)|1>|1>, joint test of Psi
    k = np.eye(2)
    Psi = np.sqrt(p) * np.kron(k[0], k[0]) + np.sqrt(1 - p) * np.kron(k[1], k[1])
    mix = p * np.outer(np.kron(k[0], k[0]), np.kron(k[0], k[0])) + (1 - p) * np.outer(np.kron(k[1], k[1]), np.kron(k[1], k[1]))
    assert np.isclose(Psi @ np.outer(Psi, Psi) @ Psi, 1)
    assert np.isclose(Psi @ mix @ Psi, 5 / 8)
    # same system marginal
    red = lambda R: np.einsum('ijkj->ik', R.reshape(2, 2, 2, 2))
    assert np.allclose(red(np.outer(Psi, Psi)), red(mix))
    # Bell pair heralding
    bell = (np.kron(k[0], k[0]) + np.kron(k[1], k[1])) / np.sqrt(2)
    E = np.diag([3 / 4, 1 / 4])
    Kh = np.diag([np.sqrt(3 / 4), np.sqrt(1 / 4)])
    prob = bell @ np.kron(E, np.eye(2)) @ bell
    assert np.isclose(prob, 1 / 2)
    post = np.kron(Kh, np.eye(2)) @ np.outer(bell, bell) @ np.kron(Kh, np.eye(2)).T / prob
    other = np.einsum('ijik->jk', post.reshape(2, 2, 2, 2))
    assert np.allclose(other, np.diag([3 / 4, 1 / 4]))
    Kc = np.diag([np.sqrt(1 / 4), np.sqrt(3 / 4)])
    unread = (np.kron(Kh, np.eye(2)) @ np.outer(bell, bell) @ np.kron(Kh, np.eye(2)).T
              + np.kron(Kc, np.eye(2)) @ np.outer(bell, bell) @ np.kron(Kc, np.eye(2)).T)
    assert np.allclose(np.einsum('ijik->jk', unread.reshape(2, 2, 2, 2)), np.eye(2) / 2)
    # semibounded spectrum is not translation invariant: min spec(H + s) = min spec(H) + s
    H = np.diag([0.0, 1.0, 2.5])
    for s in (0.5, -1.0):
        assert not np.isclose(min(np.linalg.eigvalsh(H + s * np.eye(3))), min(np.linalg.eigvalsh(H)))


def group_set_models():
    """Cardinalities of the finite cumulative hierarchy (Paper I Rem. 10C.21)."""
    sizes = [0]
    for _ in range(5):
        sizes.append(2 ** sizes[-1])
    assert sizes == [0, 1, 2, 4, 16, 65536]


GROUPS = [group_a0_realisation, group_converses, group_clauses, group_positivity, group_clifford,
          group_rational_circle, group_complex_structure, group_born_weights, group_instruments,
          group_composites, group_events_connections_entropy, group_sorkin, group_entropy,
          group_interpretations, group_set_models]


def main():
    ok = True
    for grp in GROUPS:
        try:
            grp()
            print(f"{grp.__name__}: passed")
        except AssertionError as e:
            ok = False
            import traceback
            traceback.print_exc()
            print(f"{grp.__name__}: FAILED {e}")
    print(f"check_independence.py: all {len(GROUPS)} groups passed: {ok}")


if __name__ == "__main__":
    main()
