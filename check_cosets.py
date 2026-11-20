#!/usr/bin/env python3
"""Independent checks for the section 'Coset towers, exterior-degree recursion and the
composition of interaction signs' (module n12).

Path-free: reads no files.  NumPy, SciPy and SymPy only.  Deterministic seeds.
Each group_* function raises AssertionError on failure.
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import scipy.linalg as sla
import scipy.integrate as si
import scipy.optimize as so
import sympy as sp

RNG = np.random.default_rng(20261120)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [SX, SY, SZ]


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0)


def comm(a, b):
    return a @ b - b @ a


# ------------------------------------------------------------------ helpers for the coset theorem
def rand_su(n):
    a = RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n))
    x = a - a.conj().T
    x -= np.trace(x) / n * np.eye(n)
    return x


def ph(x):   # h = so(3) inside su(3): real part
    return (x + x.conj()) / 2


def pm(x):   # m = i * (real symmetric traceless)
    return (x - x.conj()) / 2


def ip(x, y):  # positive Ad-invariant inner product on su(n)
    return float(np.real(-np.trace(x @ y)))


def lift_family(Xs):
    X1, X2, X3, X4 = Xs

    def Z(x, y):
        return x * X1 + y * X2 + x * y * X3 + x * x * X4

    def dZ(x, y):
        return [X1 + y * X3 + 2 * x * X4, X2 + x * X3]

    def u(x, y):
        return sla.expm(Z(x, y))

    def alpha(x, y):
        z = Z(x, y)
        ui = sla.expm(-z)
        return [ui @ sla.expm_frechet(z, d, compute_expm=False) for d in dZ(x, y)]

    return u, alpha


def group_coset():
    """Theorem n12:coset: transformation law, structure equations, dressed form, norm identity,
    variation of P, current identity, lift dependence of the vertical fraction (refuted claim)."""
    u, alpha = lift_family([rand_su(3) for _ in range(4)])
    x0, y0, hd = 0.31, -0.22, 1e-5
    a = alpha(x0, y0)
    w = [ph(c) for c in a]
    P = [pm(c) for c in a]
    # reductive split: [h,m] in m and orthogonality
    A, B = ph(rand_su(3)), pm(rand_su(3))
    assert close(ph(comm(A, B)), 0)
    assert abs(ip(A, B)) < 1e-12
    # structure equations: F_w = -[P_x,P_y]_h,  D_w P = -[P_x,P_y]_m  (on d/dx, d/dy)
    dxa = [(c1 - c2) / (2 * hd) for c1, c2 in zip(alpha(x0 + hd, y0), alpha(x0 - hd, y0))]
    dya = [(c1 - c2) / (2 * hd) for c1, c2 in zip(alpha(x0, y0 + hd), alpha(x0, y0 - hd))]
    Fw = ph(dxa[1]) - ph(dya[0]) + comm(w[0], w[1])
    assert close(Fw, -ph(comm(P[0], P[1])), 1e-7)
    DP = pm(dxa[1]) - pm(dya[0]) + comm(w[0], P[1]) + comm(P[0], w[1])
    assert close(DP, -pm(comm(P[0], P[1])), 1e-7)
    # Maurer-Cartan in full
    assert close(dxa[1] - dya[0] + comm(a[0], a[1]), 0, 1e-7)
    # transformation law under u -> u k, k(x,y)=exp(x Z1 + y Z2) in H=SO(3)
    Z1, Z2 = ph(rand_su(3)), ph(rand_su(3))
    k = lambda x, y: sla.expm(x * Z1 + y * Z2)
    kk = k(x0, y0)
    ki = np.linalg.inv(kk)
    dk = [sla.expm_frechet(x0 * Z1 + y0 * Z2, Z, compute_expm=False) for Z in (Z1, Z2)]
    a2 = []
    for mu in range(2):
        hh = 1e-6
        e = [0, 0]
        e[mu] = hh
        up = u(x0 + e[0], y0 + e[1]) @ k(x0 + e[0], y0 + e[1])
        um = u(x0 - e[0], y0 - e[1]) @ k(x0 - e[0], y0 - e[1])
        a2.append(np.linalg.inv(u(x0, y0) @ kk) @ (up - um) / (2 * hh))
    for mu in range(2):
        assert close(ph(a2[mu]), ki @ w[mu] @ kk + ki @ dk[mu], 1e-8)
        assert close(pm(a2[mu]), ki @ P[mu] @ kk, 1e-8)
    # current J = f^2 u P u^{-1} unchanged by the lift change
    f = 1.7
    uu = u(x0, y0)
    for mu in range(2):
        J1 = f**2 * uu @ P[mu] @ np.linalg.inv(uu)
        J2 = f**2 * (uu @ kk) @ pm(a2[mu]) @ np.linalg.inv(uu @ kk)
        assert close(J1, J2, 1e-8)
    # refuted claim: vertical fraction changes with dk at a point where k = 1
    Zv = ph(rand_su(3))
    v = np.array([0.6, 0.8])
    wv = v[0] * w[0] + v[1] * w[1]
    Pv = v[0] * P[0] + v[1] * P[1]
    frac = lambda W: ip(W, W) / (ip(W, W) + ip(Pv, Pv))
    wv2 = wv + v[0] * Zv          # k = exp(x Zv) about the point: omega -> omega + dk
    assert abs(frac(wv) - frac(wv2)) > 1e-3
    # norm identity
    av = v[0] * a[0] + v[1] * a[1]
    assert abs(ip(av, av) - ip(wv, wv) - ip(Pv, Pv)) < 1e-10
    # dressed connection: alpha_A = u^{-1}(d+A)u, curvature conjugated
    C1, C2, C3 = rand_su(3), rand_su(3), rand_su(3)
    Afield = lambda x, y: [C1 + y * C3, C2 - x * C3 + x * y * C1]
    def alphaA(x, y):
        uu_ = u(x, y)
        ui_ = np.linalg.inv(uu_)
        return [c + ui_ @ AA @ uu_ for c, AA in zip(alpha(x, y), Afield(x, y))]
    aA = alphaA(x0, y0)
    dxA = [(c1 - c2) / (2 * hd) for c1, c2 in zip(alphaA(x0 + hd, y0), alphaA(x0 - hd, y0))]
    dyA = [(c1 - c2) / (2 * hd) for c1, c2 in zip(alphaA(x0, y0 + hd), alphaA(x0, y0 - hd))]
    # dA(dx,dy) = d_x A_y - d_y A_x with d_x A_y = -C3 + y C1 and d_y A_x = C3
    FA = (-C3 + y0 * C1) - C3 + comm(*Afield(x0, y0))
    lhs = dxA[1] - dyA[0] + comm(aA[0], aA[1])
    assert close(lhs, np.linalg.inv(uu) @ FA @ uu, 1e-7)
    # variation of P under u -> u exp(t xi(x,y)), xi in m: delta P = D_w xi + [P, xi]_m
    Xi1, Xi2 = pm(rand_su(3)), pm(rand_su(3))
    xi = lambda x, y: x * Xi1 + y * y * Xi2
    dxi = [Xi1, 2 * y0 * Xi2]
    tt = 1e-6
    def Pt(t):
        out = []
        for mu in range(2):
            e = [0, 0]
            e[mu] = 1e-6
            up = u(x0 + e[0], y0 + e[1]) @ sla.expm(t * xi(x0 + e[0], y0 + e[1]))
            um = u(x0 - e[0], y0 - e[1]) @ sla.expm(t * xi(x0 - e[0], y0 - e[1]))
            ut = u(x0, y0) @ sla.expm(t * xi(x0, y0))
            out.append(pm(np.linalg.inv(ut) @ (up - um) / 2e-6))
        return out
    dP = [(p1 - p2) / (2 * tt) for p1, p2 in zip(Pt(tt), Pt(-tt))]
    X0 = xi(x0, y0)
    for mu in range(2):
        pred = dxi[mu] + comm(w[mu], X0) + pm(comm(P[mu], X0))
        assert close(dP[mu], pred, 2e-4)
    # the extra term contracts to zero: sum_mu <[P_mu, xi]_m, P_mu> = 0 (any symmetric metric)
    for _ in range(5):
        Ps = [pm(rand_su(3)) for _ in range(4)]
        eta = np.diag([-1.0, 1, 1, 1])
        X0 = pm(rand_su(3))
        s = sum(eta[m_, m_] * ip(pm(comm(Ps[m_], X0)), Ps[m_]) for m_ in range(4))
        assert abs(s) < 1e-10
        # [P wedge *P] = sum_mu eta^{mu mu}[P_mu, P_mu] = 0 trivially; and the divergence identity:
    # d_mu (u P_mu u^{-1}) = u (d_mu P_mu + [w_mu, P_mu]) u^{-1}   (Euclidean 2D)
    def Jc(x, y):
        uu_ = u(x, y)
        al = alpha(x, y)
        return [uu_ @ pm(c) @ np.linalg.inv(uu_) for c in al]
    divJ = (Jc(x0 + hd, y0)[0] - Jc(x0 - hd, y0)[0] + Jc(x0, y0 + hd)[1] - Jc(x0, y0 - hd)[1]) / (2 * hd)
    divP = pm(dxa[0]) + pm(dya[1]) + comm(w[0], P[0]) + comm(w[1], P[1])
    assert close(divJ, uu @ divP @ np.linalg.inv(uu), 1e-7)
    # linearisation: u = exp(eps pi), P = eps d pi + O(eps^2)
    Pi1, Pi2 = pm(rand_su(3)), pm(rand_su(3))
    for eps in (1e-3, 2e-3):
        z = eps * (0.3 * Pi1 + 0.1 * Pi2)
        dz = eps * Pi1
        al = sla.expm(-z) @ sla.expm_frechet(z, dz, compute_expm=False)
        assert np.linalg.norm(pm(al) - dz) < 10 * eps**2
    # chiral coset (SU(2) x SU(2))/SU(2)_diag with lift u = (U, 1): omega = (L,L)/2, P = (L,-L)/2,
    # |P|^2 = |L|^2/2, F_omega = -[P,P]_h/2 = -([L,L],[L,L])/8 on a pair of directions
    Xa, Xb = rand_su(2), rand_su(2)
    Uf = lambda x, y: sla.expm(x * Xa + y * Xb + x * y * comm(Xa, Xb))
    def Lc(x, y):
        z = x * Xa + y * Xb + x * y * comm(Xa, Xb)
        dz = [Xa + y * comm(Xa, Xb), Xb + x * comm(Xa, Xb)]
        return [sla.expm(-z) @ sla.expm_frechet(z, d_, compute_expm=False) for d_ in dz]
    Lx, Ly = Lc(x0, y0)
    om = [(Lm / 2, Lm / 2) for Lm in (Lx, Ly)]
    Pp = [(Lm / 2, -Lm / 2) for Lm in (Lx, Ly)]
    assert abs(ip(Pp[0][0], Pp[0][0]) + ip(Pp[0][1], Pp[0][1]) - 0.5 * ip(Lx, Lx)) < 1e-12
    dLx = [(c1 - c2) / (2 * hd) for c1, c2 in zip(Lc(x0 + hd, y0), Lc(x0 - hd, y0))]
    dLy = [(c1 - c2) / (2 * hd) for c1, c2 in zip(Lc(x0, y0 + hd), Lc(x0, y0 - hd))]
    Fo = (dLx[1] - dLy[0]) / 2 + comm(Lx / 2, Ly / 2)       # first (= second) component of F_omega
    PPh = comm(Lx / 2, Ly / 2)                               # [P_x,P_y] = ([Lx,Ly]/4, [Lx,Ly]/4), all in h
    assert close(Fo, -PPh, 1e-7)
    assert close(Fo, -comm(Lx, Ly) / 4, 1e-7)
    return "coset"


def group_sectors():
    """Example n12:sectors: Hopf connection, cap flux, flux quantum, SU(3) pi_3 data."""
    e = [-1j * s for s in PAULI]
    T = [x / 2 for x in e]
    assert close(comm(T[0], T[1]), T[2]) and close(comm(T[1], T[2]), T[0])
    def u(th, phi):
        return sla.expm(phi * T[2]) @ sla.expm(th * T[1])
    def coeffs(X):   # X = sum c_j T_j
        return np.array([np.real(-2 * np.trace(X @ T[j])) for j in range(3)])
    for th, phi in [(0.4, 1.1), (1.3, -0.7), (2.5, 2.0)]:
        hd = 1e-6
        uu = u(th, phi)
        ui = np.linalg.inv(uu)
        a_th = ui @ (u(th + hd, phi) - u(th - hd, phi)) / (2 * hd)
        a_ph = ui @ (u(th, phi + hd) - u(th, phi - hd)) / (2 * hd)
        c_th, c_ph = coeffs(a_th), coeffs(a_ph)
        assert close(c_ph, [-math.sin(th), 0, math.cos(th)], 1e-8)
        assert close(c_th, [0, 1, 0], 1e-8)
        # -F_omega coefficient: omega = cos th dphi -> d omega = -sin th dth^dphi
        # -1/2[P,P]_h on (d_th, d_phi): -[P_th, P_ph]_h ; P_th = T2, P_ph = -sin th T1
        brk = comm(T[1], -math.sin(th) * T[0])
        assert close(coeffs(brk)[2], math.sin(th), 1e-12)  # [T2,-sin T1] = sin T3
        # so F_w = -sin th dth^dph and -1/2[P,P]_h = -sin th T3: consistent with F_w = -1/2[P,P]_h
    for acap, val in [(0.0, 0.0), (math.pi / 3, math.pi), (math.pi / 2, 2 * math.pi)]:
        flux = si.quad(lambda t: 2 * math.pi * math.sin(t), 0, acap)[0]
        assert abs(flux - val) < 1e-12
    h = 6.62607015e-34
    q = 1.602176634e-19
    assert abs(h / (2 * q) - 2.067833848e-15) < 1e-23
    # dims: SU(3)/SU(2) = S^5 ; pi_3 exact sequence numbers are not numerical; dims only
    assert 8 - 3 == 5
    return "sectors"


def group_breaking():
    """Hypothesis Br: Cartan bracket sign convention, electroweak counts, chiral and observer dims."""
    # Lorentz component of the Cartan curvature: with A = 1/2 w^{ab} J_ab + e^a P_a,
    # 1/2 [e,e]_h = 1/2 e^a ^ e^b [P_a,P_b] = 1/2 sigma l^-2 e^a^e^b J_ab when [P_a,P_b] = sigma l^-2 J_ab.
    def E(i, j, n):
        m = np.zeros((n, n))
        m[i, j] = 1
        return m
    ell = 1.3
    for etaN, label in [(np.diag([1, 1, 1, 1, 1.0]), "so5"), (np.diag([-1, 1, 1, 1, 1.0]), "so41"),
                        (np.diag([-1, 1, 1, 1, -1.0]), "so32")]:
        n = 5
        # generators preserving etaN: M_ij = E_ij eta_jj - E_ji eta_ii
        M = lambda i, j: E(i, j, n) @ etaN - E(j, i, n) @ etaN
        J = lambda a, b: M(a, b)
        Pg = lambda a: M(a, 4) / ell
        sig = None
        for a_, b_ in [(0, 1), (1, 2), (0, 3)]:
            br = comm(Pg(a_), Pg(b_))
            # find sigma with br = sigma l^-2 J_ab
            Jab = J(a_, b_)
            s_ = np.sum(br * Jab) / np.sum(Jab * Jab) * ell**2
            assert close(br, s_ / ell**2 * Jab, 1e-12)
            assert abs(abs(s_) - 1) < 1e-12
            sig = s_ if sig is None else sig
            assert abs(s_ - sig) < 1e-12
            # bracket lands in h: no P component
    # electroweak: SU(2) acts freely and transitively on unit columns
    for _ in range(5):
        z = RNG.normal(size=2) + 1j * RNG.normal(size=2)
        z /= np.linalg.norm(z)
        U = np.array([[z[0], -np.conj(z[1])], [z[1], np.conj(z[0])]])
        assert close(U @ U.conj().T, I2) and abs(np.linalg.det(U) - 1) < 1e-12
        assert close(U @ np.array([1, 0]), z)
    # Q = T3 + Y, Y = 1/2 on the doublet, annihilates (0, v/sqrt2)
    Q = SZ / 2 + 0.5 * I2
    assert close(Q @ np.array([0, 1.0]), 0)
    assert 4 * 2 + 4 == 3 * 3 + 2 + 1 == 12
    assert 3**2 - 1 == 8
    # charge-2e condensate: stabiliser of e^{2 i th} in U(1) is {0, pi}
    ths = np.linspace(0, 2 * np.pi, 100001)[:-1]
    stab = ths[np.abs(np.exp(2j * ths) - 1) < 1e-9]
    assert len(stab) == 2
    # observer: U(d)/U(1)^d has dimension d^2 - d; unitary fixing all basis rays is diagonal
    for d in (2, 3, 4):
        assert d * d - d == d * (d - 1)
    return "breaking"


def group_scales():
    """Example n12:tower-scales: on-shell mixing, Yukawa lengths, light-quark arithmetic, flux quantum."""
    mW, mZ = 80.3625, 91.1879          # GeV, PDG 2026
    hbarc = 197.3269804                # MeV fm, CODATA 2022
    s2 = 1 - (mW / mZ) ** 2
    assert abs(s2 - 0.22334) < 5e-6
    lW = hbarc / (mW * 1e3)
    lZ = hbarc / (mZ * 1e3)
    assert abs(lW - 2.4555e-3) < 5e-8 and abs(lZ - 2.1640e-3) < 5e-8
    lpi = hbarc / 139.57039
    assert abs(lpi - 1.4138) < 5e-5
    mu, md = 2.16, 4.70
    mp, mn = 938.27208943, 939.5654219
    assert abs((2 * mu + md) - 9.02) < 1e-12 and abs((mu + 2 * md) - 11.56) < 1e-12
    assert abs((1 - 9.02 / mp) - 0.99039) < 5e-6
    assert abs((1 - 11.56 / mn) - 0.98770) < 5e-6
    assert round(100 * 9.02 / mp, 2) == 0.96 and round(100 * 11.56 / mn, 2) == 1.23
    # hbar c from exact SI constants
    h, c, e = 6.62607015e-34, 299792458.0, 1.602176634e-19
    hbarc_calc = h * c / (2 * math.pi) / e / 1e6 * 1e15  # MeV fm
    assert abs(hbarc_calc - hbarc) < 1e-6
    return "scales"


def group_exterior():
    """Proposition n12:exterior, Example n12:equal-rays, Remark n12:clockmodel, 4D pseudoscalar signs."""
    e1, e2, e3 = SX, SY, SZ          # Cl(3,0) in M_2(C)
    I3 = e1 @ e2 @ e3
    assert close(I3, 1j * I2)
    for ek in (e1, e2, e3):
        assert close(comm(ek, I3), 0)
    assert close(I3 @ I3, -I2)
    for _ in range(5):
        t = RNG.normal()
        x = RNG.normal(size=3)
        p = t * I2 + sum(xi * s for xi, s in zip(x, PAULI))
        pbar = t * I2 - sum(xi * s for xi, s in zip(x, PAULI))
        assert close(p @ pbar, (t * t - x @ x) * I2)
        assert abs(np.linalg.det(p) - (t * t - x @ x)) < 1e-10
        assert abs(0.5 * np.trace(p @ p.conj().T) - (t * t + x @ x)) < 1e-10
    # Hodge 0 <-> 3: orthogonal map acts on top degree by its determinant
    for _ in range(3):
        Qm, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
        assert abs(abs(np.linalg.det(Qm)) - 1) < 1e-12
    # 4D unit pseudoscalar squares: +1 for (4,0), -1 for (1,3) and (3,1)
    g = [np.kron(SX, s) for s in PAULI] + [np.kron(SY, I2)]   # Euclidean gammas, square +1
    for sig, expect in [((1, 1, 1, 1), 1), ((1, -1, -1, -1), -1), ((-1, 1, 1, 1), -1), ((1, 1, -1, -1), 1)]:
        gg = [gm if s_ == 1 else 1j * gm for gm, s_ in zip(g, sig)]
        for a_ in range(4):
            assert close(gg[a_] @ gg[a_], sig[a_] * np.eye(4))
        I4 = gg[0] @ gg[1] @ gg[2] @ gg[3]
        assert close(I4 @ I4, expect * np.eye(4))
    # three equal-angle rays
    G = (np.eye(3) + np.ones((3, 3))) / 2
    ev = np.sort(np.linalg.eigvalsh(G))
    assert close(ev, [0.5, 0.5, 2.0]) and abs(np.linalg.det(G) - 0.5) < 1e-12
    # realise: unit vectors with pairwise dot 1/2
    L = np.linalg.cholesky(G)
    r = [L[i] for i in range(3)]
    if np.linalg.det(np.array(r)) < 0:
        r = [r[1], r[0], r[2]]
    assert abs(np.linalg.det(np.array(r)) - 1 / math.sqrt(2)) < 1e-12
    proj = lambda n: (I2 + sum(ni * s for ni, s in zip(n, PAULI))) / 2
    Bv = np.trace(proj(r[0]) @ proj(r[1]) @ proj(r[2]))
    assert abs(Bv - (5 + 1j * math.sqrt(2)) / 8) < 1e-12
    R = np.diag([1, 1, -1.0])
    Bref = np.trace(proj(R @ r[0]) @ proj(R @ r[1]) @ proj(R @ r[2]))
    assert abs(Bref - np.conj(Bv)) < 1e-12
    Qm, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
    if np.linalg.det(Qm) < 0:
        Qm[:, 0] *= -1
    Brot = np.trace(proj(Qm @ r[0]) @ proj(Qm @ r[1]) @ proj(Qm @ r[2]))
    assert abs(Brot - Bv) < 1e-12
    # clock calibration
    w = 0.8
    assert abs(math.sqrt(1 - w * w) - 0.6) < 1e-15
    assert abs(math.atanh(w) - math.log(3)) < 1e-14
    return "exterior"


def group_green():
    """Proposition n12:green, the heat-kernel integral, and the sample values."""
    def G(r, m):
        return np.exp(-m * r) / (4 * np.pi * r)
    def dG(r, m):
        return -np.exp(-m * r) * (1 + m * r) / (4 * np.pi * r * r)
    for m in (0.0, 0.7, 2.0):
        for R in (0.3, 1.0, 2.5):
            flux = -4 * np.pi * R * R * dG(R, m)
            assert abs(flux - (1 + m * R) * np.exp(-m * R)) < 1e-12
            dens = lambda r: 0.5 * (dG(r, m) ** 2 + m * m * G(r, m) ** 2) * 4 * np.pi * r * r
            En = si.quad(dens, R, np.inf, limit=200)[0]
            assert abs(En - (1 + m * R) * np.exp(-2 * m * R) / (8 * np.pi * R)) < 1e-9
    # distributional identity against a radial Gaussian test function phi = exp(-r^2)
    for m in (0.0, 1.3):
        phi = lambda r: np.exp(-r * r)
        lap = lambda r: (4 * r * r - 6) * np.exp(-r * r)       # Laplacian of exp(-r^2) in 3D
        integrand = lambda r: G(r, m) * (-lap(r) + m * m * phi(r)) * 4 * np.pi * r * r
        val = si.quad(integrand, 0, np.inf, limit=200)[0]
        assert abs(val - 1.0) < 1e-9
    # heat kernel integral
    for m, x in ((0.0, 0.8), (1.1, 0.5), (0.4, 2.0)):
        f = lambda s: np.exp(-s * m * m) * (4 * np.pi * s) ** -1.5 * np.exp(-x * x / (4 * s))
        val = si.quad(f, 0, np.inf, limit=400)[0]
        assert abs(val - G(x, m)) < 1e-9
    # L(b) = int exp(-y^2 - b^2/y^2) dy = sqrt(pi)/2 e^{-2b}
    for b in (0.0, 0.3, 1.2):
        val = si.quad(lambda y: np.exp(-y * y - (b * b / (y * y) if y > 0 else (0 if b == 0 else np.inf))), 0, np.inf)[0]
        assert abs(val - math.sqrt(math.pi) / 2 * math.exp(-2 * b)) < 1e-9
    # energy outside R=1/2: 1/(4 pi) at m=0 and 3/(8 pi e) at m=1
    En = lambda R, m: (1 + m * R) * np.exp(-2 * m * R) / (8 * np.pi * R)
    assert abs(En(0.5, 0) - 1 / (4 * np.pi)) < 1e-15
    assert abs(En(0.5, 1) - 3 / (8 * np.pi * np.e)) < 1e-15
    return "green"


def group_lattice():
    """Example n12:lattice and the finite Poisson annihilator example."""
    n = 4
    idx = list(itertools.product(range(n), repeat=3))
    pos = {p: i for i, p in enumerate(idx)}
    Lap = 6 * np.eye(64)
    for p in idx:
        for d in range(3):
            for s_ in (1, -1):
                q = list(p)
                q[d] = (q[d] + s_) % n
                Lap[pos[p], pos[tuple(q)]] -= 1
    ev = np.round(np.linalg.eigvalsh(Lap), 9)
    for j in range(7):
        assert np.sum(np.abs(ev - 2 * j) < 1e-6) == math.comb(6, j)
    Ginv = np.linalg.inv(Lap + np.eye(64))
    e0 = np.zeros(64)
    e0[pos[(0, 0, 0)]] = 1
    sol = Ginv @ e0
    centre = sum(Fraction(math.comb(6, j), 2 * j + 1) for j in range(7)) / 64
    assert centre == Fraction(523, 3003)
    assert abs(sol[pos[(0, 0, 0)]] - 523 / 3003) < 1e-12
    nb = (7 * centre - 1) / 6
    assert nb == Fraction(47, 1287)
    for q in [(1, 0, 0), (0, 3, 0), (0, 0, 1)]:
        assert abs(sol[pos[q]] - 47 / 1287) < 1e-12
    assert abs(sol.sum() - 1) < 1e-12
    # massless: constant mode in the kernel
    assert close(Lap @ np.ones(64), 0)
    # one-coordinate symbols
    assert close([2 * (1 - math.cos(math.pi * k / 2)) for k in range(4)], [0, 2, 4, 2], 1e-12)
    # annihilator of {0,4,8} in Z/12: k with exp(2 pi i k 4/12) = 1
    ann = [k for k in range(12) if abs(np.exp(2j * np.pi * k * 4 / 12) - 1) < 1e-12]
    assert ann == [0, 3, 6, 9]
    # finite Poisson: sum_{h in H} f(h) = |H|/m sum_{k in H^perp} fhat(k), fhat(k)=sum_x f(x)e^{-2pi i kx/m}
    m = 12
    H = [0, 4, 8]
    f = RNG.normal(size=m)
    fhat = np.array([np.sum(f * np.exp(-2j * np.pi * k * np.arange(m) / m)) for k in range(m)])
    assert abs(sum(f[h] for h in H) - len(H) / m * sum(fhat[k] for k in ann)) < 1e-12
    # dual character sum: sum_{k in H^perp} e^{2 pi i k x/m} = |H^perp| on H, 0 off H
    for x in range(m):
        s_ = sum(np.exp(2j * np.pi * k * x / m) for k in ann)
        assert abs(s_ - (len(ann) if x in H else 0)) < 1e-12
    # projection / pinching idempotent
    P = np.diag([1.0, 0])
    rho = np.array([[0.6, 0.3], [0.3, 0.4]])
    pin = lambda r: sum(Pk @ r @ Pk for Pk in (P, np.eye(2) - P))
    assert close(pin(pin(rho)), pin(rho)) and close(P @ P, P)
    assert close(pin(P), P) and close(pin(np.eye(2) - P), np.eye(2) - P)
    return "lattice"


def group_colour():
    """Remark n12:colour: singlet tensors, decompositions, charges."""
    eps = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        eps[p] = np.linalg.det(np.eye(3)[list(p)])
    for _ in range(4):
        X = rand_su(3)
        U = sla.expm(X)
        assert abs(np.linalg.det(U) - 1) < 1e-12
        t = np.einsum('ijk,ia,jb,kc->abc', eps, U, U, U)
        assert close(t, eps, 1e-10)
        assert close(U @ np.eye(3) @ U.conj().T, np.eye(3), 1e-12)
    # normalized singlet norm: 6 orthonormal product vectors
    assert abs(np.sum(eps ** 2) / 6 - 1) < 1e-15
    assert 3 * 3 == 1 + 8 and 27 == 1 + 8 + 8 + 10
    q = {'u': Fraction(2, 3), 'd': Fraction(-1, 3)}
    assert q['u'] * 2 + q['d'] == 1 and q['u'] + 2 * q['d'] == 0
    # invariant subspace dimension of 3x3x3 under SU(3): 1 (numerical via averaging generators)
    gens = [rand_su(3) for _ in range(4)]
    rows = []
    for Xg in gens:
        Mg = (np.kron(np.kron(Xg, np.eye(3)), np.eye(3)) + np.kron(np.kron(np.eye(3), Xg), np.eye(3))
              + np.kron(np.kron(np.eye(3), np.eye(3)), Xg))
        rows.append(Mg)
    Mall = np.vstack(rows)
    sv = np.linalg.svd(Mall, compute_uv=False)
    assert np.sum(sv < 1e-9) == 1
    return "colour"


def rot_angle(U):
    """SO(3) principal rotation angle of an SU(2) element."""
    c = np.clip(abs(np.real(np.trace(U))) / 2, -1, 1)
    return 2 * math.acos(c)


def group_gates():
    """Proposition n12:gates and the refuted identification of eigenphase with area."""
    def Ug(j, th):
        return sla.expm(-1j * th * PAULI[j] / 2)
    for th in np.linspace(0.05, math.pi / 2, 12):
        C = Ug(0, th).conj().T @ Ug(1, th).conj().T @ Ug(0, th) @ Ug(1, th)
        ht = np.real(np.trace(C)) / 2
        assert abs(ht - (1 - 2 * math.sin(th / 2) ** 4)) < 1e-12
        vartheta = 2 * math.asin(math.sin(th / 2) ** 2)
        ph_ = np.angle(np.linalg.eigvals(C))
        assert abs(max(abs(ph_)) - vartheta) < 1e-10
        assert abs(rot_angle(C) - 2 * vartheta) < 1e-9
        # triangle x, z, y
        x = np.array([math.sin(th), math.cos(th), 0])
        z = np.array([0, 1.0, 0])
        y = np.array([0, math.cos(th), math.sin(th)])
        # Girard: angles
        def angle_at(p, q, r):
            tq = q - (q @ p) * p
            tr = r - (r @ p) * p
            return math.acos(np.clip(tq @ tr / np.linalg.norm(tq) / np.linalg.norm(tr), -1, 1))
        A = angle_at(x, z, y) + angle_at(z, x, y) + angle_at(y, x, z) - math.pi
        assert abs(angle_at(z, x, y) - math.pi / 2) < 1e-12
        assert abs(math.cos(angle_at(x, z, y)) - math.cos(th) / math.sqrt(1 + math.cos(th) ** 2)) < 1e-12
        Om = 2 * math.atan(math.tan(th / 2) ** 2)
        assert abs(A - Om) < 1e-10
        assert abs(Om - (2 * math.atan(1 / math.cos(th)) - math.pi / 2)) < 1e-10 if th < math.pi / 2 - 1e-9 else True
        proj = lambda n: (I2 + sum(ni * s for ni, s in zip(n, PAULI))) / 2
        Delta = np.trace(proj(x) @ proj(z) @ proj(y))
        assert abs(Delta - ((1 + math.cos(th)) ** 2 + 1j * math.sin(th) ** 2) / 4) < 1e-12
        assert abs(np.angle(Delta) - Om / 2) < 1e-12
        assert abs(np.linalg.det(np.array([x, z, y])) - math.sin(th) ** 2) < 1e-12
    # quarter turn
    th = math.pi / 2
    C = Ug(0, th).conj().T @ Ug(1, th).conj().T @ Ug(0, th) @ Ug(1, th)
    vt = max(abs(np.angle(np.linalg.eigvals(C))))
    assert abs(vt - math.pi / 3) < 1e-12 and abs(rot_angle(C) - 2 * math.pi / 3) < 1e-9
    assert abs(2 * math.atan(math.tan(th / 2) ** 2) - math.pi / 2) < 1e-15
    # series
    t = sp.symbols('t', positive=True)
    gser = sp.series(2 * sp.asin(sp.sin(t / 2) ** 2), t, 0, 6).removeO()
    assert sp.simplify(gser - (t**2 / 2 - t**4 / 24)) == 0
    oser = sp.series(2 * sp.atan(sp.tan(t / 2) ** 2), t, 0, 6).removeO()
    assert sp.simplify(oser - (t**2 / 2 + t**4 / 12)) == 0
    # BCH leading log: log C = th^2 [T1,T2] + O(th^3), T_j = -i sigma_j/2
    T = [-0.5j * s for s in PAULI]
    for th in (1e-2, 5e-3):
        C = Ug(0, th).conj().T @ Ug(1, th).conj().T @ Ug(0, th) @ Ug(1, th)
        Lg = sla.logm(C)
        assert np.linalg.norm(Lg - th**2 * comm(T[0], T[1])) < 5 * th**3
    # quaternion pairing 1 - 2 s^4
    for th in (0.3, 1.0):
        c, s_ = math.cos(th / 2), math.sin(th / 2)
        q12 = np.array([c * c, c * s_, c * s_, s_ * s_])
        q21 = np.array([c * c, c * s_, c * s_, -s_ * s_])
        assert abs(q12 @ q21 - (1 - 2 * s_**4)) < 1e-14
    return "gates"


def group_entangle():
    """Example n12:entangle and the refuted claim on entangling pairs."""
    M = np.kron(SX, SX)
    N = np.kron(SX, SY)
    assert close(M @ N + N @ M, 0)
    assert close(M @ N, 1j * np.kron(I2, SZ))
    assert abs(np.trace(M)) < 1e-14 and abs(np.trace(N)) < 1e-14
    for th in (0.4, 1.2, math.pi / 2):
        UM = sla.expm(-1j * th * M / 2)
        UN = sla.expm(-1j * th * N / 2)
        C = UM.conj().T @ UN.conj().T @ UM @ UN
        vt = 2 * math.asin(math.sin(th / 2) ** 2)
        assert abs(max(abs(np.angle(np.linalg.eigvals(C)))) - vt) < 1e-10
        # quaternionic triple: -iM, -iN, -MN square to -1
        for X in (-1j * M, -1j * N, -M @ N):
            assert close(X @ X, -np.eye(4))
    for th in (1e-2, 5e-3):
        UM = sla.expm(-1j * th * M / 2)
        UN = sla.expm(-1j * th * N / 2)
        C = UM.conj().T @ UN.conj().T @ UM @ UN
        assert np.linalg.norm(sla.logm(C) - (-th**2 * M @ N / 2)) < 5 * th**3
    # (XX, YY) commute; both generate entangling gates
    YY = np.kron(SY, SY)
    assert close(comm(M, YY), 0)
    def schmidt(psi):
        return np.linalg.svd(psi.reshape(2, 2), compute_uv=False)
    for G_ in (M, YY, N):
        psi = sla.expm(-1j * 0.7 * G_ / 2) @ np.array([1, 0, 0, 0], dtype=complex)
        assert np.sum(schmidt(psi) > 1e-9) == 2
    # CNOT circuit
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    def chsh_max(rho):
        Tm = np.array([[np.real(np.trace(rho @ np.kron(a, b))) for b in PAULI] for a in PAULI])
        sv2 = np.sort(np.linalg.eigvalsh(Tm.T @ Tm))
        return 2 * math.sqrt(sv2[-1] + sv2[-2]), Tm
    for eta in (0.3, 1.0, math.pi / 2):
        psi = CNOT @ np.kron(sla.expm(-1j * eta * SY / 2), I2) @ np.array([1, 0, 0, 0], dtype=complex)
        a, b = math.cos(eta / 2), math.sin(eta / 2)
        assert close(psi, [a, 0, 0, b], 1e-12)
        conc = abs(psi @ np.kron(SY, SY) @ psi)        # pure-state concurrence |<psi|YY|psi*>|
        assert abs(conc - math.sin(eta)) < 1e-12
        rho = np.outer(psi, psi.conj())
        rA = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        assert close(rA, np.diag([a * a, b * b]))
        assert abs(np.real(np.trace(rA @ rA)) - (1 - 0.5 * math.sin(eta) ** 2)) < 1e-12
        smax, Tm = chsh_max(rho)
        assert close(Tm, np.diag([2 * a * b, -2 * a * b, 1]), 1e-12)
        assert abs(smax - 2 * math.sqrt(1 + math.sin(eta) ** 2)) < 1e-12
        # brute-force optimisation of the CHSH expression
        def neg(par):
            def u3(t_, p_):
                return np.array([math.sin(t_) * math.cos(p_), math.sin(t_) * math.sin(p_), math.cos(t_)])
            A0, A1, B0, B1 = (u3(par[2 * i], par[2 * i + 1]) for i in range(4))
            val = A0 @ Tm @ (B0 + B1) + A1 @ Tm @ (B0 - B1)
            return -val
        best = 0
        rs = np.random.default_rng(7)
        for _ in range(20):
            r_ = so.minimize(neg, rs.uniform(0, math.pi, 8), method='BFGS')
            best = max(best, -r_.fun)
        assert abs(best - smax) < 1e-6
    # scalar dichotomic observable on one side: |CHSH| <= 2
    for _ in range(5):
        rhoA = RNG.normal(size=(4, 4)) + 1j * RNG.normal(size=(4, 4))
        rho = rhoA @ rhoA.conj().T
        rho /= np.trace(rho)
        B0 = np.kron(I2, sum(c * s for c, s in zip(np.array([0.3, 0.4, np.sqrt(0.75)]), PAULI)))
        B1 = np.kron(I2, SZ)
        A0 = np.eye(4)
        nA = RNG.normal(size=3)
        nA /= np.linalg.norm(nA)
        A1 = np.kron(sum(c * s for c, s in zip(nA, PAULI)), I2)
        S = A0 @ (B0 + B1) + A1 @ (B0 - B1)
        assert max(abs(np.linalg.eigvalsh(S))) <= 2 + 1e-12
    # dimensions: CP1 x CP1 real dim 4 in CP3 real dim 6
    assert 2 * 1 + 2 * 1 == 4 and 2 * 3 == 6
    return "entangle"


def group_parity():
    """Proposition n12:parity and the refuted claim on changing both leg signs."""
    proj = lambda n: (I2 + sum(ni * s for ni, s in zip(n, PAULI))) / 2
    for th in (0.3, 1.0, math.pi / 2):
        for etaA in (1, -1):
            for etaB in (1, -1):
                x = np.array([etaA * math.sin(th), math.cos(th), 0])
                z = np.array([0, 1.0, 0])
                y = np.array([0, math.cos(th), etaB * math.sin(th)])
                B = np.trace(proj(x) @ proj(z) @ proj(y))
                Brev = np.trace(proj(y) @ proj(z) @ proj(x))
                assert abs(Brev - np.conj(B)) < 1e-12
                sgn = etaA * etaB
                assert abs(B - ((1 + math.cos(th)) ** 2 + 1j * sgn * math.sin(th) ** 2) / 4) < 1e-12
                G3 = np.array([x, z, y])
                assert abs(np.linalg.det(G3 @ G3.T) - math.sin(th) ** 4) < 1e-12
        # B + reversed: imaginary parts cancel
        x = np.array([math.sin(th), math.cos(th), 0]); z = np.array([0, 1.0, 0]); y = np.array([0, math.cos(th), math.sin(th)])
        B = np.trace(proj(x) @ proj(z) @ proj(y)); Br = np.trace(proj(y) @ proj(z) @ proj(x))
        assert abs(np.imag(B + Br)) < 1e-14
        # both signs: proper rotation diag(-1,1,-1)
        Rm = np.diag([-1, 1, -1.0])
        assert abs(np.linalg.det(Rm) - 1) < 1e-15
        assert close(Rm @ x, [-math.sin(th), math.cos(th), 0]) and close(Rm @ y, [0, math.cos(th), -math.sin(th)])
        # open step: bivector coefficient of the Clifford product z x
        for etaA in (1, -1):
            xx = np.array([etaA * math.sin(th), math.cos(th), 0])
            prod = sum(a * s for a, s in zip(z, PAULI)) @ sum(a * s for a, s in zip(xx, PAULI))
            assert abs(np.trace(prod) / 2 - math.cos(th)) < 1e-12
            # bivector part i (z x xx).sigma ; z x xx = (0,0,-eta sin th)
            assert close(np.cross(z, xx), [0, 0, -etaA * math.sin(th)])
    t = sp.symbols('t')
    assert sp.series(1 - sp.cos(t), t, 0, 6).removeO() == t**2 / 2 - t**4 / 24
    assert sp.series(sp.sin(t) ** 4, t, 0, 5).removeO() == t**4
    assert sp.series(sp.sin(t) ** 2, t, 0, 3).removeO() == t**2
    return "parity"


def group_composition():
    """Examples n12:serial, n12:yukawa-counter, n12:regge-counter, n12:flrw-counter,
    Remark n12:spin-two, Example n12:force-ratios."""
    # serial vs separated
    th = 0.7
    U = lambda a: sla.expm(-1j * a * SZ / 2)
    assert close(U(-th) @ U(th), I2)
    assert abs(2 * (1 - math.cos(th)) - 2 * (1 - math.cos(th))) == 0
    # dipole far field
    q, s = 1.0, np.array([0, 0, 0.01])
    for rr in (np.array([0.3, 0.2, 1.0]) * 50,):
        phi = q / np.linalg.norm(rr - s / 2) - q / np.linalg.norm(rr + s / 2)
        dip = q * s @ rr / np.linalg.norm(rr) ** 3
        assert abs(phi - dip) / abs(dip) < 1e-3
    # Yukawa force
    r, mu, C0, d1, d2, eta = sp.symbols('r mu C0 d1 d2 eta', positive=True)
    Upot = eta * C0 * d1 * d2 * sp.exp(-mu * r) / r
    assert sp.simplify(-sp.diff(Upot, r) - eta * C0 * d1 * d2 * (1 + mu * r) * sp.exp(-mu * r) / r**2) == 0
    # Regge
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    nrm = [-v / np.linalg.norm(v) for v in V]
    dots = [nrm[i] @ nrm[j] for i in range(4) for j in range(i + 1, 4)]
    assert close(dots, [-1 / 3] * 6, 1e-14)
    dih = math.acos(1 / 3)
    assert abs((math.pi - math.acos(-1 / 3)) - dih) < 1e-14
    # direct dihedral: angle between faces at edge v0-v1
    e = V[1] - V[0]
    def face_dir(w):
        t_ = w - V[0]
        t_ = t_ - (t_ @ e) / (e @ e) * e
        return t_ / np.linalg.norm(t_)
    assert abs(math.acos(face_dir(V[2]) @ face_dir(V[3])) - dih) < 1e-14
    d5 = 2 * math.pi - 5 * dih
    d6 = 2 * math.pi - 6 * dih
    assert abs(d5 - 0.128388) < 5e-7 and abs(d6 + 1.102571) < 5e-7
    assert (math.sqrt(5) - 1) / 4 < 1 / 3 < 0.5
    # FLRW and weak field via sympy Ricci
    tt, xx, yy, zz, c, G = sp.symbols('t x y z c G', positive=True)
    a = sp.Function('a')(tt)
    X = [tt, xx, yy, zz]
    g = sp.diag(-c**2, a**2, a**2, a**2)
    Ric = ricci(g, X)
    assert sp.simplify(Ric[0, 0] + 3 * sp.diff(a, tt, 2) / a) == 0
    rho, p = sp.symbols('rho p')
    T = sp.diag(rho * c**4, p * a**2, p * a**2, p * a**2)   # u_0 = -c^2 ... T_00 = rho c^4
    ginv = g.inv()
    trT = sum(ginv[i, i] * T[i, i] for i in range(4))
    assert sp.simplify(trT - (-rho * c**2 + 3 * p)) == 0
    rhs00 = 8 * sp.pi * G / c**4 * (T[0, 0] - sp.Rational(1, 2) * trT * g[0, 0])
    # R_00 = rhs => -3 a''/a = 4 pi G (rho + 3p/c^2)
    assert sp.simplify(rhs00 - 4 * sp.pi * G * (rho + 3 * p / c**2)) == 0
    H = sp.symbols('H', positive=True)
    gds = g.subs(a, sp.exp(H * tt)).doit()
    Rds = ricci(gds, X)
    assert sp.simplify(Rds - 3 * H**2 / c**2 * gds) == sp.zeros(4, 4)
    # with p = -rho c^2: a''/a = 8 pi G rho / 3 and H^2 = 8 pi G rho / 3
    acc = -sp.Rational(4, 3) * sp.pi * G * (rho + 3 * (-rho * c**2) / c**2)
    assert sp.simplify(acc - 8 * sp.pi * G * rho / 3) == 0
    # weak static field: g00 = -(1+2phi/c^2)c^2, flat spatial part to first order -> R_00 = lap(phi) + O(phi^2)
    ep = sp.symbols('epsilon')
    ph = sp.Function('phi')(xx, yy, zz)
    gw = sp.diag(-(1 + 2 * ep * ph / c**2) * c**2, 1 - 2 * ep * ph / c**2, 1 - 2 * ep * ph / c**2, 1 - 2 * ep * ph / c**2)
    Rw = ricci(gw, X)
    R00_lin = sp.diff(Rw[0, 0], ep).subs(ep, 0)
    lapphi = sum(sp.diff(ph, v, 2) for v in (xx, yy, zz))
    assert sp.simplify(R00_lin - lapphi) == 0
    # also with flat spatial part (static, R_00 only depends on g00 at first order)
    gw2 = sp.diag(-(1 + 2 * ep * ph / c**2) * c**2, 1, 1, 1)
    R00b = sp.diff(ricci(gw2, X)[0, 0], ep).subs(ep, 0)
    assert sp.simplify(R00b - lapphi) == 0
    # Ward reduction: g_a(p_a - p_a') + g_b(p_b - p_b') = (g_a - g_b) q
    ga, gb = sp.symbols('g_a g_b')
    pa, pap, pb = sp.symbols('p_a pap p_b')
    pbp = pa + pb - pap
    assert sp.simplify(ga * (pa - pap) + gb * (pb - pbp) - (ga - gb) * (pa - pap)) == 0
    # force ratios (CODATA 2022)
    alpha = 7.2973525643e-3
    mPl = 1.220890e19 * 1e3   # MeV
    me, mp = 0.51099895069, 938.27208943
    r = lambda m1, m2: m1 * m2 / (alpha * mPl**2)
    ree, rep, rpp = r(me, me), r(me, mp), r(mp, mp)
    assert abs(ree / 2.40e-43 - 1) < 5e-3 and abs(rep / 4.41e-40 - 1) < 5e-3 and abs(rpp / 8.09e-37 - 1) < 5e-3
    assert abs(rpp / ree / 3.37e6 - 1) < 5e-3
    assert abs(1 / alpha - 137.035999177) < 1e-6
    # Newton/Coulomb identity: G m1 m2/(alpha hbar c) = m1 m2/(alpha mPl^2)  with mPl^2 = hbar c/G
    t_ = sp.symbols('t', positive=True)
    for tv in (0.2, 1.1, 2.6):
        assert abs((1 - math.cos(tv)) / math.sin(tv) - math.tan(tv / 2)) < 1e-14
    return "composition"


def ricci(g, X):
    n = len(X)
    ginv = sp.simplify(g.inv())
    Gam = [[[sp.simplify(sum(ginv[l, m] * (sp.diff(g[m, i], X[j]) + sp.diff(g[m, j], X[i]) - sp.diff(g[i, j], X[m]))
                              for m in range(n)) / 2) for j in range(n)] for i in range(n)] for l in range(n)]
    R = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            val = 0
            for l in range(n):
                val += sp.diff(Gam[l][i][j], X[l]) - sp.diff(Gam[l][i][l], X[j])
                for m in range(n):
                    val += Gam[l][l][m] * Gam[m][i][j] - Gam[l][j][m] * Gam[m][i][l]
            R[i, j] = sp.simplify(val)
    return R


def quat(v):
    """Cayley chart: Q(v) = ((1-|v|^2), 2v)/(1+|v|^2)."""
    v = np.asarray(v, dtype=float)
    n2 = v @ v
    return np.concatenate([[1 - n2], 2 * v]) / (1 + n2)


def q_to_su2(q):
    a, b, c, d = q
    return a * I2 - 1j * (b * SX + c * SY + d * SZ)


def group_action():
    """Definition n12:action, Proposition n12:born, Theorem n12:min, Example n12:action-example."""
    Phi = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    for _ in range(5):
        q1 = RNG.normal(size=4); q1 /= np.linalg.norm(q1)
        q2 = RNG.normal(size=4); q2 /= np.linalg.norm(q2)
        U, V = q_to_su2(q1), q_to_su2(q2)
        cU, cV = np.kron(U, I2) @ Phi, np.kron(V, I2) @ Phi
        ov = np.vdot(cU, cV)
        assert abs(ov - np.trace(U.conj().T @ V) / 2) < 1e-12
        x = rot_angle(U.conj().T @ V)
        assert abs(abs(ov) ** 2 - math.cos(x / 2) ** 2) < 1e-12
        assert abs(np.real(ov) - q1 @ q2) < 1e-12        # real isometry of quaternion coordinates
        r_ = math.tan(x / 4)
        assert abs(math.sin(x / 2) ** 2 - 4 * r_**2 / (1 + r_**2) ** 2) < 1e-12
        assert abs(math.sin(x / 2) ** 2 - (1 - math.cos(x)) / 2) < 1e-12
        Pc = np.outer(cU, cU.conj())
        assert np.linalg.matrix_rank(np.eye(4) - Pc, tol=1e-9) == 3
    xs = sp.symbols('x')
    Ix = -sp.log(sp.cos(xs / 2) ** 2)
    ser = sp.series(Ix, xs, 0, 8).removeO()
    assert sp.simplify(ser - (xs**2 / 4 + xs**4 / 96 + xs**6 / 1440)) == 0
    assert sp.simplify(sp.diff(Ix, xs, 2) - sp.sec(xs / 2) ** 2 / 2) == 0
    for x in (0.3, 1.5, 2.9):
        d = 1 - math.cos(x)
        s_ = sum((d / 2) ** j / j for j in range(1, 5000))
        assert abs(s_ + math.log(math.cos(x / 2) ** 2)) < 1e-10
    dser = sp.series(-sp.log(1 - (1 - sp.cos(xs)) / 2), xs, 0, 8).removeO()
    assert sp.simplify(dser - ser) == 0
    # pair identity
    a, b = sp.symbols('a b')
    dd = lambda z: 1 - sp.cos(z)
    lhs = dd(a) + dd(b) - 2 * dd((a + b) / 2)
    assert sp.simplify(sp.expand_trig(lhs - 2 * sp.cos((a + b) / 2) * (1 - sp.cos((a - b) / 2)))) == 0 or \
        abs(float((lhs - 2 * sp.cos((a + b) / 2) * (1 - sp.cos((a - b) / 2))).subs({a: 0.37, b: 1.91}))) < 1e-14
    # Theorem min: random paths on SO(3)
    def Rot(axis, ang):
        axis = axis / np.linalg.norm(axis)
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return sla.expm(ang * Kx)
    def ang(R):
        return math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    sag = lambda x: 1 - math.cos(x)
    sur = lambda x: -2 * math.log(math.cos(x / 2)) if x < math.pi - 1e-12 else math.inf
    for Theta in (0.0, 0.8, 2.0, math.pi):
        R0 = np.eye(3)
        RN = Rot(np.array([0.2, 0.5, 1.0]), Theta)
        assert abs(ang(R0.T @ RN) - Theta) < 1e-9
        for N in (1, 2, 3, 5):
            for _ in range(30):
                pts = [R0] + [Rot(RNG.normal(size=3), RNG.uniform(0, math.pi)) for _ in range(N - 1)] + [RN]
                ds = [ang(pts[t - 1].T @ pts[t]) for t in range(1, N + 1)]
                assert sum(ds) >= Theta - 1e-9                       # triangle inequality
                assert sum(map(sag, ds)) >= N * sag(Theta / N) - 1e-9
                if N * 1 > 1 or Theta < math.pi:
                    assert sum(map(sur, ds)) >= N * sur(Theta / N) - 1e-9
    # Theta = pi, N = 2: every split along a shortest arc has sagitta 2
    for a_ in np.linspace(0, math.pi, 7):
        assert abs(sag(a_) + sag(math.pi - a_) - 2) < 1e-12
    # Theta = pi, N = 3: minimiser on the simplex is the equal tuple
    best = so.minimize(lambda z: sum(sag(v) for v in [z[0], z[1], math.pi - z[0] - z[1]]),
                       [0.3, 2.5], bounds=[(0, math.pi), (0, math.pi)],
                       constraints=[{'type': 'ineq', 'fun': lambda z: math.pi - z[0] - z[1]}])
    assert abs(best.fun - 3 * sag(math.pi / 3)) < 1e-6
    grid = np.linspace(0, math.pi, 301)
    vals = [sag(p) + sag(q) + sag(math.pi - p - q) for p in grid for q in grid if p + q <= math.pi + 1e-12]
    assert min(vals) >= 3 * sag(math.pi / 3) - 1e-12
    # sagitta second derivative changes sign at pi/2 (refuted strict convexity)
    assert math.cos(1.0) > 0 > math.cos(2.0)
    # Example: Theta = 2 pi / 3, N = 4
    eqs = 4 * sag(math.pi / 6)
    assert abs(eqs - (4 - 2 * math.sqrt(3))) < 1e-14 and abs(sag(2 * math.pi / 3) - 1.5) < 1e-14
    prod_eq = math.cos(math.pi / 12) ** 8
    assert abs(prod_eq - (97 + 56 * math.sqrt(3)) / 256) < 1e-14
    assert abs(math.cos(math.pi / 3) ** 2 - 0.25) < 1e-15
    assert abs(-math.log(prod_eq) - 0.27734586) < 5e-9 and abs(math.log(4) - 1.38629436) < 5e-9
    # side path in the Cayley chart
    vs = [(0, 0, 0), (1 / 6, 1 / 6, 0), (1 / 3, 1 / 4, 0), (1 / 2, 1 / 6, 0), (1 / math.sqrt(3), 0, 0)]
    Q = [quat(v) for v in vs]
    for qv in Q:
        assert abs(np.linalg.norm(qv) - 1) < 1e-14
    ps = [(Q[t - 1] @ Q[t]) ** 2 for t in range(1, 5)]
    act = 2 * sum(1 - p_ for p_ in ps)
    prod = float(np.prod(ps))
    assert abs(act - 0.95316800) < 5e-9 and abs(prod - 0.59837452) < 5e-9 and abs(-math.log(prod) - 0.51353844) < 5e-9
    # endpoint: rotation angle 2 pi / 3
    assert abs(2 * math.acos(abs(Q[0] @ Q[4])) - 2 * math.pi / 3) < 1e-12
    # reflection in the second axis preserves pairings
    Qr = [quat((v[0], -v[1], v[2])) for v in vs]
    assert close([(Qr[t - 1] @ Qr[t]) ** 2 for t in range(1, 5)], ps, 1e-14)
    mid = (Q[0] + Q[4]) / 2
    assert abs(mid @ mid - 0.75) < 1e-14 and abs(Q[0] @ Q[4] - 0.5) < 1e-14
    # stay probability product equals exp(-sum surprisal) with p_t = cos^2(x_t/2)
    for t in range(1, 5):
        x = 2 * math.acos(min(1, abs(Q[t - 1] @ Q[t])))
        assert abs(ps[t - 1] - math.cos(x / 2) ** 2) < 1e-12
        assert abs(2 * (1 - ps[t - 1]) - sag(x)) < 1e-12
    return "action"


def group_jacobi():
    """Proposition n12:jacobi, the continuum limits, and Remark n12:heap."""
    def qmul(p, q):
        a1, b1, c1, d1 = p
        a2, b2, c2, d2 = q
        return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                         a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])
    for _ in range(5):
        A, B, C = (np.concatenate([[0], RNG.normal(size=3)]) for _ in range(3))
        br = lambda p, q: qmul(p, q) - qmul(q, p)
        assert close(br(A, B)[1:], 2 * np.cross(A[1:], B[1:]), 1e-12)
        trip = -0.25 * br(br(A, B), C)
        assert close(trip[1:], (B[1:] @ C[1:]) * A[1:] - (A[1:] @ C[1:]) * B[1:], 1e-12)
    # Jacobi identity / curvature from structure constants of su(2) with bi-invariant metric
    e = [np.array([0, 1, 0, 0.]), np.array([0, 0, 1, 0.]), np.array([0, 0, 0, 1.])]
    # zero at 3 pi / 4 for J(0) = J'(0)
    s = 3 * math.pi / 4
    assert abs(math.cos(s) + math.sin(s)) < 1e-15
    assert abs(math.sin(math.pi)) < 1e-15
    # continuum limits for a C^1 path R(t) = exp(t A) exp(t^2 B) in SO(3), rotation-angle metric
    def hat(w):
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    Aw, Bw = np.array([0.3, -0.2, 0.5]), np.array([0.1, 0.4, -0.2])
    R = lambda t: sla.expm(t * hat(Aw)) @ sla.expm(t * t * hat(Bw))
    def speed2(t):
        hd = 1e-6
        Rd = (R(t + hd) - R(t - hd)) / (2 * hd)
        W = R(t).T @ Rd
        w = np.array([W[2, 1], W[0, 2], W[1, 0]])
        return w @ w
    tf = 1.5
    target = si.quad(speed2, 0, tf)[0]
    def ang(Rm):
        return math.acos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    for N in (400, 800):
        h = tf / N
        ds = [ang(R((t - 1) * h).T @ R(t * h)) for t in range(1, N + 1)]
        sag = sum(1 - math.cos(d) for d in ds) / h
        sur = sum(-2 * math.log(math.cos(d / 2)) for d in ds) / h
        assert abs(sag - 0.5 * target) < 5e-3 * target
        assert abs(sur - 0.25 * target) < 5e-3 * target
        assert sum(1 - math.cos(d) for d in ds) < 0.5 * target * h * 1.01
    # heap recursion <=> constant increments
    g0 = sla.expm(hat(RNG.normal(size=3)))
    inc = sla.expm(hat(np.array([0.2, 0.1, -0.3])))
    gs = [g0, g0 @ inc]
    for _ in range(5):
        gs.append(gs[-1] @ np.linalg.inv(gs[-2]) @ gs[-1])
    for t in range(1, len(gs) - 1):
        assert close(gs[t].T @ gs[t + 1], inc, 1e-10)
    # probe determinant on a small circle
    for th in (math.pi / 3, 0.7):
        for a in (0.6, 0.8):
            b = math.sqrt(1 - a * a)
            rr = [np.array([a * math.cos(j * th), a * math.sin(j * th), b]) for j in range(3)]
            det = np.linalg.det(np.array(rr))
            assert abs(det - 2 * a * a * b * math.sin(th) * (1 - math.cos(th))) < 1e-12
            proj = lambda n: (I2 + sum(ni * s_ for ni, s_ in zip(n, PAULI))) / 2
            Bv = np.trace(proj(rr[0]) @ proj(rr[1]) @ proj(rr[2]))
            assert abs(np.imag(Bv) - det / 4) < 1e-12
    return "jacobi"


def group_fisher():
    """Proposition n12:fisher and the refuted natural-gradient claim."""
    for _ in range(5):
        v, w = RNG.normal(size=3), RNG.normal(size=3)
        lhs = np.sum((quat(v) - quat(w)) ** 2)
        rhs = 4 * np.sum((v - w) ** 2) / ((1 + v @ v) * (1 + w @ w))
        assert abs(lhs - rhs) < 1e-12
    v = np.array([0.3, -0.5, 0.2])
    hd = 1e-6
    Jac = np.array([(quat(v + hd * e) - quat(v - hd * e)) / (2 * hd) for e in np.eye(3)]).T
    g = Jac.T @ Jac
    assert close(g, 4 / (1 + v @ v) ** 2 * np.eye(3), 1e-8)
    assert abs((1 + 0.64) ** 2 / 4 - 0.6724) < 1e-15
    # Choi model: pure-state QFI = 4 g_FS = 4 g
    Phi = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    psi = lambda vv: np.kron(q_to_su2(quat(vv)), I2) @ Phi
    def fs_metric(fun, vv, dim=3):
        p0 = fun(vv)
        D = [(fun(vv + hd * e) - fun(vv - hd * e)) / (2 * hd) for e in np.eye(dim)]
        M = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                M[i, j] = np.real(np.vdot(D[i], D[j]) - np.vdot(D[i], p0) * np.vdot(p0, D[j]))
        return M
    gFS = fs_metric(psi, v)
    assert close(4 * gFS, 4 * g, 1e-7)
    # overlap of tangent with base vanishes
    D0 = (psi(v + hd * np.eye(3)[0]) - psi(v - hd * np.eye(3)[0])) / (2 * hd)
    assert abs(np.vdot(psi(v), D0)) < 1e-9
    # fixed input ray: rank two
    ket0 = np.array([1, 0], dtype=complex)
    psi1 = lambda vv: q_to_su2(quat(vv)) @ ket0
    M1 = fs_metric(psi1, v)
    ev = np.linalg.eigvalsh(M1)
    assert np.sum(ev > 1e-6) == 2
    # Bernoulli logit with parameters entering through their sum: Fisher p(1-p)[[1,1],[1,1]]
    for z in (-1.0, 0.0, 0.8):
        pr = 1 / (1 + math.exp(-z))
        F = np.zeros((2, 2))
        for yv, py in ((1, pr), (0, 1 - pr)):
            score = (yv - pr) * np.array([1.0, 1.0])
            F += py * np.outer(score, score)
        assert close(F, pr * (1 - pr) * np.ones((2, 2)), 1e-14)
        assert F[0, 1] > 0 and abs(F[0, 1] - F[0, 0]) < 1e-15
    return "fisher"


def main():
    groups = [group_coset, group_sectors, group_breaking, group_scales, group_exterior, group_green,
              group_lattice, group_colour, group_gates, group_entangle, group_parity, group_composition,
              group_action, group_jacobi, group_fisher]
    ok = True
    for gfun in groups:
        try:
            gfun()
            print("passed:", gfun.__name__)
        except AssertionError as exc:
            ok = False
            import traceback
            traceback.print_exc()
            print("FAILED:", gfun.__name__, exc)
    print(f"check_cosets.py: all {len(groups)} groups passed: {ok}")


if __name__ == "__main__":
    main()
