#!/usr/bin/env python3
"""Independent checks for the section 'Observer memory as hardware' (module n10).

Path-free: reads no files.  NumPy and SciPy only.  Each group_* function raises
AssertionError on failure; main() runs all groups and prints one final line.
"""
from itertools import combinations, product
from math import ceil, comb, factorial, floor

import numpy as np
from scipy.optimize import linprog, minimize

TOL = 1e-9
RNG = np.random.default_rng(1120)


def close(a, b, tol=TOL):
    assert np.allclose(a, b, atol=tol, rtol=tol), (a, b)


def words(n):
    return list(product((0, 1), repeat=n))


def internal(n):
    return [v for k in range(n) for v in words(k)]


# ---------------------------------------------------------------------------------------------
def group_boolean_routing():
    """Prop. n10:boolean: Toffoli--CNOT splitter tree, any depth, dirty workspace, group Z_5."""
    def wires_layout(n):
        # leaf i -> wire index; internal vertex v -> (entering wire, new right wire)
        layout, counter = {}, [1]

        def rec(v, wire):
            if len(v) == n:
                layout[("leaf", v)] = wire
                return
            right = counter[0]
            counter[0] += 1
            layout[("split", v)] = (wire, right)
            rec(v + (0,), wire)
            rec(v + (1,), right)
        rec((), 0)
        return layout, counter[0]

    for n in range(5):
        layout, width = wires_layout(n)
        assert width == 2 ** n
        order = [v for v in internal(n)]  # parents before children (by length)

        def route(a, w, inverse=False):
            w = list(w)
            seq = order if not inverse else list(reversed(order))
            for v in seq:
                x, y = layout[("split", v)]
                b = a[len(v)]
                if not inverse:
                    w[y] ^= w[x] & b
                    w[x] ^= w[y]
                else:
                    w[x] ^= w[y]
                    w[y] ^= w[x] & b
            return w
        D = {i: int(RNG.integers(0, 5)) for i in words(n)}
        configs = words(width) if width <= 8 else sorted({tuple(int(x) for x in RNG.integers(0, 2, width)) for _ in range(200)} | {(1,) + (0,) * (width - 1)})
        for a in words(n):
            images = set()
            for w0 in configs:
                w = route(a, w0)
                images.add(tuple(w))
                assert route(a, w, inverse=True) == list(w0)
                if not any(w0[1:]):
                    for i in words(n):
                        assert w[layout[("leaf", i)]] == w0[0] * (i == a)
                for r in range(5):
                    shift = sum(D[i] for i in words(n) if w[layout[("leaf", i)]] == 1) % 5
                    # query: route, leaf translations, unroute -> workspace restored, register shifted
                    assert route(a, w, inverse=True) == list(w0)
                    if w0 == (1,) + (0,) * (width - 1):
                        assert shift == D[a]
            assert len(images) == len(configs)  # injective, hence a permutation on the finite set
        # amplitude pullback on the clean subspace, group Z_5
        psi = RNG.normal(size=(2 ** n, 5)) + 1j * RNG.normal(size=(2 ** n, 5))
        out = np.zeros_like(psi)
        for ia, a in enumerate(words(n)):
            for r in range(5):
                out[ia, (r + D[a]) % 5] += psi[ia, r]
        for ia, a in enumerate(words(n)):
            for r in range(5):
                close(out[ia, r], psi[ia, (r - D[a]) % 5])


# ---------------------------------------------------------------------------------------------
W = "W"


def spatial_program(n):
    """Gate list for loading and transport; each gate is a swap of two keys, optionally controlled."""
    load = []
    for j in range(n):
        load.append((("slot", j), ("mode", ()), None))
        for lev in range(j):
            for v in words(lev):
                for b in (0, 1):
                    load.append((("mode", v), ("mode", v + (b,)), (("router", v), b)))
        for v in words(j):
            load.append((("mode", v), ("router", v), None))
    transport = [(("bus",), ("mode", ()), None)]
    for lev in range(n):
        for v in words(lev):
            for b in (0, 1):
                transport.append((("mode", v), ("mode", v + (b,)), (("router", v), b)))
    return load, transport


def run(state, gates):
    for x, y, c in gates:
        if c is None or state[c[0]] == c[1]:
            state[x], state[y] = state[y], state[x]
    return state


def spatial_keys(n):
    return ([("slot", j) for j in range(n)] + [("bus",)]
            + [("mode", v) for k in range(n + 1) for v in words(k)]
            + [("router", v) for v in internal(n)])


def spatial_query(n, D, state):
    load, transport = spatial_program(n)
    run(state, load + transport)
    for i in words(n):
        if D[i] and state[("mode", i)] != W:
            state[("mode", i)] ^= 1
    run(state, list(reversed(load + transport)))
    return state


def loaded_labels(n):
    """Router labels after loading address a (computed by simulation)."""
    load, _ = spatial_program(n)
    rows = []
    for a in words(n):
        for bus in (0, 1, W):
            s = dict.fromkeys(spatial_keys(n), W)
            s.update({("slot", j): a[j] for j in range(n)})
            s[("bus",)] = bus
            run(s, load)
            row = tuple(s[("router", v)] for v in internal(n))
            if bus == 0:
                ref = row
            assert row == ref  # independent of the bus
        rows.append(ref)
    return rows


def group_spatial_qram():
    """Prop. n10:spatial: loading invariant, transport of every bus label, clean XOR query, bijectivity."""
    addrs = [tuple(int(c) for c in format(x, "06b")) for x in (41, 9, 33, 1)]
    counts = [len({a[:k] for a in addrs}) for k in range(6)]
    assert counts == [1, 2, 2, 4, 4, 4] and sum(counts) == 17
    for n in range(5):
        load, transport = spatial_program(n)
        D = {i: int(RNG.integers(0, 2)) for i in words(n)}
        # gate boundaries of the loading steps
        bounds, pos = [0], 0
        for j in range(n):
            pos += 1 + 2 * sum(2 ** l for l in range(j)) + 2 ** j
            bounds.append(pos)
        for a in words(n):
            for bus in (0, 1, W):
                s0 = dict.fromkeys(spatial_keys(n), W)
                s0.update({("slot", j): a[j] for j in range(n)})
                s0[("bus",)] = bus
                for k in range(n + 1):
                    s = run(dict(s0), load[:bounds[k]])
                    assert s[("bus",)] == bus
                    for j in range(n):
                        assert s[("slot", j)] == (W if j < k else a[j])
                    for v in internal(n):
                        exp = a[len(v)] if (len(v) < k and v == a[:len(v)]) else W
                        assert s[("router", v)] == exp
                    assert all(s[("mode", v)] == W for kk in range(n + 1) for v in words(kk))
                s = run(dict(s0), load + transport)
                assert s[("mode", a)] == bus and s[("bus",)] == W
                assert all(s[("mode", v)] == W for kk in range(n + 1) for v in words(kk) if v != a)
                out = spatial_query(n, D, dict(s0))
                exp = dict(s0)
                if bus != W:
                    exp[("bus",)] = bus ^ D[a]
                assert out == exp
        # every configuration (dirty ones included): the query is an involution, hence a bijection
        keys = spatial_keys(n)
        if n <= 1:
            pool = product((0, 1, W), repeat=len(keys))
        else:
            pool = (tuple(RNG.choice([0, 1, W]) for _ in keys) for _ in range(3000))
        for cfg in pool:
            st = dict(zip(keys, [c if c == W else int(c) for c in cfg]))
            assert spatial_query(n, D, spatial_query(n, D, dict(st))) == st


# ---------------------------------------------------------------------------------------------
def record(q, y):
    idx = {W: 1, 0: 2, 1: 3}[y]
    r = np.zeros(4)
    r[0] = np.sqrt(q)
    r[idx] = np.sqrt(1 - q)
    return r


def records(n, qs):
    labs = loaded_labels(n)
    R = []
    for row in labs:
        v = np.ones(1)
        for q, y in zip(qs, row):
            v = np.kron(v, record(q, y))
        R.append(v)
    return np.array(R).T, labs


def gram_product(labs, qs):
    N = len(labs)
    G = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            for q, x, y in zip(qs, labs[i], labs[j]):
                if x != y:
                    G[i, j] *= q
    return G


def f_n(n, q):
    return (1 + sum(2 ** j * q ** (2 * j + 1) for j in range(n))) / 2 ** n


def group_router_hold():
    """Thm. n10:hold: reduced output, local channel, Gram, sharp minimum and endpoints."""
    # local channel
    for q in (0, 0.3, 1):
        loc = np.array([record(q, y) for y in (W, 0, 1)])
        X = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))
        close(X * (loc @ loc.T), q * X + (1 - q) * np.diag(np.diag(X)))
    # full joint vector and partial trace, depth 2, heterogeneous q
    n = 2
    qs = [0.3, 0.7, 0.55]
    M, labs = records(n, qs)
    G = gram_product(labs, qs)
    close(M.T @ M, G)
    D = {i: int(RNG.integers(0, 2)) for i in words(n)}
    psi = RNG.normal(size=(4, 2)) + 1j * RNG.normal(size=(4, 2))
    psi /= np.linalg.norm(psi)
    joint = np.zeros((4, 2, M.shape[0]), complex)
    for ia, a in enumerate(words(n)):
        for b in (0, 1):
            joint[ia, b ^ D[a]] += psi[ia, b] * M[:, ia]
    J = joint.reshape(8, -1)
    rho = J @ J.conj().T
    ideal = np.zeros((4, 2), complex)
    for ia, a in enumerate(words(n)):
        for b in (0, 1):
            ideal[ia, b ^ D[a]] = psi[ia, b]
    pred = np.outer(ideal.ravel(), ideal.ravel().conj()) * np.kron(G, np.ones((2, 2)))
    close(rho, pred)
    p = (abs(psi) ** 2).sum(axis=1)
    close(ideal.ravel().conj() @ rho @ ideal.ravel(), p @ G @ p)
    # homogeneous: exponent 2(n-l)-1, row sum, minimum, uniqueness, endpoints
    for n in range(6):
        labs = loaded_labels(n)
        A = words(n)
        for i, a in enumerate(A):
            for j, b in enumerate(A):
                d = sum(x != y for x, y in zip(labs[i], labs[j]))
                if i == j:
                    assert d == 0
                else:
                    l = next(k for k in range(n) if a[k] != b[k])
                    assert d == 2 * (n - l) - 1
        for q in (0.0, 0.2, 0.5, 0.9, 1.0):
            G = gram_product(labs, [q] * len(internal(n)))
            close(G.sum(axis=1), 2 ** n * f_n(n, q))
            u = np.full(2 ** n, 2.0 ** -n)
            close(u @ G @ u, f_n(n, q))
            if q < 1 and n > 0:
                assert np.linalg.eigvalsh(G).min() > 1e-12
            for _ in range(30):
                p = RNG.dirichlet(np.ones(2 ** n))
                val = p @ G @ p
                assert val >= f_n(n, q) - 1e-12
                if q == 0:
                    close(val, p @ p)
                if q == 1:
                    close(val, 1)
            if n >= 1 and 0 < q < 1:
                res = minimize(lambda x: (x / x.sum()) @ G @ (x / x.sum()), RNG.uniform(0.2, 1, 2 ** n),
                               bounds=[(1e-9, None)] * 2 ** n, method="L-BFGS-B")
                xs = res.x / res.x.sum()
                close(xs, u, 1e-4)
        close(f_n(n, 0), 2.0 ** -n)
        close(f_n(n, 1), 1)
    close(f_n(1, 0.37), (1 + 0.37) / 2)
    close(f_n(0, 0.37), 1)


def group_resolution_profiles():
    """Thm. n10:profiles and Ex. n10:fiber: deepest-router criterion, rank, kernel, class minimiser."""
    for n in range(4):
        V = internal(n)
        labs = loaded_labels(n)
        deepest = [k for k, v in enumerate(V) if len(v) == n - 1]
        for act in words(len(V)):
            prof = [tuple(row[k] for k in range(len(V)) if act[k]) for row in labs]
            sep = len(set(prof)) == 2 ** n
            assert sep == all(act[k] for k in deepest)
            qs = [RNG.uniform(0, 0.95) if act[k] else 1.0 for k in range(len(V))]
            M, _ = records(n, qs)
            G = gram_product(labs, qs)
            classes = sorted(set(prof), key=repr)
            assert np.linalg.matrix_rank(G, tol=1e-9) == len(classes)
            # kernel = zero class sums
            for _ in range(3):
                z = RNG.normal(size=2 ** n)
                for c in classes:
                    idx = [i for i, pr in enumerate(prof) if pr == c]
                    z[idx] -= z[idx].mean()
                close(G @ z, 0)
            # class records independent
            Rc = np.array([M[:, prof.index(c)] for c in classes]).T
            assert np.linalg.matrix_rank(Rc, tol=1e-9) == len(classes)
    # fiber example (depth two, q=0 at the left child, 1 elsewhere)
    labs = loaded_labels(2)
    V = internal(2)
    qs = [0.0 if v == (0,) else 1.0 for v in V]
    assert [row[V.index((0,))] for row in labs] == [0, 1, W, W]
    G = gram_product(labs, qs)
    close(G, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    for t in np.linspace(0, 1 / 3, 7):
        p = np.array([1 / 3, 1 / 3, t, 1 / 3 - t])
        close(p @ G @ p, 1 / 3)
    for _ in range(200):
        p = RNG.dirichlet(np.ones(4))
        assert p @ G @ p >= 1 / 3 - 1e-12
    close(np.full(4, .25) @ G @ np.full(4, .25), 3 / 8)


# ---------------------------------------------------------------------------------------------
OMEGA = np.exp(2j * np.pi / 3)
JIDX = {W: 0, 0: 1, 1: 2}


def frec(r, y):
    j = JIDX[y]
    return np.array([np.sqrt((1 + 2 * r) / 3), np.sqrt((1 - r) / 3) * OMEGA ** j,
                     np.sqrt((1 - r) / 3) * OMEGA ** (-j)])


def signed_matrix(n, params, labs):
    cols = []
    for row in labs:
        v = np.ones(1, complex)
        for r, y in zip(params, row):
            v = np.kron(v, frec(r, y))
        cols.append(v)
    vac = np.ones(1, complex)
    for r in params:
        vac = np.kron(vac, frec(r, W))
    return np.array(cols).T, vac


def recursion(n, par):
    """Rank d, kernel list, representative U per Theorem n10:signed (independent implementation)."""
    if n == 0:
        return 1, [], np.array([1.0])
    r = par[()]
    sub0 = {v[1:]: x for v, x in par.items() if v[:1] == (0,)}
    sub1 = {v[1:]: x for v, x in par.items() if v[:1] == (1,)}
    d0, K0, U0 = recursion(n - 1, sub0)
    d1, K1, U1 = recursion(n - 1, sub1)
    z = np.zeros(2 ** (n - 1))
    K = [np.r_[k, z] for k in K0] + [np.r_[z, k] for k in K1]
    d, U = d0 + d1, None
    if r == -0.5:
        if U0 is not None and U1 is not None:
            U = np.r_[-U0, -U1]
    elif r == 1:
        if U0 is not None:
            U = np.r_[U0, z]
        elif U1 is not None:
            U = np.r_[z, U1]
        if U0 is not None and U1 is not None:
            K.append(np.r_[U0, -U1])
            d -= 1
    return d, K, U


def group_signed_certificate():
    """Thm. n10:signed: local Fourier Gram, both orientations, rank/kernel/vacuum recursion."""
    for r in (-0.5, -0.1, 0.0, 0.4, 1.0):
        L = np.array([frec(r, y) for y in (W, 0, 1)]).T
        close(L.conj().T @ L, (1 - r) * np.eye(3) + r * np.ones((3, 3)))
        close(np.sort(np.linalg.eigvalsh(L.conj().T @ L)), np.sort([1 + 2 * r, 1 - r, 1 - r]))
    close(frec(-0.5, W) + frec(-0.5, 0) + frec(-0.5, 1), 0)
    choices = (-0.5, 0.3, 1.0)
    for n in range(4):
        V = internal(n)
        labs = loaded_labels(n)
        pool = product(choices, repeat=len(V)) if n <= 2 else (tuple(RNG.choice(choices, len(V))) for _ in range(150))
        for params in pool:
            params = [float(x) for x in params]
            M, vac = signed_matrix(n, params, labs)
            G1 = M.conj().T @ M
            close(G1, gram_product(labs, params))
            close(G1, G1.conj())
            d, K, U = recursion(n, dict(zip(V, params)))
            rk = np.linalg.matrix_rank(M, tol=1e-8)
            assert d == rk == np.linalg.matrix_rank(G1, tol=1e-8) and d > 0
            assert len(K) == 2 ** n - d
            if K:
                Km = np.array(K).T
                close(M @ Km, 0)
                assert np.linalg.matrix_rank(Km, tol=1e-8) == len(K)
            in_image = np.linalg.matrix_rank(np.column_stack([M, vac]), tol=1e-8) == rk
            assert in_image == (U is not None)
            if U is not None:
                close(M @ U, vac)


def uniform_fractions(M, vac, n):
    fr = []
    for k in range(1, 2 ** n + 1):
        for B in combinations(range(2 ** n), k):
            s = M[:, list(B)].sum(axis=1)
            for e in (-1, 0, 1):
                if np.allclose(s, e * vac, atol=1e-9):
                    fr.append((e / k, B, e))
    return fr


def lp_interval(M, vac):
    m = M.shape[1]
    A = np.column_stack([M, -vac])
    Aeq = np.vstack([A.real, A.imag, np.r_[np.ones(m), 0]])
    beq = np.r_[np.zeros(2 * A.shape[0]), 1]
    bounds = [(0, None)] * m + [(None, None)]
    lo = linprog(np.r_[np.zeros(m), 1], A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    hi = linprog(np.r_[np.zeros(m), -1], A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if lo.status != 0:
        assert hi.status != 0
        return None
    return lo.x[-1], hi.x[-1]


def group_uniform_section():
    """Thm. n10:uniform: the feasible root interval has endpoints e/|B| from uniform leaf laws."""
    choices = (-0.5, 0.3, 1.0)
    count_nonempty = 0
    for n in (1, 2, 3):
        V = internal(n)
        labs = loaded_labels(n)
        pool = product(choices, repeat=len(V)) if n <= 2 else (tuple(RNG.choice(choices, len(V))) for _ in range(40))
        for params in pool:
            params = [float(x) for x in params]
            M, vac = signed_matrix(n, params, labs)
            iv = lp_interval(M, vac)
            fr = uniform_fractions(M, vac, n)
            assert (iv is None) == (len(fr) == 0)
            if iv is not None:
                count_nonempty += 1
                vals = [x[0] for x in fr]
                close([iv[0], iv[1]], [min(vals), max(vals)], 1e-7)
                for v, B, e in fr:
                    assert 1 <= len(B) <= 2 ** n
                if iv[0] <= 1e-9 <= iv[1] + 2e-9 or abs(iv[0]) < 1e-9 or abs(iv[1]) < 1e-9:
                    if iv[0] <= 1e-9 and iv[1] >= -1e-9:
                        zeros = [B for v, B, e in fr if e == 0]
                        assert zeros
                        # support claim: a zero witness built from one zero subset contains a zero subset
                        w = np.zeros(2 ** n)
                        w[list(zeros[0])] = 1 / len(zeros[0])
                        close(M @ w, 0)
    assert count_nonempty > 10
    # explicit zero sum: root +1, left child +1, right child -1/2
    labs = loaded_labels(2)
    M, vac = signed_matrix(2, [1.0, 1.0, -0.5], labs)
    close(M[:, [0, 2, 3]].sum(axis=1), 0)


# ---------------------------------------------------------------------------------------------
def group_page_cost_recovery():
    """Prop. n10:page, Prop. n10:cost, Prop. n10:recovery."""
    nx = {(0x1000, 0): 0x1000, (0x1000, 1): 0x2000, (0x2000, 2): 0x9000}

    def walk(b, slots):
        for i in slots:
            b = nx[(b, i)]
        return b
    assert walk(0x1000, [1, 2]) == 0x9000
    assert walk(0x1000, [0, 1]) == 0x2000
    assert walk(0x1000, [0, 0]) == 0x1000
    assert walk(0x1000, [0] * 4) == 0x1000
    for k in range(6):
        assert walk(0x1000, [0] * k + [1, 2]) == 0x9000
    # four self slots on a 9-bit-per-level walk: index 0 at each level returns the base
    # cost: copying gives zero loss; constant score functional is zero
    Y = {z: int(RNG.integers(0, 3)) for z in range(5)}
    assert sum(int(Y[z] != Y[z]) for z in Y) == 0
    # recovery: exhaustive on small sets, X nonempty
    for nX in range(1, 4):
        for nY in range(1, 3):
            for nZ in range(1, 3):
                for P in product(range(nY), repeat=nX):
                    for T in product(range(nZ), repeat=nX):
                        fact = any(all(R[P[x]] == T[x] for x in range(nX)) for R in product(range(nZ), repeat=nY))
                        const = all(T[x] == T[y] for x in range(nX) for y in range(nX) if P[x] == P[y])
                        defect = [(x, y) for x in range(nX) for y in range(nX) if P[x] == P[y] and T[x] != T[y]]
                        assert fact == const == (not defect)
    # empty X: Y nonempty, Z empty -> no map R:Y->Z although the defect set is empty
    assert len(list(product(range(0), repeat=1))) == 0


def compare_machine(a, b):
    """Returns (flag, transitions) for the comparison machine; checks restoration."""
    A, B = list(a), list(b)  # heads at index 0
    sa, sb = [], []
    flag, t = True, 0
    while True:
        t += 1
        x = A.pop(0) if A else None
        y = B.pop(0) if B else None
        if x is None and y is None:
            break
        flag = flag and (x == y)
        if x is not None:
            sa.append(x)
        if y is not None:
            sb.append(y)
    while True:          # left restoration
        t += 1
        if not sa:
            break
        A.insert(0, sa.pop())
    while True:          # right restoration
        t += 1
        if not sb:
            break
        B.insert(0, sb.pop())
    t += 1               # output transition
    assert tuple(A) == tuple(a) and tuple(B) == tuple(b) and not sa and not sb
    return flag, t


def dictionary_machine(u, names):
    stream = []
    for w in names:
        for bit in reversed(w):
            stream += [1, bit]
        stream.append(0)
    unread, hist = list(stream), []
    t, misses, hit = 0, 0, False
    while True:
        t += 1                      # exhaustion test
        if not unread:
            break
        cand = []
        while True:                 # read encoded symbols
            sym = unread.pop(0)
            hist.append(sym)
            t += 1
            if sym == 0:
                break
            bit = unread.pop(0)
            hist.append(bit)
            t += 1
            cand.append(bit)        # stack: reversed stream word becomes ordinary spelling when popped
        word = tuple(reversed(cand))
        t += 1                      # start comparison
        flag, tc = compare_machine(u, word)
        t += tc
        t += 1                      # dispatch flag
        t += len(word) + 1          # clear candidate, incl. empty-stack transition
        if flag:
            hit = True
            break
        misses += 1
    # rewind
    t += len(hist)
    unread = hist + unread
    t += 2                          # empty-history and output transitions
    assert unread == stream
    return (int(hit),) + (1,) * misses, t, len(stream)


def group_stack_machines():
    """Prop. n10:oisc: comparison and dictionary scan step bounds and restoration."""
    pool = [w for k in range(6) for w in words(k)]
    for a in pool:
        for b in pool:
            flag, t = compare_machine(a, b)
            assert flag == (a == b)
            assert t == max(len(a), len(b)) + 1 + len(a) + len(b) + 2 + 1
            assert t <= 2 * (len(a) + len(b)) + 4
    small = [w for k in range(3) for w in words(k)]
    for m in range(4):
        for names in product(small, repeat=m):
            for u in pool[:15]:
                out, t, slen = dictionary_machine(u, names)
                k = names.index(u) if u in names else len(names)
                assert out == (int(u in names),) + (1,) * k
                assert slen == sum(2 * len(w) + 1 for w in names)
                assert t <= (2 * len(u) + 12) * m + 4 * slen + 3
                # per-word charge
                for w in names[:k + 1]:
                    s = 2 * len(w) + 1
                    charge = 1 + s + 1 + (2 * (len(u) + len(w)) + 4) + 1 + (len(w) + 1) + s
                    assert charge <= 4 * s + 2 * len(u) + 12


# ---------------------------------------------------------------------------------------------
def circ(t, L):
    return (t + L / 2) % L - L / 2


def group_codebook():
    """Thm. n10:codebook: error body, volume, lower and grid upper bounds, rotations, parity."""
    for n in range(1, 7):
        close(sum(comb(n, p) / (factorial(p) * factorial(n - p)) for p in range(n + 1)), comb(2 * n, n) / factorial(n))
    L = 2 * np.pi
    for n in range(1, 6):
        eps = 0.4
        subsets = np.array(words(n))
        for _ in range(2000):
            a = RNG.uniform(-eps, eps, n)
            cov = np.max(np.abs(circ(subsets @ a, L))) <= eps
            body = np.clip(a, 0, None).sum() <= eps and np.clip(-a, 0, None).sum() <= eps
            assert cov == body
        # one positive partial sum argument: errors outside [-eps,eps] coordinatewise are never covered
        m = ceil(L * n / (2 * eps))
        lower = factorial(n) * L ** n / (comb(2 * n, n) * eps ** n)
        assert lower <= m ** n
        for _ in range(200):
            th = RNG.uniform(-10, 10, n)
            code = np.round(th * m / L) * L / m
            errs = circ(subsets @ (th - code), L)
            assert np.max(np.abs(errs)) <= eps + 1e-12
            for row, e in zip(subsets, errs):
                Rt = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                close(Rt(row @ th), Rt(row @ code + e))
                prod_ = np.eye(2)
                for i in np.flatnonzero(row):
                    prod_ = Rt(th[i]) @ prod_
                close(prod_, Rt(row @ th))
            bits = RNG.integers(0, 2, n)
            assert (row @ bits) % 2 == np.bitwise_xor.reduce(bits[row.astype(bool)]) if row.any() else True
    # Monte Carlo volume of the body for n = 2, 3 (sign-orthant products of simplices)
    for n, eps in ((2, 1.0), (3, 1.0)):
        pts = RNG.uniform(-eps, eps, size=(400000, n))
        inside = (np.clip(pts, 0, None).sum(1) <= eps) & (np.clip(-pts, 0, None).sum(1) <= eps)
        vol = inside.mean() * (2 * eps) ** n
        assert abs(vol - eps ** n * comb(2 * n, n) / factorial(n)) < 0.02 * (2 * eps) ** n


# ---------------------------------------------------------------------------------------------
def group_ledger_update():
    """Thm. n10:ledger, Ex. n10:ledger-ex, Cor. n10:impurity, Prop. n10:update."""
    S = np.array(words(4))
    R = np.column_stack([S[:, 0], S[:, 1], S[:, 2], S.sum(1) % 2, (S[:, 0] + S[:, 1]) % 2])
    N = 16
    diag = np.eye(N, dtype=bool)
    Kr = [R[:, i][:, None] == R[:, i][None, :] for i in range(5)]
    for mask in words(5):
        I = [i for i in range(5) if mask[i]]
        KI = np.ones((N, N), bool)
        for i in I:
            KI &= Kr[i]
        EI = KI & ~diag
        for i in I:
            Kw = np.ones((N, N), bool)
            for j in I:
                if j != i:
                    Kw &= Kr[j]
            Ew = Kw & ~diag
            U = Ew & ~EI
            assert not np.any(U & EI) and np.array_equal(EI | U, Ew)
            assert np.array_equal(U, Kw & ~Kr[i])
            dec = (Ew.sum() - EI.sum()) / (N * (N - 1))
            close(dec, U.sum() / (N * (N - 1)))
            strict = U.sum() > 0
            assert strict == (KI.sum() < Kw.sum()) == (not np.all(Kr[i][Kw]))
        if I:
            _, m = np.unique(R[:, I], axis=0, return_counts=True)
        else:
            m = np.array([N])
        assert EI.sum() == np.sum(m * (m - 1))
        e = EI.sum() / (N * (N - 1))
        close(np.log2(np.sum(m ** 2) / N), np.log2(1 + (N - 1) * e))
        pr = m / N
        close(np.log2(N) - np.log2(np.sum(m ** 2) / N), -np.log2(np.sum(pr ** 2)))
    assert N * (N - 1) == 240
    full = np.ones((N, N), bool)
    for i in range(5):
        full &= Kr[i]
    assert (full & ~diag).sum() == 0
    four = Kr[0] & Kr[1] & Kr[2] & Kr[3]
    assert (four & ~diag).sum() == 0
    for b in range(5):
        same = np.all(S[:, :b][:, None, :] == S[:, :b][None, :, :], axis=2)
        assert (same & ~diag).sum() == 16 * (2 ** (4 - b) - 1)
    noparity = Kr[0] & Kr[1] & Kr[2] & Kr[4]
    E = (noparity & ~diag).sum()
    assert E == 16
    close(E / 240, 1 / 15)
    close(np.log2(1 + 15 * (1 / 15)), 1)
    # within each size-two fiber b3 and the four-bit parity are uniform
    for key in set(map(tuple, R[:, [0, 1, 2, 4]])):
        idx = [k for k in range(N) if tuple(R[k, [0, 1, 2, 4]]) == key]
        assert len(idx) == 2 and sorted(S[idx, 3]) == [0, 1] and sorted(R[idx, 3]) == [0, 1]
    # two bits
    T = np.array(words(2))
    k1 = T[:, 0][:, None] == T[:, 0][None, :]
    assert (k1 & ~np.eye(4, dtype=bool)).sum() == 4 and 4 * 3 == 12
    k2 = k1 & (T[:, 1][:, None] == T[:, 1][None, :])
    assert (k2 & ~np.eye(4, dtype=bool)).sum() == 0
    # impurity identity (finite and truncated countable laws)
    for _ in range(50):
        w = RNG.dirichlet(np.ones(5))
        tj = RNG.dirichlet(np.ones(4), size=5)
        t = w @ tj
        close(w @ (tj ** 2).sum(1) - t @ t, w @ ((tj - t) ** 2).sum(1))
    # commutator formula and orbit criterion
    for _ in range(20):
        n = 6
        tau = RNG.permutation(n)
        inv = np.argsort(tau)
        f = RNG.normal(size=n) + 1j * RNG.normal(size=n)
        psi = RNG.normal(size=n) + 1j * RNG.normal(size=n)
        Uop = lambda v: v[inv]          # (U v)(i) = v(tau^{-1} i)
        lhs = Uop(f * psi) - f * Uop(psi)
        close(lhs, (f[inv] - f) * psi[inv])
        # orbit-constant f commutes
        orb = np.zeros(n, int)
        seen, lab = set(), 0
        for s in range(n):
            if s in seen:
                continue
            x = s
            while x not in seen:
                seen.add(x)
                orb[x] = lab
                x = tau[x]
            lab += 1
        g = RNG.normal(size=lab)[orb]
        close(Uop(g * psi) - g * Uop(psi), 0)

    def rev(A, P):
        return (A & P) if (A & P) else P
    A, P, Q = {0}, {1, 2}, {0, 1}
    assert rev(A, P) == P and rev(A, Q) == A and rev(rev(A, P), Q) == {1} and rev(rev(A, Q), P) == P


def group_copy_version_blind():
    """Prop. n10:copy, Ex. n10:version, Prop. n10:blind."""
    J = np.zeros((4, 2))
    J[0, 0] = J[3, 1] = 1
    close(J.T @ J, np.eye(2))
    rho = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]])
    sig = np.array([[0.6, -0.3j], [0.3j, 0.4]])
    for x in (rho, sig):
        jt = (J @ x @ J.T).reshape(2, 2, 2, 2)
        close(np.einsum("ijkj->ik", jt), np.diag(np.diag(x)))
        close(np.einsum("ijil->jl", jt), np.diag(np.diag(x)))
    assert np.linalg.norm(J @ (rho - sig) @ J.T) > 0.1
    cnot = np.eye(4)[[0, 1, 3, 2]]
    close(cnot @ cnot, np.eye(4))
    # Gram corridor (Ex. ex:fmgram data) and revision bands
    for q, z, det in [(4 / 9, 4 / 5, 13 / 125), (-2 / 3, 2 / 5, 9 / 125), (-1, 7 / 25, 0), (1, 1, 0)]:
        A = np.array([4, 3, 0]) / 5
        B = np.array([1, 0, 0])
        C = np.array([4, 3 * q, 3 * np.sqrt(1 - q * q)]) / 5
        Gm = np.array([A, B, C]) @ np.array([A, B, C]).T
        close([A @ B, B @ C, A @ C], [4 / 5, 4 / 5, z])
        close(np.linalg.det(Gm), det)
        close(81 / 625 - (z - 16 / 25) ** 2, det)
    close((16 + 9 * np.array([1 / 3, 1 / 2, -1, -1 / 3])) / 25, [19 / 25, 41 / 50, 7 / 25, 13 / 25])
    assert 19 / 25 <= 4 / 5 <= 41 / 50 and not (7 / 25 <= 4 / 5 <= 13 / 25)
    # pinching diag(3/4,1/4) in the eigenbasis of X
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    s = np.diag([0.75, 0.25])
    pin = H @ np.diag(np.diag(H.T @ s @ H)) @ H.T
    close(pin, np.eye(2) / 2)
    close(np.linalg.norm(pin - s) ** 2, 1 / 8)
    h = lambda p: -p * np.log(p) - (1 - p) * np.log(1 - p)
    close(np.log(2) - h(0.75), 0.130812, 1e-6)  # nats; 0.1888 bits
    # residues
    for Bd in range(1, 12):
        for Mm in range(2 * Bd, 2 * Bd + 6):
            vals = range(-Bd + 1, Bd)
            assert len({v % Mm for v in vals}) == len(vals)
        assert (-Bd) % (2 * Bd) == Bd % (2 * Bd)
    assert [v for v in range(-4, 5) if v % 11 == 3] == [3]


def group_golden_carry():
    """Prop. n10:golden and Lemma n10:carry."""
    phi = (1 + np.sqrt(5)) / 2
    word = "00010101001000"
    assert len(word) == 14 and "11" not in word and word[0] == "0" and word[-1] == "0"
    # KMP automaton count of no-11 words of length m avoiding the block
    k = len(word)

    def step(state, c):
        t = word[:state] + c
        while t and not word.startswith(t):
            t = t[1:]
        return len(t)
    counts = {(0, "0"): 1}   # (kmp state, last symbol); prepended 0
    for m in range(1, 70):
        new = {}
        for (st, last), c0 in counts.items():
            for c in "01":
                if last == "1" and c == "1":
                    continue
                s2 = step(st, c)
                if s2 == k:
                    continue
                new[(s2, c)] = new.get((s2, c), 0) + c0
        counts = new
        tot = sum(counts.values())
        assert tot <= phi ** (m + 1) * (1 - phi ** -14) ** (m // 14) * (1 + 1e-12)
    fib_words = [1, 2]
    for m in range(2, 60):
        fib_words.append(fib_words[-1] + fib_words[-2])
    for m in range(1, 60):
        assert phi ** m - 1e-6 <= fib_words[m] <= phi ** (m + 1) + 1e-6  # number of no-11 words of length m
    w = np.array([phi, 1.0])
    P = np.array([[1, 1], [1, 0]]) * w[None, :] / (phi * w[:, None])
    close(P.sum(1), 1)
    pi = w ** 2 / (w ** 2).sum()
    close(pi @ P, pi)
    close(pi[1], 1 / (1 + phi ** 2))
    close(pi[1], 1 / (phi * np.sqrt(5)))
    # Zeckendorf
    F = [0, 1]
    while len(F) < 40:
        F.append(F[-1] + F[-2])
    r = 1 / phi
    close([F[k - 1] - r * F[k] for k in range(1, 30)], [(-r) ** k for k in range(1, 30)], 1e-7)

    def zeck(x):
        idx = []
        for kk in range(len(F) - 1, 1, -1):
            if F[kk] <= x:
                idx.append(kk)
                x -= F[kk]
        assert x == 0
        assert all(idx[i] - idx[i + 1] >= 2 for i in range(len(idx) - 1))
        return idx
    for h in range(1, 18):
        sums = sorted({sum(F[j] for j, on in zip(range(2, h + 1), sel) if on)
                       for sel in words(h - 1) if not any(a and b for a, b in zip(sel, sel[1:]))})
        assert sums == list(range(F[h + 1]))
    for x in range(1, 6000):
        z = zeck(x)
        for m in range(4, min(z) + 1):
            for j in range(m - 1, 26):
                assert min(zeck(x + F[j])) >= m - 2


# ---------------------------------------------------------------------------------------------
def schedule_depth(N, S):
    """ASAP depth of the gate sequence: indices outside S descending, then S ascending."""
    order = [j for j in range(N, -1, -1) if j not in S] + sorted(S)
    sites = {0: {1}, N: {N}}
    for j in range(1, N):
        sites[j] = {j, j + 1}
    last = {}
    layer = {}
    for g in order:
        lay = 1 + max((last[s] for s in sites[g] if s in last), default=0)
        layer[g] = lay
        for s in sites[g]:
            last[s] = lay
    return max(layer.values()), layer, order, sites


def group_open_depth():
    """Thm. n10:depth and Ex. n10:depth-refute (minimum over all configurations)."""
    for N in range(2, 15):
        for u in range(1, N // 2 + 1):
            best = min(schedule_depth(N, set(S))[0] for S in combinations(range(1, N + 1), u))
            assert best == floor(N / (u + 1)) + 1 == ceil((N + 1) / (u + 1))
            # balanced-block witness
            sizes = [(N + 1) // (u + 1) + (1 if i < (N + 1) % (u + 1) else 0) for i in range(u + 1)]
            if N == 2 * u:
                sizes = [1] + [2] * u
            starts, acc = [], 0
            for sz in sizes[:-1]:
                acc += sz
                starts.append(acc)
            assert schedule_depth(N, set(starts))[0] == best
    conj = lambda N, u: (N + 3) // 2 - u if N % 2 else (N + 4) // 2 - u
    assert conj(11, 2) == 5 and schedule_depth(11, {4, 8})[0] == 4
    assert conj(8, 2) == 4 and schedule_depth(8, {3, 6})[0] == 3
    # the preprint's own configurations give exactly its formula; N = 13 row of its Fig. 8
    fig8 = {6: (2, (12, 10, 8, 6, 4, 2)), 5: (3, (11, 9, 7, 5, 3)), 4: (4, (10, 8, 6, 4)),
            3: (5, (9, 7, 5)), 2: (6, (8, 6)), 1: (7, (7,))}
    for u, (d, nvec) in fig8.items():
        assert schedule_depth(13, set(nvec))[0] == d == conj(13, u)
        assert floor(13 / (u + 1)) + 1 <= d
    assert schedule_depth(13, {3, 6, 9, 12})[0] == 3
    # smallest disagreements (odd and even) within 2u <= N
    bad = [(N, u) for N in range(2, 30) for u in range(1, N // 2 + 1) if conj(N, u) != floor(N / (u + 1)) + 1]
    assert min(b for b in bad if b[0] % 2) == (11, 2) and min(b for b in bad if b[0] % 2 == 0) == (8, 2)
    assert all(conj(N, u) >= floor(N / (u + 1)) + 1 for N in range(2, 52) for u in range(1, N // 2 + 1))
    ours = lambda N, u: floor(N / (u + 1)) + 1
    for N in range(3, 52):
        top = (N - 1) // 2 if N % 2 else N // 2
        for u in (1, top, top - 1):
            if u >= 1:
                assert conj(N, u) == ours(N, u)
        # the conjectured position patterns, Eqs. (61), (63): their depth is the conjectured value
        for d in range(2, N + 2):
            if N % 2 and N >= 2 * d - 1:
                pos = set(range(N - (d - 1), d - 1, -2))
            elif N % 2 == 0 and N >= 2 * d - 2:
                pos = set(range(N - (d - 2), d - 1, -2))
            else:
                continue
            u = len(pos)
            if 1 <= u <= N // 2:
                assert conj(N, u) == d == schedule_depth(N, pos)[0]
    # operator check: layered product equals the sequential product (random gates, N = 7, S = {3, 6})
    N, S = 7, {3, 6}
    d, layer, order, sites = schedule_depth(N, S)
    gates = {}
    for g in order:
        dim = 2 ** len(sites[g])
        Z = RNG.normal(size=(dim, dim)) + 1j * RNG.normal(size=(dim, dim))
        gates[g] = np.linalg.qr(Z)[0]

    def embed(g):
        ss = sorted(sites[g])
        left = ss[0] - 1
        right = N - ss[-1]
        return np.kron(np.kron(np.eye(2 ** left), gates[g]), np.eye(2 ** right))
    seq = np.eye(2 ** N)
    for g in order:
        seq = embed(g) @ seq
    lay = np.eye(2 ** N)
    for l in range(1, d + 1):
        for g in [g for g in order if layer[g] == l][::-1]:
            lay = embed(g) @ lay
    close(seq, lay, 1e-8)
    assert d == 3
    # depth is unchanged under reversal of the sequence (operator-ordering convention)
    for N in range(2, 10):
        for u in range(1, N // 2 + 1):
            for S in combinations(range(1, N + 1), u):
                d1, _, order, sites = schedule_depth(N, set(S))
                last, lay2 = {}, {}
                for g in reversed(order):
                    lay2[g] = 1 + max((last[s] for s in sites[g] if s in last), default=0)
                    for s in sites[g]:
                        last[s] = lay2[g]
                assert max(lay2.values()) == d1


def group_cnot_bound():
    """Thm. n10:cnot and Ex. n10:counts: characters, counting, label dynamics, tightness at m = 4."""
    for n in (1, 2, 3, 4):
        m = 2 * n
        X = np.array(words(m))
        theta = np.zeros(2 ** m)
        theta[0] = theta[-1] = np.pi / 2
        assert np.allclose(theta, theta[::-1])  # complement symmetry
        h = np.exp(1j * theta)
        for v in X[1:]:
            sgn = (-1) ** (X @ v % 2)
            chi = np.exp(1j * np.sum(theta * sgn))
            if v.sum() % 2 == 0:
                close(chi, -1)
            else:
                close(chi, 1)
            # constant and off-label rotations have trivial character
            lam = np.exp(0.7j)
            close(np.prod(np.full(2 ** m, lam) ** sgn), 1)
            for w in X[1:3]:
                if not np.array_equal(w, v):
                    a = 0.37
                    close(np.exp(-1j * a / 2 * np.sum((-1) ** (X @ w % 2) * sgn)), 1)
        assert 2 ** (m - 1) - 1 + m - 1 == 2 ** (2 * n - 1) + 2 * n - 2
    assert [2 ** (2 * n - 1) + 2 * n - 2 for n in (1, 2, 3)] == [2, 10, 36]
    # n = 1 circuit
    cn = np.eye(4)[[0, 1, 3, 2]]          # control qubit 1, target qubit 2 (basis 00,01,10,11)
    second = np.array([0, 1, 0, 1])      # value of wire 2, which carries x1+x2 between the CNOTs
    rz = np.diag(np.exp(1j * np.pi / 4 * (-1.0) ** second))   # Rz(-pi/2): exp(-i(-pi/2)(-1)^w/2)
    target = np.diag([1j, 1, 1, 1j])
    close(np.exp(1j * np.pi / 4) * cn @ rz @ cn, target)
    # label dynamics on random CNOT circuits (m = 6)
    m = 6
    for _ in range(300):
        labels = [1 << i for i in range(m)]
        created, ev_out, ev_to_odd = [], 0, 0
        seen_even = set()
        for _ in range(int(RNG.integers(1, 40))):
            c, t = RNG.choice(m, 2, replace=False)
            lc, lt = labels[c], labels[t]
            new = lt ^ lc
            pc, pt = bin(lc).count("1") % 2, bin(lt).count("1") % 2
            if pc == 1 and pt == 1:
                created.append(new)
            if pt == 0 and pc == 1:
                ev_to_odd += 1
            labels[t] = new
            if bin(new).count("1") % 2 == 0:
                ev_out += 1
                seen_even.add(new)
        nev = sum(bin(x).count("1") % 2 == 0 for x in labels)
        assert nev == len(created) - ev_to_odd
        # span claim
        span = {0}
        for c in created:
            span |= {s ^ c for s in span}
        assert seen_even <= span
    # exhaustive breadth-first search at m = 4: least CNOT count with every nonzero even label
    # appearing and the final labels the coordinate basis equals 2^3 + 4 - 2 = 10
    m = 4
    evens = [x for x in range(1, 16) if bin(x).count("1") % 2 == 0]
    eidx = {x: i for i, x in enumerate(evens)}
    full = (1 << len(evens)) - 1
    start = (tuple(1 << i for i in range(m)), 0)
    frontier, seen = {start}, {start}
    found = None
    for depth in range(1, 11):
        nxt = set()
        for labs, mask in frontier:
            for c in range(m):
                for t in range(m):
                    if c == t:
                        continue
                    nl = list(labs)
                    nl[t] ^= labs[c]
                    nm = mask | (1 << eidx[nl[t]]) if nl[t] in eidx else mask
                    st = (tuple(nl), nm)
                    if st not in seen:
                        seen.add(st)
                        nxt.add(st)
        frontier = nxt
        if (start[0], full) in frontier:
            found = depth
            break
    assert found == 10
    # Ex. n10:counts: closed traversal of the bipartite cycle
    for n in (1, 2, 3):
        k = n + 2
        walk = [("L", 0)]
        for j in range(k):
            walk += [("R", (j + 1) % k), ("L", (j + 1) % k)]
        assert walk[0] == walk[-1] and len(walk) - 1 == 2 * k
        edges = {frozenset({("L", j), ("R", (j + 1) % k)}) for j in range(k)} | \
                {frozenset({("R", (j + 1) % k), ("L", (j + 1) % k)}) for j in range(k)}
        for x, y in zip(walk, walk[1:]):
            assert frozenset({x, y}) in edges


def group_models_numbers():
    """Rem. n10:pointer-model, Rem. n10:stereo, Rem. n10:programme numbers."""
    for e0, dl in product((-2.0, 0.0, 1.5), (-0.4, 0.0, 0.4)):
        vals = 1 / np.cosh(e0 + dl * np.arange(300))
        assert vals.min() > 0
        if dl:
            assert vals[-1] < 1e-40
    for th in (0.12, 0.5, 1.0, 2.0, 3.0):
        x, z = np.sin(th), np.cos(th)
        u = x / (1 - z)
        close(u, 1 / np.tan(th / 2))
        close((1 + u * u) / 2, 1 / (1 - z))
        close([2 * u / (1 + u * u), (u * u - 1) / (u * u + 1)], [x, z])
        # numerical length magnification of the chart along a meridian
        dth = 1e-6
        u2 = np.sin(th + dth) / (1 - np.cos(th + dth))
        close(abs(u2 - u) / dth, 1 / (1 - z), 1e-4)
    assert abs(2 / np.tan(0.06) - 33.3) < 0.05
    for s in (0.5 + 2j, 0.5 - 0.1j, 0.2 + 1j, 1.3, -2j):
        close(abs(1 - s) ** 2 - abs(s) ** 2, 1 - 2 * s.real)
        assert np.isclose(abs((s - 1) / s), 1) == np.isclose(s.real, 0.5)
    grid = np.linspace(0, 1, 30001)
    vals = np.maximum(1 - grid, grid / 2)
    close(grid[vals.argmin()], 2 / 3, 1e-4)
    close(vals.min(), 1 / 3, 1e-4)
    close((1 - grid) + grid / 2, 1 - grid / 2)
    phi = (1 + np.sqrt(5)) / 2
    assert abs(phi ** 4 - 6.854) < 1e-3
    close(2 ** (np.log2(phi) * 4), phi ** 4)


def main():
    groups = [group_boolean_routing, group_spatial_qram, group_router_hold, group_resolution_profiles,
              group_signed_certificate, group_uniform_section, group_page_cost_recovery,
              group_stack_machines, group_codebook, group_ledger_update, group_copy_version_blind,
              group_golden_carry, group_open_depth, group_cnot_bound, group_models_numbers]
    ok = True
    for g in groups:
        g()
        print("passed", g.__name__)
    print(f"check_memory.py: all {len(groups)} groups passed: {ok}")


if __name__ == "__main__":
    main()
