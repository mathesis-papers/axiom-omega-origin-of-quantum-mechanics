"""One inventory of the inputs of the paper, and the four views generated from it.

Run from the paper's directory:  python3 make_inventory.py
It rewrites, between markers in main.tex, (1) the list of Section `sec:assumptions`,
(2) Table `tab:scope` and Table `tab:falsify` in Section `sec:scope`, and (3) the inventory
Table `tab:audit` in Appendix `app:inventory`.  It checks that every hypothesis paragraph,
conjecture and programme remark of main.tex is in the inventory exactly once, that every
inventory entry is named in the scope table, and that the falsifiability classes are valid.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")

# Falsifiability classes (names chosen so that they cannot be read as hypothesis names).
CLASSES = {
    "math": "mathematical input or consequence; no empirical content of its own",
    "std": "consequence shared with standard physics; no distinguishing test",
    "data": "measured input or fit",
    "test": "could fail against data or against an explicit completion",
    "fails": "fails against data unless a further input is added",
    "open": "programme or interpretive input; no test is specified yet",
}
def cls(c):
    return r"\textsc{%s}" % c

# ---------------------------------------------------------------------------------------------
# The inventory.  kind: reading, postulate, convention, hypothesis, conjecture, programme, problem.
# name: as typeset; key: identifier; part: I-V; defined: LaTeX reference; tokens: regexes for use.
# ---------------------------------------------------------------------------------------------
def H(key, name, part, kind, defined, status, klass, statement, tokens=None, intro=None):
    return dict(key=key, name=name, part=part, kind=kind, defined=defined, status=status, klass=klass,
                statement=statement, tokens=tokens or [], intro=intro)

def br(n):  # tokens for a bracketed or named hypothesis
    e = re.escape(n)
    return [r"\[" + e + r"\]", r"Hypothes[ie]s~?\s*" + e + r"(?![A-Za-z$])", r"hyp:" + e + r"\}"]

FG = r"F$_{\mathrm g}$"; FR = r"F$_{\mathrm r}$"; FHn = r"F$_{\mathrm H}$"; FHF = r"F$_{\mathrm{Hf}}$"
FS = r"F$_{\mathrm S}$"; FSG = r"F$_{\mathrm{Sg}}$"; FSR = r"F$_{\mathrm{Sr}}$"; BL = r"Bil$_\cL$"

INV = [
 # ---------------- Part I: readings and composite inputs
 H("NC", "[NC]", "I", "reading", r"Sec.~\ref{sec:assumptions}", "input", "math",
   r"the noncommutative solution $S^3$ is the one studied", br("NC"),
   r"The noncommutative solution $M\cong S^3$ is the one studied (the commutative one is the real qubit, Proposition~\ref{prop:rebit})."),
 H("L", "[L]", "I", "reading", r"Def.~\ref{def:reading}", "input", "math",
   r"a closed unobservable group $K$, normalised by the left transports", [r"\[L\]"],
   r"A reading of states declares a closed group $K$ of automorphisms unobservable, and the left transports remain symmetries (they normalise $K$) (Definition~\ref{def:reading})."),
 H("W", "[W]", "I", "reading", r"Def.~\ref{def:reading}", "input", "math",
   r"weight of the form $\mathsf F\circ s_K$", [r"\[W\]"],
   r"\emph{Weight form.}  The weight of a reading is $\mathsf F\circ s_K$, $s_K=\sup_Kc$, with $\mathsf F$ continuous; for a clock this is automatic for continuous $\cL$-invariant weights."),
 H("B", "[B]", "I", "reading", r"Thm.~\ref{thm:readings}", "input", "math",
   r"every class has exactly one orthogonal class", [r"\[B\]"],
   r"\emph{Binary states.}  Every class of states has exactly one orthogonal class ($s_K=0$).  Under [L] it is equivalent to [Cl] with [NT] (Proposition~\ref{prop:clockB})."),
 H("Cl", "[Cl]", "I", "reading", r"Sec.~\ref{sec:readings}", r"input; a reformulation under [Ph] (Cor.~\ref{cor:clbridge})", "math",
   r"the unobservable group contains a clock", [r"\[Cl\]"],
   r"\emph{Unobservable clock.}  The unobservable group of an elementary system contains a clock: for a closed system the phase of its own clock is not observable.  An input, not derived from the axiom; under the phase bridge [Ph] it is equivalent to a restricted clause of [Ph] (Corollary~\ref{cor:clbridge})."),
 H("NT", "[NT]", "I", "reading", r"Sec.~\ref{sec:readings}", "input", "math",
   r"some pair of classes is orthogonal", [r"\[NT\]"],
   r"\emph{Non-triviality.}  Some pair of classes is orthogonal."),
 H("T", "[T]", "I", "convention", r"Conv.~\ref{conv:T}", "convention", "math",
   r"complete tests, weights read as probabilities", [r"\[T\]", r"Convention~?\s*T\b", r"conv:T"],
   r"Convention~\ref{conv:T}: a complete test is a maximal family of pairwise orthogonal classes, weights are numbers in $[0,1]$ read as probabilities, and weights of disjoint blocks add."),
 H("N", "[N]", "I", "reading", r"Sec.~\ref{sec:assumptions}", "input", "math",
   r"weights normalised on maximal orthogonal families", [r"\[N\]"],
   r"\emph{Normalisation.}  For every maximal family of pairwise orthogonal classes the weights of any class against the family sum to one."),
 H("Comp", "[Comp]", "I", "reading", r"Def.~\ref{def:reading}", "input", "math",
   r"class correlations and weights of independent systems multiply", [r"\[Comp\]"],
   r"\emph{Composability.}  Class-correlations and weights of independent systems multiply."),
 H("Bil", "[Bil]", "I", "postulate", r"Sec.~\ref{sec:oneclock}", "input (first route to C0)", "math",
   r"products are a unit-valued multilinear map, torus equivariant", [r"\[Bil\]"],
   r"Product configurations of $N$ systems are a unit-valued real-multilinear map into a real inner-product space $\mathcal W$, equivariant for the clock torus; [Bil$_\cL$]: also for the left transports."),
 H("BilL", "["+BL+"]", "I", "postulate", r"Sec.~\ref{sec:oneclock}", "input (form of the product map)", "math",
   r"the product map is also equivariant for the left transports", [r"\[Bil\$_\\cL\$\]"], None),
 H("Span", "[Span]", "I", "postulate", r"Sec.~\ref{sec:oneclock}", "definition", "math",
   r"the composite is spanned by products", [r"\[Span\]"],
   r"$\mathcal W$, and $V_N$, are by definition the span of the product configurations."),
 H("Loc", "[Loc]", "I", "postulate", r"Sec.~\ref{sec:oneclock}", "input", "math",
   r"a composite hides only the clocks of its factors", [r"\[Loc\]"],
   r"The unobservable group of a composite is the group generated by the clocks of its factors."),
 H("O", "[O]", "I", "postulate", r"Sec.~\ref{sec:oneclock}", "input", "math",
   r"the composite weight is symmetric, one on the diagonal, zero exactly at $s_K=0$", [r"\[O\]"],
   r"The weight of a composite is symmetric, as $\kappa$ is, equals one on the diagonal and vanishes exactly where $s_K=0$."),
 H("C0", "C0", "I", "postulate", r"Cor.~\ref{post:C0}", r"theorem for clock-read factors (Thm.~\ref{thm:oneclock}); input on the second route", "std",
   r"one clock: the composite is the complex tensor product", [r"\bC0\b", r"\[C0\]", r"C0--C"],
   r"\emph{One clock.}  A composite is read through one clock, as the complex tensor product.  For clock-read factors a theorem under [Bil], [Loc], [O], [C2], [N] (Theorem~\ref{thm:oneclock}); on a second route C0 and C1(a) are assumed instead (Theorem~\ref{thm:product})."),
 H("C1", "C1", "I", "postulate", r"Post.~\ref{post:C1}", r"(a) a theorem by Thm.~\ref{thm:oneclock}(iv); (b) input", "std",
   r"(a) probabilities of independent tests multiply; (b) each factor keeps its orientation", [r"\bC1\b", r"\[C1\]", r"C0--C[123]", r"C1--C"],
   r"(b) Each factor keeps its orientation; (a) probabilities of independent ray tests multiply (Postulate~\ref{post:C1})."),
 H("C2", "C2", "I", "postulate", r"Post.~\ref{post:C2}", r"input; essential (Prop.~\ref{prop:productsalone})", "std",
   r"states contain products and are closed under round-trip reflections", [r"\bC2\b", r"\[C2\]", r"C0--C[23]", r"C1--C[23]"],
   r"The states of a composite contain the products and are closed under round-trip reflections (Postulate~\ref{post:C2})."),
 H("C3", "C3", "I", "postulate", r"Post.~\ref{post:C3}", r"input; not derivable from C0--C2 (Prop.~\ref{prop:universal})", "std",
   r"local clock-preserving automorphisms and pair exchanges are operations", [r"\bC3\b", r"\[C3\]", r"C0--C3", r"C1--C3"],
   r"Local clock-preserving automorphisms and the pair exchanges are operations (Postulate~\ref{post:C3})."),
 H("Sub", "[Sub]", "I", "postulate", r"Sec.~\ref{sec:otherdim}", "input", "math",
   r"a clock-invariant round-trip-closed sub-structure is treated as a system", [r"\[Sub\]"],
   r"A clock-invariant round-trip-closed sub-structure $S(U)$ of a register is treated as a system: only its configurations are prepared, complete tests are maximal orthogonal families within it, and operations are those of the host that preserve $U$."),
 H("Gle", r"$w\ge0$, $w=f(\beta_N)$, (a)--(c)", "I", "hypothesis", r"Thm.~\ref{thm:bornN}(iii)", "input (Gleason-type uniqueness)", "math",
   r"hypotheses of the uniqueness of the composite Born weight", [r"thm:bornN\}\(iii\)", r"w=f\(\\beta"],
   None),
 # ---------------- Part II
 H("Ph", "[Ph]", "II", "hypothesis", r"Sec.~\ref{hyp:Ph}", "hypothesis; [Cl] is a reformulation under it", "math",
   r"complex composite reading built on one clock; (b) restricted to transports commuting with $T$", br("Ph") + [r"Ph\((a|b)\)"]),
 H("U", "U", "II", "hypothesis", r"Sec.~\ref{hyp:U}", "physical reading only; no statement uses it", "open",
   r"physical specification of a closed-universe state", br("U")),
 H("A", "A", "II", "hypothesis", r"Sec.~\ref{hyp:A}", "physical reading", "open",
   r"an observer's accessible algebra is a tensor factor", br("A")),
 H("Gw", "Gw", "II", "hypothesis", r"Sec.~\ref{hyp:Gw}", "hypothesis", "math",
   r"a finite Gram window of stored comparisons", br("Gw")),
 H("Ue", "Ue", "II", "hypothesis", r"Sec.~\ref{hyp:Ue}", "hypothesis", "open",
   r"entropy drop of a reset equals the log area ratio of cap corridors", br("Ue")),
 H("Ca", "Ca", "II", "hypothesis", r"Sec.~\ref{hyp:Ca}", "hypothesis", "math",
   r"causal record access in tensor-factor form", br("Ca")),
 H("Oq", "Oq", "II", "hypothesis", r"Sec.~\ref{hyp:Oq}", "physical reading", "open",
   r"observer memory as a static QRAM read by coherent queries, router environment inaccessible", br("Oq")),
 H("Pa", "Pa", "II", "hypothesis", r"Sec.~\ref{hyp:Pa}", "physical reading", "open",
   r"memory access by a finite pointer with versioned translation cache and page walk", br("Pa")),
 H("Ou", "Ou", "II", "hypothesis", r"Rem.~\ref{ob:observermodel}", r"interpretive; Thm.~\ref{ob:locking} does not use it", "open",
   r"synchronous reciprocal observer updates by specified maps", br("Ou")),
 H("Pl", "Pl", "II", "hypothesis", r"Rem.~\ref{ob:learninghypothesis}", "hypothesis", "open",
   r"a predictive task fixed in advance of training", br("Pl")),
 H("Lp", "Lp", "II", "hypothesis", r"Rem.~\ref{ob:past}", "hypothesis", "open",
   r"a common low-entropy past", br("Lp")),
 H("R", "[R]", "II", "hypothesis", r"Def.~\ref{def:soprocess}", "hypothesis (record process)", "std",
   r"a record process with one terminal vertex", [r"\[R\]", r"def:soprocess"]),
 H("Tr", "Tr", "II", "hypothesis", r"Sec.~\ref{hyp:Tr}", "hypothesis", "math",
   r"a linear number of terminal transcripts in the tick count", br("Tr")),
 H("Ap", "Ap", "II", "hypothesis", r"Sec.~\ref{hyp:Ap}", "hypothesis", "test",
   r"preparation-independent, unitary-invariant apparatus sampling", br("Ap")),
 # ---------------- Part III
 H("Cd", "Cd", "III", "hypothesis", r"Sec.~\ref{hyp:Cd}", "hypothesis", "std",
   r"calibrated dynamics: Hamiltonian, time scale and projector given", br("Cd")),
 H("Bs", "Bs", "III", "hypothesis", r"Prop.~\ref{hyp:Bs}", "hypothesis", "std",
   r"common preparation, common speed cap, sequential schedule", br("Bs") + [r"Hypothesis Bs"]),
 H("Ac", "Ac", "III", "hypothesis", r"Sec.~\ref{hyp:Ac}", "hypothesis", "std",
   r"access to a graph resource with local tests and feedforward", br("Ac")),
 H("Lq", "Lq", "III", "hypothesis", r"Sec.~\ref{hyp:Lq}", "hypothesis and programme", "open",
   r"local indistinguishability for a fusion space", br("Lq")),
 # ---------------- Part IV
 H("E", "E", "IV", "hypothesis", r"Sec.~\ref{hyp:E}", "hypothesis", "math",
   r"events are the dichotomies of an elementary structure, causal order is inclusion", br("E"),
   None),
 H("F", "F", "IV", "hypothesis", r"Sec.~\ref{hyp:F}", "hypothesis", "std",
   r"fields of structures with a clock field, connections and an invariant kinetic form", br("F") + [r"\bF\((i|ii|iii|iv)\)", r"\[F\]\((i|ii|iii|iv)"]),
 H("Fg", "["+FG+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:Fg}", "hypothesis", "fails",
   r"(a) dynamical connection; (b) cap geometry with a flat vacuum; with F alone: no scalar", [re.escape(FG), r"hyp:Fg\}"]),
 H("Fr", "["+FR+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:Fr}", "hypothesis", "std",
   r"variable radius: values in $V\setminus\{0\}$ with the renormalisable action", [re.escape(FR), r"hyp:Fr\}"]),
 H("FH", "["+FHn+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:FH}", "hypothesis", "test",
   r"round $\HH P^1$ target of radius $f$; not selected by the pair", [re.escape(FHn), r"hyp:FH\}"]),
 H("FHf", "["+FHF+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:FHf}", "hypothesis", "test",
   r"fermion embeddings of the round target", [re.escape(FHF), r"hyp:FHf\}"]),
 H("FS", "["+FS+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:FS}", "hypothesis", "math",
   r"a Skyrme (area-square) term with coefficient $1/(32e_{\mathrm S}^2)$", [re.escape(FS), r"hyp:FS\}"]),
 H("FSg", "["+FSG+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:FSg}", "hypothesis", "math",
   r"covariant area square with a dynamical connection", [re.escape(FSG), r"hyp:FSg\}"]),
 H("FSr", "["+FSR+"]", "IV", "hypothesis", r"Sec.~\ref{hyp:FSr}", "hypothesis", "math",
   r"area square with the radial mode", [re.escape(FSR), r"hyp:FSr\}"]),
 H("P", "P", "IV", "hypothesis", r"Sec.~\ref{hyp:P}", r"hypothesis; electroweak scale $\approx14.6\,\mathrm{TeV}/e_{\mathrm S}$", "test",
   r"degree textures read as baryon-like (electroweak-scale) excitations", br("P")),
 H("G", "G", "IV", "hypothesis", r"Sec.~\ref{hyp:G}", "hypothesis", "std",
   r"Pati--Salam group, multiplets, breaking, equal couplings (i)--(v)", br("G") + [r"\bG\((i|ii|iii|iv|v)\)", r"\[G\]\((i|v)"]),
 H("GR", "GR", "IV", "hypothesis", r"Sec.~\ref{hyp:GR}", "hypothesis", "std",
   r"Einstein's equations with a source", br("GR")),
 H("Br", "Br", "IV", "hypothesis", r"Sec.~\ref{hyp:Br}", "hypothesis (group, counting and flux statements derived)", "std",
   r"five symmetry-breaking geometrisations: Cartan spacetime, electroweak, chiral, crystal and superconductor, observer bases",
   br("Br")),
 H("Oc", "Oc", "IV", "hypothesis", r"Sec.~\ref{hyp:Oc}", "model hypothesis; inter-level map not specified", "open",
   r"exterior-degree recursion between levels, particle--field readings, Standard Model degree assignments",
   br("Oc")),
 H("Tw", "Tw", "IV", "hypothesis", r"Sec.~\ref{hyp:Tw}", "model hypothesis; closure $L_5\to L_0$ not supplied", "open",
   r"six-level concept tower $L_0$--$L_5$ with a proposed return through the spinor double cover",
   br("Tw")),
 H("Lr", "Lr", "IV", "hypothesis", r"Sec.~\ref{hyp:Lr}", "model hypothesis", "open",
   r"the sagitta action as a calibrated physical action and as a model of inference",
   br("Lr")),
 H("KK", "KK", "IV", "hypothesis", r"Sec.~\ref{hyp:KK}", "hypothesis", "math",
   r"a fixed connection metric on the total space", br("KK")),
 H("Sn", "Sn", "IV", "hypothesis", r"Sec.~\ref{hyp:Sn}", "hypothesis", "math",
   r"a single fermion trace in the Dirac heat coefficient", br("Sn")),
 H("Om", "Om", "IV", "hypothesis", r"Sec.~\ref{hyp:Om}", "hypothesis", "data",
   r"one-loop matching with a specified (direct-breaking) spectrum", br("Om")),
 H("Ps", "Ps", "IV", "hypothesis", r"Sec.~\ref{hyp:Ps}", "hypothesis", "data",
   r"intermediate Pati--Salam spectrum with step matching", br("Ps")),
 H("Cc", "Cc", "IV", "hypothesis", r"Sec.~\ref{hyp:Cc}", "programme hypothesis", "open",
   r"a comparison action with a clock", br("Cc")),
 H("Ea", "Ea", "IV", "hypothesis", r"Sec.~\ref{hyp:Ea}", r"programme hypothesis (input of Rem.~\ref{ub:arclimits})", "open",
   r"an effective action for comparison fields", br("Ea")),
 H("Sb", "Sb", "IV", "hypothesis", r"Sec.~\ref{hyp:Sb}", r"programme hypothesis (Rem.~\ref{tr:programme}); no theorem uses it", "open",
   r"separate colour, weak and hypercharge bundles, a scalar bridge", br("Sb")),
 H("Ix", "Ix", "IV", "hypothesis", r"Sec.~\ref{hyp:Ix}", "hypothesis (charge three is an input)", "math",
   r"an internal spin space with a twist of charge three", br("Ix")),
 H("Zw", "Zw", "IV", "hypothesis", r"Sec.~\ref{hyp:Zw}", "hypothesis", "data",
   r"reading of the invisible $Z$ width as a generation count", br("Zw")),
 H("Eq", "Eq", "IV", "hypothesis", r"Sec.~\ref{hyp:Eq}", "hypothesis (Jacobson's inputs)", "open",
   r"local continuum entanglement equilibrium", br("Eq")),
 H("Ck", "Ck", "IV", "hypothesis", r"Sec.~\ref{hyp:Ck}", r"model identification (Rem.~\ref{ub:clock})", "open",
   r"laboratory velocity read as the Bloch vector; saturated clock budget", br("Ck")),
 H("Gl", "Gl", "IV", "programme", r"Sec.~\ref{hyp:Gl}", "programme: the lapse is inserted by hand", "open",
   r"static information load (Schwarzschild lapse)", br("Gl")),
 # ---------------- Part V
 H("Hw", "Hw", "V", "hypothesis", r"Sec.~\ref{hyp:Hw}", "hypothesis", "std",
   r"a calibrated resonant fixed-axis drive with an ideal frame register", br("Hw")),
]
# conjectures, open problem from a conjecture, programmes
CONJ = [
 ("conj:cueth", "local equilibration in the singlet sector", "II"),
 ("conj:fmcontrol", "small-phase control in a cap corridor", "II"),
 ("ob:prefixconjecture", "five steps throughout the real admissible interval", "II"),
 ("conj:hbmicro", "microscopic matching of the pair scalar", "IV"),
 ("conj:metastable", "weak-gauging metastability of textures", "IV"),
 ("ub:uniform", "entropy gap with increasing bond dimensions", "IV"),
 ("gn:conjecture", "a comparison-determined index", "IV"),
 ("cr:conjecture", "a comparison regulator for the trace", "IV"),
 ("n12:inference-conj", "lower sagitta cost and intermediate factual accuracy (no empirical estimate)", "IV"),
]
PROB = [("conj:somixing", "invariant sampling from apparatus dynamics", "II"),
 ("n12:gravproblem", "gravitational dynamics from comparison costs", "IV"),
]
PROG = [
 ("rem:cuprogramme", "thermal time, area law, arrow, cosmology", "II"),
 ("rem:fmprogramme", "dynamical records", "II"),
 ("n10:programme", "lookup resource hierarchy for observer memory", "II"),
 ("ob:controlprogramme", "observer control and learning", "II"),
 ("ob:thermalprogramme", "temperature, resetting and observer dynamics", "II"),
 ("rem:soprogramme", "physical origin of the record law", "II"),
 ("rem:bdprogramme", "physical clock identification", "III"),
 ("cp:compprogramme", "conditional dynamics and circuit examples", "III"),
 ("cp:recoveryprogramme", "recovery and physical fault tolerance", "III"),
 ("cp:anyonprogramme", "finite-size and dynamical requirements of protected sectors", "III"),
 ("rem:hbtests", "microscopic origin and matching of the round target", "IV"),
 ("rem:skprogramme", "particle identification of textures", "IV"),
 ("tr:programme", "comparison-field hypotheses for triality", "IV"),
 ("gn:programme", "a selection mechanism for generations", "IV"),
 ("cr:programme", "continuum and matching of coupling ratios", "IV"),
 ("n12:inference-programme", "scale recursion, clock readout and inference tests", "IV"),
 ("ub:programme", "the cap dynamics problem (continuum gravity)", "IV"),
 ("ob:fluidprogramme", "fluid closure under restricted observation", "App."),
 ("n11:programme", "records in spacetime, timeless clocks and evaporation", "App."),
]
for lab, title, part in CONJ:
    INV.append(H(lab, r"Conj.~\ref{%s}" % lab, part, "conjecture", r"Conj.~\ref{%s}" % lab, "conjecture", "math", title,
                 [re.escape(r"\ref{%s}" % lab)]))
for lab, title, part in PROB:
    INV.append(H(lab, r"Problem~\ref{%s}" % lab, part, "problem", r"Problem~\ref{%s}" % lab, "open problem", "open", title,
                 [re.escape(r"\ref{%s}" % lab)]))
for lab, title, part in PROG:
    INV.append(H(lab, r"Rem.~\ref{%s}" % lab, part, "programme", r"Rem.~\ref{%s}" % lab, "programme", "open", title,
                 [re.escape(r"\ref{%s}" % lab)]))

# ---------------------------------------------------------------------------------------------
# Scope table: statements by topic, with status and the inventory keys they involve.
# ---------------------------------------------------------------------------------------------
SCOPE = [
 (r"Carrier $S^1$ or $S^3$; kernel; round-trip law $\Leftrightarrow$ two-readout determinacy; linear space $\R^4$; clocks are the complex structures commuting with $\cL$", r"theorem (Paper I; Thm.~\ref{thm:trd})", ["NC"]),
 (r"States are clock orbits; weight $\beta$; real and quaternionic readings", r"theorem under [L], [B] or [L], [Cl], [NT] (Prop.~\ref{prop:clockB}); exponent $2$ with [W], [Comp], [N] (Thm.~\ref{thm:readings})", ["L", "W", "B", "Cl", "NT", "N", "Comp", "T"]),
 (r"[Cl] and the phase bridge", r"reformulation under [Ph] (Cor.~\ref{cor:clbridge}); unrestricted Ph(b) contradicts [NT] (Rem.~\ref{rem:clreal})", ["Ph", "Cl"]),
 (r"Born weight, symmetries, $\Theta^2=-1$, Schr\"odinger evolution, uncertainty equality (one system)", r"theorem; probability reading [T]; $H$ free", ["T"]),
 (r"One clock, tensor product, entanglement, Born weights and instruments for composites", r"theorem for clock-read factors (Thm.~\ref{thm:oneclock}); second route Thm.~\ref{thm:product}; uniqueness by Gleason (Thm.~\ref{thm:bornN})", ["Bil", "BilL", "Span", "Loc", "O", "C0", "C1", "C2", "C3", "Gle"]),
 (r"Sub-structures of every dimension; exchange sector", r"theorem (Prop.~\ref{prop:rtclosed}, Cor.~\ref{cor:substructure}); as systems under [Sub]", ["Sub"]),
 (r"Closed register: singlets, conditional dynamics, finite modular flow and Gibbs criterion", r"theorem under C0--C3, [Span], [T]; readings U, A; equilibration conjecture; programme", ["U", "A", "conj:cueth", "rem:cuprogramme"]),
 (r"Finite memory: Gram windows, reset heat with side information, eraser, gluing, barycentre", r"theorem; window Gw, encoding Ue, access Ca; control conjecture; programme", ["Gw", "Ue", "Ca", "conj:fmcontrol", "rem:fmprogramme"]),
 (r"Observer memory as hardware: QRAM hold bound, deepest-router resolution, signed Gram certificates; page walk, lookup and ledger bounds; open-boundary depth $\lfloor N/(\upsilon+1)\rfloor+1$; CNOT bound $2^{2n-1}+2n-2$",
  r"theorem (Thms.~\ref{n10:hold}, \ref{n10:profiles}, \ref{n10:depth}, \ref{n10:cnot}); conjectured depth formulas refuted (Ex.~\ref{n10:depth-refute}); readings Oq, Pa; programme",
  ["Oq", "Pa", "n10:programme"]),
 (r"Observers: locking iff $\operatorname{spr}(AB)<1$ (uniqueness alone does not suffice), noise, records, prediction, horizons", r"theorem (Thm.~\ref{ob:locking}); readings Ou, Pl, Lp; prefix conjecture; programmes", ["Ou", "Pl", "Lp", "ob:prefixconjecture", "ob:controlprogramme", "ob:thermalprogramme"]),
 (r"Single outcomes: Born at every stage $\Leftrightarrow$ martingale; height-threshold and exponential-race rules Born $\Leftrightarrow$ invariant law; inverse-tick bound", r"theorem under [R] (Thms.~\ref{thm:sorigidity}, \ref{thm:sosphere}, \ref{thm:sosimplex}); [Tr]; apparatus law Ap and Problem~\ref{conj:somixing}; programme", ["R", "Tr", "Ap", "conj:somixing", "rem:soprogramme"]),
 (r"Speed limits with sharp constants; Tsirelson bound; monogamy budget (CKW, Osborne--Verstraete)", r"theorem (Props.~\ref{prop:bdmt}, Thms.~\ref{thm:bdchsh}, \ref{thm:bdbudget}); time bounds under Cd, Bs; programme", ["Cd", "Bs", "rem:bdprogramme"]),
 (r"Codes and measurement-based computation: compressed Knill--Laflamme, five-qubit minimum, gflow, parameter shift", r"theorem under C0--C3, [T], [Sub]; access Ac; protected sectors Lq; programmes", ["Ac", "Lq", "cp:compprogramme", "cp:recoveryprogramme", "cp:anyonprogramme"]),
 (r"Divergences, barycentres, magic; statements cited from a Lean library", r"theorem; cited (unrefereed) where so marked", []),
 (r"$dS_4$ of dichotomies, causal order, static patches, invariant metrics", r"classical; theorem; physical reading under E", ["E"]),
 (r"Ungauged centre field as a source: no maximal solution carries the cap order; $\epsilon_G=4\pi G_Nv^2\approx5\times10^{-33}$ at $v=246$\,GeV", r"theorem under E, F, GR (Props.~\ref{prop:nullobstruction}, \ref{prop:friedmann}, Rem.~\ref{rem:friedmannsize})", ["E", "F", "GR"]),
 (r"$\U(2)$, mass-form ratio, $m_W=m_Z\cos\theta_W$; no scalar", r"theorem under F; masses under " + FG + r"(a); contradicted by the observed scalar", ["F", "Fg"]),
 (r"Scalar completions: flat radial mode; round $S^4$ with $c_V=\sqrt{1-\xi}$, $c_{2V}=1-2\xi$", r"theorem under " + FR + r" or " + FHn + r" (Yukawas " + FHF + r"); microscopic origin conjecture; programme", ["Fr", "FH", "FHf", "conj:hbmicro", "rem:hbtests"]),
 (r"Skyrme term: $E\ge6\pi^2(v/e_{\mathrm S})\lvert\mathcal Q\rvert$, Hopf charge, spin $\tfrac12$ allowed", r"theorem under " + FS + r" (" + FSG + r", " + FSR + r"); existence cited; metastability conjecture; particle reading P", ["FS", "FSg", "FSr", "P", "conj:metastable", "rem:skprogramme"]),
 (r"Pati--Salam group, global form, hypercharges; $\sin^2\theta_W=\tfrac38$", r"theorem under G(i)--(iv); G(v) not realised by one-loop running", ["G"]),
 (r"Coset split: structure equations, Pythagorean comparison budget, Noether current; winding and degree sectors",
  r"theorem (Thm.~\ref{n12:coset}, Ex.~\ref{n12:sectors}); CCWZ cited; stabiliser fraction as current refuted (Rem.~\ref{n12:ref-current}); models under Br",
  ["Br"]),
 (r"Exterior degree $1,3,3,1$, $I_3^2=-1$, paravector forms $t^2\mp\lvert x\rvert^2$; point sources and Fourier readouts",
  r"theorem (Props.~\ref{n12:exterior}, \ref{n12:green}); signature and clock are inputs (Rem.~\ref{n12:ref-lorentz}); recursion under Oc; programme",
  ["Oc", "n12:inference-programme"]),
 (r"Gate commutator against triangle area ($\pi/3$ against $\pi/2$ at a quarter turn); entangling pairs; orientation parity",
  r"theorem (Props.~\ref{n12:gates}, \ref{n12:parity}); refuted readings (Rems.~\ref{n12:ref-area}, \ref{n12:ref-entangle}, \ref{n12:ref-legs}); concept tower Tw",
  ["Tw"]),
 (r"Composition of interaction signs: four counterexamples to additive cell sources; spin-two universality; force ratios",
  r"theorem (Rem.~\ref{n12:ref-additive}); cited (Rem.~\ref{n12:spin-two}); open problem",
  ["n12:gravproblem"]),
 (r"Sagitta action, Born-surprisal series, Jensen bound, free fixed-endpoint minimisers, Choi Fisher metric",
  r"theorem (Thm.~\ref{n12:min}, Props.~\ref{n12:born}, \ref{n12:fisher}); inference model Lr; conjecture without estimate",
  ["Lr", "n12:inference-conj"]),
 (r"Coupling ratios: Kaluza--Klein and heat-kernel normalisations, radius tests", r"theorem under KK, Sn; selection by the structure refuted (Rem.~\ref{rem:crstatus}); regulator conjecture; programmes", ["KK", "Sn", "Cc", "cr:conjecture", "cr:programme"]),
 (r"One-loop running: inputs, two-scale fit $M_R=1.29\times10^{10}$, $M_{\mathrm{PS}}=9.71\times10^{16}$\,GeV, direct-breaking fit", r"fit with zero predictions under Ps, Om (Thm.~\ref{ub:fit}, Prop.~\ref{cr:twostage}); one loop", ["Ps", "Om", "Ea"]),
 (r"Triality: three eight-dimensional modules; obstruction for gauge actions factoring through $\mathrm{Spin}(8)$", r"theorem (Thm.~\ref{tr:main}, Prop.~\ref{tr:obstruction}); larger groups open; programme hypothesis Sb", ["Sb", "tr:programme"]),
 (r"Generations: index gives three with charge three; other mechanisms refuted as selectors", r"theorem under Ix (Cor.~\ref{gn:conditional}); Zw for the $Z$ width; index conjecture; programme", ["Ix", "Zw", "gn:conjecture", "gn:programme"]),
 (r"Gravity: cut bound, cap first law, Jacobson's conditional linearised Einstein equation", r"theorem (Thms.~\ref{ub:cut}, \ref{ub:caplaw}); restated under Eq (Thm.~\ref{ub:einstein}); tensor-network results cited (App.~\ref{app:tensor}); gap conjecture; programme", ["Eq", "ub:uniform", "ub:programme"]),
 (r"Clock rates on the velocity ball; Schwarzschild lapse", r"model identification under Ck (Rem.~\ref{ub:clock}); Gl programme", ["Ck", "Gl"]),
 (r"Examples: frame updates and virtual $Z$ gates", r"under Hw (Rem.~\ref{rem:exframe})", ["Hw"]),
 (r"Further worked material: fluid closure, recurrences and arithmetic norms", r"proved in place or cited; outside the main line (Apps.~\ref{app:fluid}, \ref{app:arith})", ["ob:fluidprogramme"]),
 (r"Further independence models (Table~\ref{tab:n11witness}); Axiom A$_0$ and Axiom $\Omega$; the effect route to $\beta$; L\"uders branch",
  r"models and theorems (Sec.~\ref{sec:n11}); Busch's theorem cited (Prop.~\ref{n11:busch})",
  ["NC", "Cl", "Comp", "N", "Span", "Sub", "C1", "Cd", "E", "F"]),
 (r"Third-order interference vanishes; collision entropy and mutually unbiased bases",
  r"theorem (Prop.~\ref{n11:sorkin}, Thm.~\ref{n11:entropy}); antecedents cited", []),
 (r"Comparison with interpretations: operational content proved, remainder open",
  r"theorems as cited in Table~\ref{tab:n11interp}; Ap and Problem~\ref{conj:somixing}; programmes",
  ["n11:programme", "rem:soprogramme", "Ap", "conj:somixing", "R", "T"]),
 (r"Number of structures and generations, scales, masses, $g'/g$, unit of length, $G_Nv^2$", r"not derived; open", []),
]

FALSIFY = [
 (r"Kinematics: one system, composites, sub-structures", "std", r"reproduces standard quantum mechanics; a measured $\lvert S\rvert>2\sqrt2$ would contradict C0--C2"),
 (r"Events: four dimensions, Lorentzian, causal order by inclusion", "math", r"under E; no direct empirical test of nesting as causal order"),
 (r"$\Lambda>0$, $\Lambda\ell^2=3$", "data", r"for the invariant metric; no length calibration, hence no number"),
 (r"Ungauged centre field as a source", "std", r"under F, GR; at $v=246$\,GeV the factor $1-4\pi G_Nv^2$ differs from one by $5\times10^{-33}$ and is unobservable; a test would need a separate texture scale"),
 (r"Electroweak sector under F, " + FG + r"(a): no scalar", "fails", r"contradicted by the observed scalar ($125.13\pm0.11$\,GeV \cite{ex-PDG26}); one of the two completions below is needed"),
 (r"Flat radial mode " + FR, "std", r"$\rho=1$, $c_V=c_{2V}=1$ at tree level; $m_h$ fitted"),
 (r"Round $S^4$ target " + FHn, "test", r"$c_{2V}=2c_V^2-1$, $c_V\le1$; observed bound $\xi\le0.039$ ($f\ge1.24$\,TeV), expected $\xi\le0.084$ ($f\ge0.85$\,TeV), separate Gaussians; compare \cite{rf:ATLAS2015}"),
 (r"Textures under " + FS + r" and Hypothesis~P", "test", r"electroweak-scale (techni-)skyrmions of mass $\gtrsim14.6\,\mathrm{TeV}/e_{\mathrm S}$ once $e_{\mathrm S}$ and the quantisation are fixed"),
 (r"Pati--Salam group and hypercharges (G(i)--(iv))", "std", r"agree with the Standard Model; no breaking scale supplied"),
 (r"$\sin^2\theta_W=\tfrac38$ (G(v)) as a boundary value", "test", r"fails with Standard Model running and no thresholds (Remark~\ref{rem:scales})"),
 (r"Coset currents, Goldstone counts and topological sectors (Br)", "std",
  r"standard effective field theory and homotopy; no distinguishing statement"),
 (r"Exterior-degree recursion (Oc) and concept tower (Tw)", "open",
  r"interpretive; no inter-level map or closure map is specified"),
 (r"Sagitta inference conjecture (Lr)", "test",
  r"a matched comparison on held-out tasks with an external truth criterion; no estimate exists yet"),
 (r"Two-scale Pati--Salam fit (Ps)", "data", r"two scales fitted to two differences; a test only if $M_R$ or $M_{\mathrm{PS}}$ is fixed beforehand (proton decay, seesaw)"),
 (r"Coupling ratios from the total space (KK, Sn)", "math", r"selection by the structure refuted; $g'/g$ open"),
 (r"Triality count", "math", r"the count three and the obstruction; generations not obtained"),
 (r"Generation count", "data", r"an input; $N_\nu=2.9840\pm0.0082$ and $2.9963\pm0.0074$ count light $Z$-coupled neutrinos only"),
 (r"Gravity from cap entanglement", "open", r"conditional on Eq; no test until Eq is derived"),
 (r"Clock rates (Ck) and the lapse (Gl)", "open", r"a model identification and a programme"),
 (r"Observers, records, horizons", "math", r"locking, channel and record statements; no consensus of estimates in general (Example~\ref{ob:cosine}); no temperature from a finite region (Example~\ref{ob:curvature})"),
 (r"Codes and computation", "std", r"standard quantum information; nothing distinguishing"),
 (r"Bounds", "std", r"statements of standard quantum mechanics (Mandelstam--Tamm, Margolus--Levitin, Tsirelson, CKW)"),
 (r"Finite memory", "std", r"the reset bound is the Landauer--Reeb--Wolf balance with side information"),
 (r"Observer memory as hardware", "math",
  r"finite routing, lookup, ledger and circuit statements; the readings Oq and Pa have no test until a decoder, a layout and a charged protocol are specified (Rem.~\ref{n10:programme})"),
 (r"Third-order interference (Prop.~\ref{n11:sorkin})", "std",
  r"a nonzero third-order term with one additive amplitude assignment and one calibration across the seven settings would contradict the composite Born weights (Thm.~\ref{thm:bornN})"),
 (r"Interpretations (Table~\ref{tab:n11readings})", "open",
  r"the five readings share the operational content proved here; recoherence (Example~\ref{ex:sorecoherence}) tests objective collapse, not these readings"),
 (r"Single outcomes", "std", r"Born statistics as in standard quantum mechanics; an objective-collapse completion would be \textsc{test} through recoherence (Example~\ref{ex:sorecoherence})"),
 (r"Closed register and internal clocks", "math", r"Conjecture~\ref{conj:cueth} can be refuted by computation; the conditional dynamics is Page--Wootters"),
]

# ---------------------------------------------------------------------------------------------
def tex_escape_check(s):
    return s

def build(src):
    lines = src.split("\n")
    # section map
    secs = []
    for i, l in enumerate(lines):
        m = re.match(r"\\section\{.*\\label\{([^}]*)\}", l)
        if m:
            secs.append((i, m.group(1)))
    def sec_of(i):
        lab = None
        for j, s in secs:
            if j <= i:
                lab = s
        return lab
    def first(pat):
        h = [i for i, l in enumerate(lines) if pat in l]
        return h[0]
    excl = [(0, first(r"\label{sec:axiom}")),
            (first(r"\label{sec:closed}"), first(r"\subsection{Conventions and tags}")),
            (first(r"\label{sec:scope}"), first(r"\label{sec:crqo}"))]
    def excluded(i):
        return any(a <= i < b for a, b in excl)
    # where used
    for h in INV:
        used = []
        pats = [re.compile(t) for t in h["tokens"]]
        for i, l in enumerate(lines):
            if excluded(i) or "%" == l[:1]:
                continue
            if any(p.search(l) for p in pats):
                s = sec_of(i)
                if s and s not in used:
                    used.append(s)
        h["used"] = used
    return lines

def refs(labs):
    if not labs:
        return "---"
    return ", ".join(r"\ref{%s}" % l for l in labs)

def chunked(rows, colspec, header, caption, label, chunk):
    out = []
    for k in range(0, len(rows), chunk):
        cap = caption if k == 0 else r"(continued)"
        out += [r"\begin{table}[p]", r"\caption{%s}" % cap]
        if k == 0:
            out.append(r"\label{%s}" % label)
        out += [r"\footnotesize", r"\begin{tabular}{%s}" % colspec, r"\toprule", header + r"\\", r"\midrule"]
        out += rows[k:k + chunk]
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)

def inventory_table():
    rows = []
    part = None
    titles = {"I": "Part I: readings and composite inputs", "II": "Part II", "III": "Part III", "IV": "Part IV", "V": "Part V", "App.": "Appendices"}
    for h in INV:
        if h["part"] != part:
            part = h["part"]
            rows.append(r"\multicolumn{6}{@{}l}{\emph{%s}}\\" % titles[part])
        rows.append(r"{%s} & %s & %s & %s & %s & %s\\" % (h["name"], h["statement"], h["defined"], refs(h["used"]), h["status"], cls(h["klass"])))
    return chunked(rows, r"@{}p{0.11\textwidth}p{0.29\textwidth}p{0.11\textwidth}p{0.17\textwidth}p{0.2\textwidth}p{0.06\textwidth}@{}",
                   r"Name & Content & Defined & Used in (sections) & Status & Class",
                   r"Inventory of the inputs beyond Axiom~$\Omega$ (generated by \texttt{make\_inventory.py}).  ``Used in'' lists the sections, outside the introduction and the status sections, that name the entry.",
                   "tab:audit", 24)

def scope_table():
    name = {h["key"]: h["name"] for h in INV}
    rows = [r"{%s} & %s & %s\\" % (stmt, status, ", ".join(name[k] for k in keys) if keys else "---") for stmt, status, keys in SCOPE]
    return chunked(rows, r"@{}p{0.42\textwidth}p{0.36\textwidth}p{0.18\textwidth}@{}", r"Statement & Status & Inventory entries",
                   r"Status of the main statements (generated from the inventory, Table~\ref{tab:audit}).  The examples that instantiate them are tagged in Section~\ref{sec:examples}.",
                   "tab:scope", 15)

def falsify_table():
    out = [r"\begin{table}[h]",
           r"\caption{What could fail (generated from the inventory).  Classes: " + "; ".join(r"%s %s" % (cls(c), d) for c, d in CLASSES.items()) + r".  The class of each inventory entry is in Table~\ref{tab:audit}.}",
           r"\label{tab:falsify}", r"\footnotesize",
           r"\begin{tabular}{@{}p{0.3\textwidth}p{0.08\textwidth}p{0.57\textwidth}@{}}", r"\toprule",
           r"Statement & Class & Status\\", r"\midrule"]
    for stmt, c, txt in FALSIFY:
        out.append(r"{%s} & %s & %s\\" % (stmt, cls(c), txt))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)

def assumptions_list():
    out = [r"\begin{itemize}[leftmargin=1.6em]", r"\item[$(\Omega)$] Axiom $\Omega$: $(\Omega1)$, $(\Omega2\mathrm{a{-}d})$, $(\Omega3)$ (Definition~\ref{def:cs})."]
    for h in INV:
        if h["part"] == "I" and h["intro"]:
            out.append(r"\item[{%s}] %s" % (h["name"], h["intro"]))
    out.append(r"\end{itemize}")
    later = {}
    for h in INV:
        if h["part"] != "I" and h["kind"] in ("hypothesis", "programme") and not h["name"].startswith("Rem."):
            later.setdefault(h["part"], []).append(h["name"].strip("[]") if h["name"].startswith("[") and h["name"].endswith("]") else h["name"])
    parts = []
    for p, title in (("II", "Part~II"), ("III", "Part~III"), ("IV", "Part~IV"), ("V", "Part~V")):
        if p in later:
            parts.append(r"%s: %s" % (title, ", ".join(later[p])))
    out.append(r"\noindent That states are clock orbits is a theorem under [L], [B] (Theorem~\ref{thm:readings}), equivalently under [L], [Cl], [NT] (Proposition~\ref{prop:clockB}); that the weight is $\beta$ needs [W], [Comp] and [N] in addition.  The later Parts add hypotheses that are local to one topic; each is stated as a labelled paragraph where it is first used, and the statements that use it name it.  They are " + "; ".join(parts) + r".  Appendix~\ref{app:inventory} (Table~\ref{tab:audit}) lists every input, conjecture and programme once, with its status and falsifiability class.")
    return "\n".join(out)

def replace_between(src, begin, end, body):
    a = src.index(begin) + len(begin)
    b = src.index(end, a)
    return src[:a] + "\n" + body + "\n" + src[b:]

def check(src):
    errs = []
    keys = [h["key"] for h in INV]
    if len(keys) != len(set(keys)):
        errs.append("duplicate inventory keys")
    hyp_labels = set(re.findall(r"\\label\{hyp:([A-Za-z]+)\}", src))
    inv_hyp = {h["key"] for h in INV}
    for k in hyp_labels:
        if k not in inv_hyp:
            errs.append("hypothesis not in inventory: " + k)
    for lab in re.findall(r"\\begin\{conjecture\}(?:\[[^\]]*\])?\s*\\label\{([^}]*)\}", src):
        if lab not in inv_hyp:
            errs.append("conjecture not in inventory: " + lab)
    for lab in re.findall(r"\\label\{([^}]*programme[^}]*)\}", src):
        if lab not in inv_hyp:
            errs.append("programme not in inventory: " + lab)
    in_scope = {k for _, _, ks in SCOPE for k in ks}
    for h in INV:
        if h["key"] not in in_scope:
            errs.append("inventory entry not in scope table: " + h["key"])
        if h["klass"] not in CLASSES:
            errs.append("bad class: " + h["key"])
    for _, c, _ in FALSIFY:
        if c not in CLASSES:
            errs.append("bad class in falsify table: " + c)
    tops = [t for t, _, _ in FALSIFY]
    if len(tops) != len(set(tops)):
        errs.append("duplicate rows in falsify table")
    return errs

def main():
    src = open(TEX).read()
    build(src)
    errs = check(src)
    for e in errs:
        print("ERROR:", e)
    if errs:
        sys.exit(1)
    src = replace_between(src, "%% BEGIN GENERATED INVENTORY (make_inventory.py)", "%% END GENERATED INVENTORY", inventory_table())
    src = replace_between(src, "%% BEGIN GENERATED STATUS TABLES (make_inventory.py)", "%% END GENERATED STATUS TABLES", scope_table() + "\n\n" + falsify_table())
    src = replace_between(src, "%% BEGIN GENERATED ASSUMPTIONS LIST (make_inventory.py)", "%% END GENERATED ASSUMPTIONS LIST", assumptions_list())
    open(TEX, "w").write(src)
    n = {k: sum(1 for h in INV if h["kind"] == k) for k in ("reading", "postulate", "convention", "hypothesis", "conjecture", "programme", "problem")}
    print("inventory entries:", len(INV), n)
    print("scope rows:", len(SCOPE), " falsify rows:", len(FALSIFY))
    print("INVENTORY CONSISTENT: True")

if __name__ == "__main__":
    main()
