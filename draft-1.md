# Inductive Biases for Semantic Representation Learning in Self-Supervised Learning

## 0. Central question

SSL theory takes the positive pair and the loss as given and describes what representation they optimize. The data do not say what to keep, what to drop, or which geometry is useful. Those answers come from outside: augmentations, a regularizer, a prediction target, or a model of future tasks.

The thesis asks which inductive biases turn unlabeled data into a semantically useful representation.

- **Part I. Theory (chapters 2-6).** What each SSL theory proves, what it assumes, and where semantics enters from outside the loss.
- **Part II. Application (chapters 7-8).** Two settings where we must know which semantics is needed: downstream-guided SSL and SSL on small samples.

Working principle: semantics does not follow from the loss. Every theorem buys a stronger conclusion with stronger assumptions. The contribution is to write the assumptions down and test them.

"Understand" here means: predict before the run what happens when temperature, batch size, augmentation strength or projector size change, and call the prediction wrong if the run disagrees.

### Where the work stands

| Section | Content | Main open point |
| --- | --- | --- |
| 2.1-2.5 Contrastive theory | InfoNCE, temperature, Wang-Isola, shortcuts, collapse saddle, downstream theory | ==Wang-Isola gives no kernel width and no semantics== |
| 2.6 PPS | $\tau$ = size of the positive cluster | ==transfer; inter-object dynamics== |
| 3 Regularizers and prediction | VICReg, BYOL, LeJEPA/SIGReg, SPHERE-JEPA, JEPA failures | ==why BYOL does not collapse; how many moments SIGReg needs== |
| 4 Unification | three families, drift, MMD, shape vs content | ==no single language yet: main theory gap== |
| 5 Reachability of semantics | task family, graph, open questions | ==hypothesis on good manifolds== |
| 6 Score matching | flow $V$, stop-gradient, generator | ==is an SGD step an Euler step of a Wasserstein flow?== |
| 7 Downstream-guided SSL | predictive information, $G_T$, gradient projection | ==what to measure== |
| 8 Small samples | application | ==how much data is enough== |
| 9.1 Robinson reproduction | shortcuts on Trifeature, STL-digits | ==no $\tau$ sweep, no seeds== |

The PPS paper (ICOMP 2026) is finished. The theory of chapters 2-6 is written with statements and proofs. The DriftSSL project produced ImageNet runs and A100 sweeps, but deriving $\sigma$ from the loss failed (2.6.8). Of the experiments, only the Robinson reproduction at one setting exists (9.1).

Unifying the methods is the largest open piece of theory. ==The experiments of 9.2-9.6 test the open claims of chapters 3-8.==

## 1. Notation

The symbols below are used throughout; the last two paragraphs fix what "probe" and "semantics" mean.

**Embedding.** Encoder $f:\mathcal X\to\mathbb R^D$, $z=f(x)/\|f(x)\|\in S^{D-1}$. On the sphere $\|z-z'\|^2=2-2z^\top z'$, so dot product and distance carry the same information.

**Graph.** A positive pair $(x,x^+)\sim p^+$ comes from two augmentations of one $x$. Negatives come from the marginal $p$. Graph $G$: vertices are observations, edge weight $w(x,x')=p^+(x,x')$, degree $w(x)=p(x)$. Normalized adjacency $\bar A=D^{-1/2}WD^{-1/2}$, Laplacian $L=I-\bar A$. Every method in chapter 4 is a way to define $G$ and learn functions consistent with it.

**Probe.** A linear probe on a frozen encoder measures linear accessibility of information, not its presence. Other measures: per-factor probe error, effective rank (exponent of the entropy of normalized singular values), uniformity.

**Task family.** A task is $y:\mathcal X\to\mathcal Y$; $\mathcal F$ is the set of tasks the representation will serve. "Semantics" means the choice of $\mathcal F$ (chapter 5).

## 2. Contrastive learning: theory

The InfoNCE denominator is a kernel density estimate on the sphere, and its limit is alignment plus uniformity. That limit does not choose semantics. Downstream guarantees need extra assumptions, and collapse is a saddle point. PPS ties the temperature to the positive-pair distance.

### 2.1. InfoNCE and temperature

InfoNCE is a softmax over one positive and $K$ negatives. On the sphere the softmax is a Gaussian kernel with $\sigma^2=\tau$, so temperature is a neighbourhood size. The gradient pushes close negatives harder.

For anchor $z$, positive $z^+$ and $K$ negatives $z_k^-$:

$$
\mathcal L=-\frac{s^+}{\tau}+\log\Big(e^{s^+/\tau}+\sum_{k=1}^K e^{s_k/\tau}\Big),\qquad s^+=z^\top z^+,\ s_k=z^\top z_k^-.
$$

It is the cross-entropy of "find the positive among $K+1$ candidates".

**2.1.1. Mutual information.**
*Prop. 2.1 (CPC).* For any critic, $I(x;x^+)\ge\log(K+1)-\mathcal L$; the bound is tightest for the critic $\propto p(x^+|x)/p(x^+)$.
*Proof.* InfoNCE is the Barber-Agakov bound with a critic self-normalized over $K+1$ samples (Poole et al.).
*Consequence.* The bound saturates at $\log(K+1)$ (McAllester-Stratos), and good representations do not come from maximizing information (Tschannen et al.).

**2.1.2. Softmax is a Gaussian kernel.** On the sphere $e^{z^\top z'/\tau}=e^{1/\tau}e^{-\|z-z'\|^2/(2\tau)}$, a Gaussian kernel with $\sigma^2=\tau$. Temperature is the size of the neighbourhood the loss treats as "close". The kernel $\propto e^{\kappa z^\top y}$ is von Mises-Fisher (vMF), a Gaussian bump on the sphere with sharpness $\kappa=1/\tau$. So the InfoNCE denominator is a kernel density estimate (KDE) of the embeddings.

**2.1.3. Gradient.**
*Prop. 2.2.* With $p_k=e^{s_k/\tau}/Z$, $p^+=e^{s^+/\tau}/Z$: $\partial\mathcal L/\partial z_k^-=\tfrac{p_k}{\tau}z$ and $\partial\mathcal L/\partial z^+=-\tfrac{1-p^+}{\tau}z$.
*Proof.* Differentiate log-sum-exp.
A close negative is pushed harder, and the pull toward the positive is scaled by $1-p^+$, which tends to zero once the negatives are far and $p^+\to1$. DCL removes this factor (2.3.3).

*Prop. 2.3 (after Wang-Liu).* The entropy of $p_k$ over negatives is non-decreasing in $\tau$.
*Proof.* For a Gibbs distribution $p_\beta\propto e^{\beta s}$, $dH/d\beta=-\beta\,\mathrm{Var}_{p_\beta}(s)\le0$, with $\beta=1/\tau$, so $dH/d\tau=\mathrm{Var}_{p}(s)/\tau^3\ge0$.
Small $\tau$ puts the pressure on the nearest negative, and large $\tau$ spreads it.

**Does the softmax repel similar points less, so that it acts like an augmentation that blurs them?** The opposite. The gradient of a negative is proportional to $p_k$, so a negative close to the anchor is pushed hardest; tolerance of close pairs comes only from large $\tau$.

### 2.2. Wang-Isola: alignment and uniformity

As $K\to\infty$ the loss splits into alignment and uniformity, and the uniform law minimizes the uniformity term. The result concerns the limit functional; training dynamics are outside it.

*Prop. 2.4 (Wang-Isola, Thm 1).* As $K\to\infty$,
$$
\mathcal L-\log K\to-\tfrac1\tau\,\mathbb E\,z^\top z^+\;+\;\mathbb E_x\log\mathbb E_{x^-}e^{z^\top z^-/\tau}.
$$
*Proof.* Divide the denominator by $K$; the negative sum converges by the law of large numbers (the integrand is bounded, $|s|\le1$), and $\log$ is continuous. The positive term $e^{s^+/\tau}/K$ vanishes; the random loss converges at $O_P(K^{-1/2})$, its expectation at $O(1/K)$ (Prop. 2.7).
The first term is alignment; the second is uniformity, the log-mean of a Gaussian kernel, which is smallest for spread-out embeddings.

*Prop. 2.5.* If an encoder exists with perfect alignment ($z=z^+$ a.s.) and uniform $z$ on the sphere, it is a global minimum of the limit functional (Wang-Isola).
*Our second-order check.* Take $\mu=(1+\varepsilon\varphi)\sigma_{\rm unif}$, $\int\varphi=0$, expand $\varphi$ in spherical harmonics. The kernel operator multiplies degree $\ell$ by $a_\ell$, and the increment is $\varepsilon^2\sum_\ell(a_\ell/a_0-a_\ell^2/(2a_0^2))\|\varphi_\ell\|^2>0$ because $0<a_\ell\le a_0$ for $e^{t\,u^\top v}$ ($a_\ell\propto I_{\ell+(D-2)/2}(t)>0$ by Funk-Hecke, decreasing in $\ell$). So the second variation is positive in every direction; global minimality is Wang-Isola's result.
Not every distribution is reachable by an encoder, so the result reads "if reachable". Wang-Liu (Prop. 2.3) carries over: $\tau$ trades uniformity against tolerance of close pairs.

### 2.3. What Wang-Isola does not give

Uniformity does not pick semantics: an easy feature can displace a useful one. Finite batches bias the loss, and DCL, debiasing and hard negatives patch this. Wang-Isola gives no kernel width and no notion of embedding quality; ==how to get a width from theory is open==.

**2.3.1. Uniformity does not choose semantics (shortcuts).**
*Prop. 2.6 (after Robinson et al.).* Let $x$ have an easy feature $s$ and a useful feature $t$; positives agree on both. A random negative shares the anchor's $s$ with probability $\delta$. If $g(s)^\top g(s')\le0$ for $s\ne s'$, the encoder $f=g(s)$ has loss at most $\log(1+K\delta+K(1-\delta)e^{-1/\tau})$. Using $t$ as well removes the $K\delta$ term, lowering the loss by $\log\frac{1+K\delta+Ke^{-1/\tau}}{1+Ke^{-1/\tau}}\le K\delta$.
*Proof.* Alignment is perfect. A negative with the same $s$ adds $e^0=1$ to the sum; the rest add at most $e^{-1/\tau}$.
*Consequence.* If $K\delta\ll1$, the gradient does not force learning $t$. Lower InfoNCE need not lower the per-factor error. Robinson et al. show this on Trifeature and propose Implicit Feature Modification (IFM). Reproduction: 9.1.

**Why not keep only repulsion and a narrow model, and let semantics emerge from compression?** Repulsion keeps points distinct and does not say which differences matter; attraction supplies that. A narrow model must drop something, and by Prop. 2.6 what it drops is decided by which features are cheapest. ==Whether a small width alone selects semantics is untested (chapter 8).==

**2.3.2. Finite sample.**
*Prop. 2.7 (delta method).* The bias vs the limit is $\approx-\mathrm{Var}(e^{s/\tau})/(2K(\mathbb Ee^{s/\tau})^2)$.
*Proof.* Second-order expansion of $\log$ (Jensen gap).
*Consequence.* Small $\tau$ needs a big batch; gradient accumulation does not help because $\log$ sits inside the batch.

**2.3.3. Variants.**
- *DCL (Yeh et al.).* The positive leaves the denominator, so the positive gradient is $-z/\tau$ without the $1-p^+$ factor. Same limit, better at small batch.
- *Debiased (Chuang et al.).* A fraction $\theta^+$ of "negatives" is same-class; subtract the estimated positive contribution with a floor $e^{-1/\tau}$.
- *Hard negatives (Robinson et al.).* Sample negatives $\propto e^{\beta s}p(x^-)$ by importance sampling.

**Why does DCL work better if its denominator is not a probability?** The limit $K\to\infty$ is the same, so nothing is lost asymptotically. At a small batch the factor $1-p^+$ weakens the positive gradient exactly when the negatives are easy, and DCL removes it.

**2.3.4. False negatives.** FNC (Huynh et al.) and IFND (Chen et al.) use the current encoder to mark negatives lying close to the anchor as false negatives and drop or attract them, so a shared background can pass for a shared class.

**2.3.5. Augmentations are hidden supervision.** Requiring $f(x)\approx f(Tx)$ for every augmentation $T$ declares $x$ and $Tx$ equivalent; the transitive closure of this relation is the partition of $G$ into connected components (Prop. 5.2). Cui et al. write downstream risk as a function of the augmentation choice: strong augmentations cut within-class variance but erase task features.

**Which augmentations suit which tasks?** By Prop. 5.4 the optimal views share exactly the information about the task, so a rule that picks augmentations needs the task. ==Experiment "invariant vs random vs equivariant augmentations" (9.2) compares three types on one task.==

**2.3.6. Temperature.** The best fixed $\tau$ depends on architecture, batch and dataset and is found by search. Adaptive rules and PPS: 2.6.

**2.3.7. Further limits.** Koromilas et al.: ==finite vs asymptotic losses, kernel losses==. AnInfoNCE (Rusak et al.): the number of recovered factors does not fix downstream accuracy. Nie et al., Fang et al.: alignment and uniformity are imperfect quality metrics.

**2.3.8. What is embedding quality, and how to measure it against the world?** There is no general embedding quality. Proxies (alignment, uniformity, RankMe) are many and none suffices. Quality is the match between loss and task, so it is always defined through $\mathcal F$ (Notation). To measure it against the world, fix $\mathcal F$ on a toy with known factors and compare the proxies with probe error and retrieval ==(experiment "embedding quality vs world", 9.2)==. Attempts to derive the kernel width from Wang-Isola fail for the same reason (2.6.8): the theorem describes the limit functional, not the path to it.

> ==Wang-Isola is phenomenology: it says what happens at $K\to\infty$, and the mechanism has to come from dynamics and from $G$ (2.5, 5). Does it yield one prediction that a run can refute?==

### 2.4. Downstream theory

Three lines add assumptions to get downstream guarantees: latent classes (Saunshi), an augmentation graph (HaoChen), identifiability (Zimmermann). The graph assumption fails in practice, and ==the architecture's role is not yet in the theory==.

**2.4.1. Saunshi et al.: latent classes.**
*Prop. 2.8.* If positives come from one latent class, negatives from the marginal, and $\theta^+$ is the chance a negative has the same class, the supervised loss of the class-mean classifier is $\lesssim\frac1{1-\theta^+}(\mathcal L_{\rm un}-\theta^+)$.
*Proof.* Convexity of the logistic loss (Jensen), a collision correction, Rademacher bound.
The theorem assumes that positives are independent within a class, which real augmentations are not.

**2.4.2. HaoChen et al.: augmentation graph.**
*Prop. 2.9.* The spectral contrastive loss $\mathcal L_{\rm spec}(F)=-2\mathbb E_{x,x^+}F(x)^\top F(x^+)+\mathbb E_{x,x'}(F(x)^\top F(x'))^2$ equals $\|FF^\top-\bar A\|_F^2$ up to a constant, after rescaling rows by $\sqrt{w(x)}$.
*Proof.* Expand $\sum(\sqrt{w_xw_{x'}}f_x^\top f_{x'}-w_{xx'}/\sqrt{w_xw_{x'}})^2$ into quadratic, linear and constant parts.
*Consequence (Eckart-Young-Mirsky).* The best rank-$k$ fit of $\bar A$ is its top eigenvectors (bottom ones of $L$).
*Prop. 2.10 (HaoChen et al., Thm 3.8).* If labels are recoverable from augmentations with error $\alpha$ (few edges cross classes) and $\rho_{\lfloor k/2\rfloor}$ is the conductance of the sparsest partition of $G$ into $\lfloor k/2\rfloor$ parts, the linear-probe error of the population minimizer is $\tilde O(\alpha/\rho_{\lfloor k/2\rfloor}^2)$.
If classes are connected components of $G$, the zero eigenvectors of $L$ are the class indicators and the probe is perfect; weak inter-class edges perturb this (Davis-Kahan). The theorem assumes that the graph reflects classes, which is the outside information of chapter 5.

Augmentations of different images almost never overlap, so the graph is nearly discrete, yet the method works: the theorem survives, its explanation does not.

> ==Is there a guarantee for a soft graph, where closeness in a learned metric replaces a shared edge?==

**2.4.3. Identifiability.**
*Prop. 2.11 (Zimmermann et al.).* If $z$ is uniform on $S^{d-1}$, $p(\tilde z|z)\propto e^{\kappa z^\top\tilde z}$, the generator $g$ is injective and the encoder $h$ maps to the sphere, every minimizer of the contrastive loss at $K\to\infty$ has $h\circ g=R$ with $R$ orthogonal.
*Proof.* At $K\to\infty$ the loss is, up to constants, the cross-entropy between the true vMF conditional and the model conditional $\propto e^{h(x)^\top h(\tilde x)/\tau}$ (Prop. 2.4). At its minimum $h\circ g$ preserves inner products, and such a map of the sphere is orthogonal.
*Prop. 2.12 (Locatello et al.).* Without assumptions, disentanglement is impossible.
*Proof sketch (Gaussian case).* If $z\sim\mathcal N(0,I)$, then $Rz$ has the same law for any rotation $R$, so data cannot tell the coordinates of $z$ from those of $Rz$. For any factorized prior the same holds with nonlinear bijections (Locatello et al.).

**2.4.4. Inductive biases and feature learning.**
- ==Saunshi et al. (2022): graph-based guarantees are vacuous without the architecture, since a rich function class minimizes the loss without useful semantics.== The architecture bias then belongs in the theory next to the graph, which is the thesis title.
- ==Wen, Li (2021): augmentations decorrelate dense features between positives and leave sparse ones intact, so contrastive learning extracts the sparse features; this is why it needs stronger augmentations than supervised learning.== The augmentation choice then decides which features survive, as in 2.3.5 and Prop. 2.6.

> ==Can the assumptions of Saunshi et al. (2022) and Wen-Li be stated as weak cuts of $G$ (5.1)?==

### 2.5. Dynamics and collapse

The all-equal state is a strict saddle of InfoNCE with escape rate $\propto1/\tau$. Without negatives collapse is a minimum, so the safeguard must come from elsewhere.

**2.5.1. Collapse is a saddle.**
*Prop. 2.13 (ours).* The state with all embeddings equal to $c$ is a critical point of InfoNCE with a negative Hessian eigenvalue.
*Proof.* Perturb $z_a=c+\varepsilon u_a$, $u_a\perp c$, $d_{ab}=\|u_a-u_b\|^2$. On the sphere $s_{ab}\approx1-\tfrac{\varepsilon^2}2d_{ab}$. For anchor $i$ with candidates $C_i$ (positive and $K$ negatives)
$$
\mathcal L_i\approx\text{const}+\frac{\varepsilon^2}{2\tau}\Big(d_{ii^+}-\frac1{K+1}\sum_{j\in C_i}d_{ij}\Big).
$$
First order vanishes. Take $u_i=u_{i^+}=v_i$ with independent $v_i$: $d_{ii^+}=0$, the mean distance to candidates is positive, the quadratic form is negative.
*Consequence.* Collapse is a strict saddle; escape separates different pairs and keeps pairs together; the rate scales as $1/\tau$.
An alignment-only loss has quadratic form $\sum d_{ii^+}\ge0$, so collapse is a minimum there, and methods without negatives (chapter 3) need a regularizer or dynamics.

**2.5.2. Ziyin et al.** Linear-model landscape: contrastive and non-contrastive losses have similar collapse structure; they differ in what stabilizes the nontrivial minimum.

**2.5.3. Lyapunov, mean-field.** Gradient flow decreases the loss, $\dot{\mathcal L}=-\|\nabla\mathcal L\|^2\le0$, so the loss is a Lyapunov function. As $N\to\infty$ particles become a PDE for the density on the sphere (Mei-Montanari-Nguyen; Chizat-Bach). Physics analogies (Debye/Yukawa potentials, phase transition) are intuition only: particles are embeddings of one shared encoder, not independent bodies.

**2.5.4. Levels of theorem.** Theorems about SSL sit at six levels (local stability, global minimum, convergence of dynamics, generalization, finite sample, adaptive control), and none of them says which semantics is learned.

### 2.6. Temperature as geometry scale: PPS (own work)

The Positive-Pair Schedule (PPS) sets $\tau$ to the mean squared distance between the two views of one image (ICOMP 2026, "Positive-Pair Distance Schedules Temperature in Contrastive Learning Without a Grid Search"). The drift identity is exact; the dynamics prove a safe regime, and the accuracy optimum lies outside the model. ==Whether PPS transfers beyond one setup is open.==

**2.6.1. Motivation.** Since $\tau=\sigma^2$ is the kernel width of a density estimate of the embeddings (2.1.2), it should follow their current geometry rather than a grid. The mean-shift identity (2.6.3, Prop. 2.14) is exact; the choice $\sigma^2=p$ is a heuristic. Silverman's AMISE optimizes density estimation, and the contrastive loss has no such criterion (6.3), so the KDE language links PPS to score matching but does not justify $\sigma$.

**2.6.2. Rule.** $\tau(t)=p(t)$, the mean squared distance between two views of one image. A kernel as wide as the positive cluster sees it as one point and sees other clusters as separate.

**2.6.3. Theory.**
*Prop. 2.14 (exact).* $\nabla_{z_i}\mathcal L_i=-\frac{1-w^+}{\sigma^2}V(i)$ with drift $V=\mu^+-\mu^-$: $\mu^+$ is the centroid of positives, $\mu^-$ the softmax-weighted centroid of negatives. Equivalently $V/\sigma^2=\nabla_z\log(\hat p^+/\hat p^-)$.
*Proof.* For a Gaussian KDE $\hat p$ of width $\sigma$, mean-shift gives $\nabla\log\hat p(z)=(m(z)-z)/\sigma^2$ with $m$ the kernel-weighted mean. Apply to positives and negatives and subtract (Fukunaga-Hostetler, Hyvarinen, Vincent).
*Prop. 2.15 (mean-field model).* Assume A1 (two views give nearly equal negative terms), A2 (embeddings spread, mean near zero), A3 (positive distance above $p_{\min}>0$). Then $\dot p=-a\phi(p-p_{\min})$, $a>0$, with $\phi(0)=0$ and $\phi(u)>0$ for $u>0$, and $\Phi=p-p_{\min}$ is a Lyapunov function. The inter-object coordinate $q$ has a fixed point $q^*(\sigma)$ growing from zero with $\sigma$; the schedule stays in the healthy region if its scale does not depend on $\sigma$ and stays large while the gap is small.
*Prop. 2.16.* Choosing $\sigma$ to maximize the gradient norm gives $\sigma^{*2}=\max\big(0,((d^-)^2-(d^+)^2)/c_N-d_{\rm clust}^2\big)$, $c_N=2(1+W(N_{\rm eff}/e))$ ($W$ Lambert). It is clipped to zero at initialization, where the gap is small, and training collapses.

**2.6.4. Results.** SimCLR, ResNet-18, CIFAR-10/100, ImageNet-100. PPS matches the best tuned constant $\tau$ (CIFAR-10: probe 75.62, kNN 80.74) and the adaptive schedules of Kukleva, Huang, Manna, Qiu.

**2.6.5. Limits.** One seed, one architecture, one batch size. The inter-object dynamics of $q$ depends on the encoder and has no closed form.

**Does convergence speed depend on the temperature schedule?** ==Open: the results compare final accuracy, and the experiment "convergence under $\tau$ schedules" (9.2) measures epochs to a fixed probe accuracy.==

> ==Does PPS hold across seeds, architectures, batches, other methods and domains (9.3)?==

**2.6.6. Role.** Bridge between chapters 2 and 6: the same score and kernel scale as in diffusion. Drift $V$ is also a common language for regularizers (4.3).

**2.6.7. Related temperature rules.** Six works schedule or adapt the temperature. They differ in what sets the width (the signal), at what level (global, anchor, pair) and how the rule is justified.
| Work | Rule | Signal | Level | Justification |
| --- | --- | --- | --- | --- |
| Kukleva et al. | ==cosine in epoch between $\tau^-$ and $\tau^+$== | time only | global | ==intuition (group vs instance discrimination); no derivation== |
| Manna et al. (DySTreSS) | ==bounded function of pair cosine== | pair similarity | pair | ==design criteria; form fitted== |
| Qiu et al. (iSogCLR) | ==per-anchor $\tau_i$ as Lagrange multiplier in KL-DRO== | negative distribution | anchor | ==Sion duality; $O(N)$ memory== |
| Huang et al. (MACL) | ==$\tau=\tau_0[1+\alpha(A-A_0)]$== | alignment $A$ | global | ==InfoNCE gradient analysis; linearity is a postulate== |
| Khaertdinov et al. | ==$\tau$ from an external autoencoder== | external model | pair | ==engineering for sensor data== |
| Wang et al. (AMCL) | ==$\tau_{ij}^{(c)}$ per head and pair== | likelihood over heads | head x pair | ==regularized MLE== |
| PPS (ours) | $\tau=p$, mean squared positive distance | positive-pair geometry | global | drift $V$ and dynamics (Prop. 2.14-2.16) |

Kim (temperature-free loss) and learned $\tau$ (CLIP) stand apart. Huang goes the opposite way to PPS: their $\tau$ grows as positive pairs tighten. Closest in spirit is Qiu (both derive $\tau$ from a principle); they differ in the principle (robustness vs kernel geometry) and level (anchor vs global).

> ==Does any work set the kernel width inside the uniformity term, rather than the softmax $\tau$, by density estimation?==

**2.6.8. What failed in DriftSSL.**
- Deriving $\sigma$ directly from the loss does not work: it assumes the representation has already learned something, otherwise everything collapses at init (Prop. 2.16). A MoCo-like moving average for stable statistics is a possible fix; it does not work yet.
- A controller on the drift norm $V$ is unstable.
- The DriftSSL loss without a BYOL-like scheme collapses almost immediately: shrinking the vector norm pulls all points together, which is locally optimal.
- The loss collapses although the drift identity holds, so the claim that the drift norm tracks a Fisher divergence (4.3) needs a new check.
- The kernel analogy with InfoNCE holds. The Laplace-distribution argument generalizes to all symmetric exponential laws, so Laplace is not special. Vanishing Anchor Trick has no point; Tangent Space Projection is sensible but not required.
- The parts that preserve local structure match baselines without hand-tuned hyperparameters; baselines are not beaten.

> ==Describing points by pairwise distances makes the link to score matching fit much better (6.6).==

## 3. Regularizers and prediction

Without negatives there is no repulsion, and by 2.5.1 such a loss has collapse as a minimum, so a regularizer or the dynamics must stop it. VICReg fixes two moments, SIGReg the whole distribution, and neither says what information to keep. ==Where they fail (slow features, temporal collapse) has not been reproduced.==

### 3.1. Regularizers against degeneracy

VICReg and Barlow Twins approximate spectral embeddings of the positive graph, BYOL relies on dynamics, LeJEPA/SIGReg targets an isotropic Gaussian, SPHERE-JEPA a uniform law on the sphere. ==Why BYOL does not collapse, and how many moments a regularizer needs, are open.==

**3.1.1. VICReg and Barlow Twins.** VICReg: $\lambda s(Z,Z')+\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')]$; $s$ is MSE between views, $v$ a hinge on per-coordinate variance, $c$ a penalty on off-diagonal covariance. Barlow Twins pushes the cross-correlation matrix of two views to $I$.

*Prop. 3.1.* Under $Z^\top Z=I$, $\min\mathrm{tr}(Z^\top LZ)$ is solved by the bottom eigenvectors of $L$ (Ky Fan).
*Proof.* Rayleigh-Ritz for the sum of the $k$ smallest eigenvalues.
VICReg is a soft version of this, so it approximates Laplacian Eigenmaps on the positive-pair graph (Balestriero-LeCun, 4.1), and Barlow Twins is close to CCA.

**3.1.2. Why BYOL and SimSiam do not collapse.** BYOL and SimSiam train an online network with a predictor against a target branch under stop-gradient; in BYOL the target is an EMA copy. The loss admits collapse, so the dynamics protect it, not $G$. ==Tian, Chen, Ganguli: on a linear model the predictor aligns with the eigenspaces of the correlation matrix, and weight decay is needed for a stable nontrivial solution.== Several explanations compete and none closes the question: predictor dynamics with stop-gradient and the spectral picture (DirectPred), eigenspace alignment, implicit centering. ==Jing et al. describe dimensional collapse and propose DirectCLR (loss on a subvector, no trainable projector).== In drift terms (4.3) BYOL is the only method where repulsion is not written in the loss.

> ==Where do the three theories of BYOL's non-collapse make different predictions (experiment "three theories, one question", 9.2)? The disagreements map what SSL does not understand.==

**3.1.3. LeJEPA and SIGReg.**
*Prop. 3.2 (simplified LeJEPA argument).* Ridge probe with parameter $\lambda$, unknown task direction $w$ uniform on the sphere, embedding covariance eigenvalues $s_j$ with fixed $\sum s_j$. The expected bias $\frac{\|w\|^2}{d}\sum_j(\frac{\lambda}{s_j+\lambda})^2$ is minimal at $s_1=\dots=s_d$.
*Proof.* $s\mapsto(\lambda/(s+\lambda))^2$ is convex; apply Jensen.
The ridge variance term $\sum_j s_j/(s_j+\lambda)^2$ is concave for $s_j<2\lambda$, so the Jensen step covers the bias term only; ==the full LeJEPA argument uses Fisher information instead==.

SIGReg pushes the embedding distribution to $\mathcal N(0,I)$. It tests random 1D projections (Cramer-Wold: projections determine the law) with the Epps-Pulley statistic, the distance between empirical and Gaussian characteristic functions. The statistic is zero iff the projection is $\mathcal N(0,1)$, because the characteristic function determines the law and the Gaussian weight has full support.

**3.1.4. Assumption.** Let $\eta(z)=\mathbb E[y\,|\,z]$. For an unknown task LeJEPA assumes $\mathbb E[\nabla\eta\nabla\eta^\top]\propto I$, so no direction of $z$ is more useful; Prop. 3.2 assumes the same through $w$ uniform on the sphere.

**3.1.5. Is LeJEPA better, and why.**
- *VICReg is LeJEPA truncated at the second moment.* Variance and covariance fix two moments; everything above is free, so embeddings may have heavy tails or multimodality at zero VICReg loss. SIGReg matches the characteristic function, all moments at once.
- *Versus contrastive.* Uniformity on the sphere is also a full target geometry, but it is the loss asymptotics, not derived from downstream risk, and needs negatives or big batches. LeJEPA drops stop-gradient, teacher-student and schedules; one hyperparameter remains.
- ==*Empirics.* 79% linear probe on ImageNet-1k with ViT-H/14, stable across dozens of architectures, shorter schedule than I-JEPA. Comparable to DINOv2 in quality, with fewer components and a theory behind it.==
- *Caveat of the theory.* Optimality of the isotropic Gaussian is worst-case over tasks. If the task is known, anisotropy may be better, and isotropy erases global scale structure.
- *Untouched.* Augmentation choice, architecture bias, layer choice (before or after the projector).

Testable: if only two moments matter, the VICReg-LeJEPA gap should vanish where embeddings are nearly Gaussian. ==Weak-SIGReg (project into a sketch space, push covariance to identity) asks how many moments are needed, but was tested as a supervised stabilizer, so it gives no direct SSL answer.==

> ==How many moments does a regularizer need? VICReg, Weak-SIGReg and SIGReg on data with controlled non-Gaussianity (9.4).==

**3.1.6. SPHERE-JEPA: target on the sphere.** ==SPHERE-JEPA (arXiv:2605.26900) replaces the Gaussian target of SIGReg with the uniform law on the sphere; same machinery (random projections, Cramer-Wold). Argument: the Gaussian density is non-uniform, so kNN neighbourhoods are anisotropic, and the worst case over tasks gives uniform density. In high dimension a projection of the uniform law is about $\mathcal N(0,1/D)$, so SIGReg per direction nearly equals uniformity and the difference is mostly the radius. Expanding SPHERE-JEPA integrates projections analytically into a family of sphere regularizers (MMD, KSD, KL over a KDE; heat and band-limited kernels): MMD and KSD give local clusters, KL gives instance separation.== Section 4.3 uses these regularizers.

### 3.2. Prediction targets

Squared prediction learns the conditional mean: a blur in pixel space, a collapse risk in latent space. What is predictable is not necessarily useful.

*Prop. 3.3.* The minimizer of squared prediction loss of target $y$ from context $c$ is $\mathbb E[y|c]$.
*Proof.* $\mathbb E\|y-g(c)\|^2=\mathbb E\|y-\mathbb E[y|c]\|^2+\mathbb E\|\mathbb E[y|c]-g(c)\|^2$.
*Consequences.* (1) MAE: in pixel space with a multimodal target the prediction is the mean of modes, a blur. (2) I-JEPA, data2vec: in latent space the target encoder can drop the unpredictable, but then a regularizer (SIGReg, var/cov, EMA) is needed or the target collapses. (3) Prediction learns what is predictable from context; predictable need not mean useful (5.3.1).

**3.2.1.** Reconstruction keeps information that need not be linearly readable (MAE-CT, superposition). **3.2.2.** VICRegL applies the same losses to local features.

### 3.3. What regularizers do not set

Variance, covariance and Gaussian terms depend only on the set of embeddings, so they cannot decide who is close to whom. That comes from the positive pairs and the architecture.

*Prop. 3.4.* Variance, covariance and $\mathcal N(0,I)$ terms depend only on the multiset of rows of $Z$; any relabeling of objects leaves them unchanged.
*Proof.* All three are functions of the empirical distribution.
*Consequence.* Who is close to whom is set by the positive pairs ($G$) and by the architecture's inductive bias (2.4.4). A regularizer keeps the space from degenerating and gives it no meaning.

### 3.4. Where JEPA and SIGReg break

Three failures show that the information to keep comes from a prior or from the task: a constant distractor wins as the slowest feature, the static part suppresses the dynamic part (in the prediction target and in the regularizer), and differences alone give weak semantics. They also set criteria for chapter 7.

- **Slow features (arXiv:2211.10831).** ==A constant distractor is the slowest signal, so a JEPA with a slow-feature objective prefers it to the useful signal.== This is an instance of Prop. 3.3 and 5.3.1.
- ==**MotionJEPA (arXiv:2609.23881).** Temporal feature collapse: the static part suppresses the dynamic part. The DISReg regularizer predicts the embedding of frame differences. On Pong, ball NMSE falls from 1.26 (forward-only baseline) to 0.005.==
- ==**TC-LeWM (arXiv:2607.26924).** In LeWorldModel, SIGReg on the whole latent leaves the temporally centered residual with little variance: the persistent part fills the unit variance budget of each projection and prediction favours it. Applying SIGReg to $r_t=z_t-\bar z_t$ (window of 4 frames) raises the success of a policy on frozen features from 63.6% to 83.8% on LIBERO (suite-wise, 10 tasks). The variance argument is a Monte Carlo toy with independent Gaussian parts.==
- ==**TDV (arXiv:2606.15956).** Learning from temporal differences; kNN top-5 on ImageNet-1k after pretraining on SSv2: 17.05 for TDV vs 40.19 for DINO with full augmentations. Possible reading: dynamics alone are not enough for semantics.==

A regularizer removes collapse but does not choose which information to keep (Prop. 3.4). The choice comes from a hand-made prior (DISReg) or a signal from the task (chapter 7).

> ==Can a few labels close the same gap as DISReg on a toy with known factors (7.5, direction 1)?==

## 4. Unification of SSL losses

The goal is one language for all SSL losses. The spectral picture (4.1) covers graphs and linear or kernel embeddings; the drift picture (4.3) covers dynamics and regularizers with a target law. ==No single language yields both and makes testable predictions, so the unification is the main gap in the theory.== Section 4.7 places each method by the one thing it changes.

> ==A unifying language is useful only if it predicts something that is not known without it.==

### 4.1. Spectral picture

Under conditions, VICReg is Laplacian Eigenmaps, SimCLR is Kernel ISOMAP, Barlow Twins is CCA. Methods differ in how $G$ is built and normalized and whether the embedding is linear.

By Props 2.9 and 3.1 the loss is $\|ZZ^\top-\bar A\|_F^2$ or a soft version, and the solution is a spectral embedding of $G$.

*Prop. 4.1 (Balestriero-LeCun).* Under suitable conditions VICReg is Laplacian Eigenmaps, SimCLR is Kernel ISOMAP, Barlow Twins is CCA.
*Proof sketch.* Reduce each loss to a spectral problem: Ky Fan (Prop. 3.1) for the Laplacian, Eckart-Young for a kernel matrix. Exact conditions (linearity, infinite batch, augmentation type) are in the paper.

### 4.2. Three families

Methods fall into three families: explicit negatives, distillation, redundancy reduction. Each method has three parts: where $G$ comes from (augmentations, time, masks), what prevents collapse, and what is predicted. With the wide meaning of "contrastive" all three families repel.

| Family | Methods | What prevents collapse | Negatives |
| --- | --- | --- | --- |
| Explicit negatives | SimCLR, MoCo, NNCLR | InfoNCE denominator | batch, EMA queue, bank |
| Distillation without negatives | BYOL, DINO (v1) | asymmetry: EMA teacher + predictor (BYOL), centering + sharpening (DINO) | implicit |
| Redundancy reduction | VICReg, Barlow Twins | variance and covariance terms | implicit |
| Related | SwAV (online clustering, Sinkhorn), W-MSE (whitening + MSE), DirectPred/DirectCLR (spectrum regularization instead of a predictor) | partition constraint, whitening, spectrum | implicit |

**Which of these are contrastive?** It depends on the definition. Narrow (explicit negative denominator): SimCLR, MoCo, NNCLR, DirectCLR. Wide (attraction and repulsion both present): nearly everything except pure BYOL, since DINO repels via centering, VICReg and Barlow Twins via decorrelation, W-MSE via whitening. The wide definition is used below: everything repels, only the implementation differs. One representative per family is enough for experiments: SimCLR, BYOL (or DINO), VICReg.

### 4.3. Drift as a common language

One form covers the families: the drift $V(i)=\mu^+-\mu^-$ of 2.6.3, where they differ only in what implements $\mu^-$. The MMD gradient has the same form with $\nabla\hat p$ instead of $\nabla\log\hat p$, and Epps-Pulley is MMD on a slice. ==Stalls of the MMD flow and the link of the drift norm to the kernelized Stein discrepancy (KSD) are open.==

| Family | Role of $\mu^-$ |
| --- | --- |
| Explicit negatives | softmax-weighted mean of negatives |
| Distillation | implicit "mean collapse direction" that centering and EMA steer away from |
| Redundancy reduction | projection onto already covered directions (redundant correlations) |
| Target regularizer (SIGReg, SPHERE-JEPA) | the batch itself (self-repulsion); attraction goes to a sample from the target law |

*Prop. 4.2 (MMD gradient as drift).* Let $z_1..z_N$ be embeddings, $y_1..y_M$ a sample from the target, $k(z,y)=e^{-\|z-y\|^2/(2\sigma^2)}$, $\hat p_P=\frac1N\sum k(\cdot,z_l)$, $\hat p_Q=\frac1M\sum k(\cdot,y_j)$ (unnormalized). Then
$$
-\nabla_{z_i}\widehat{\mathrm{MMD}}^2=\tfrac2N\big(\nabla\hat p_Q(z_i)-\nabla\hat p_P(z_i)\big).
$$
*Proof.* Expand $\widehat{\mathrm{MMD}}^2=\frac1{N^2}\sum k(z_l,z_{l'})-\frac2{NM}\sum k(z_l,y_j)+\mathrm{const}$ and use $\nabla_zk(z,y)=k(z,y)(y-z)/\sigma^2$: attraction to target points, repulsion from own points.
This is the same "attraction minus repulsion" as $V$, with $\nabla\hat p$ instead of $\nabla\log\hat p$: normalized weights (InfoNCE, KL over a KDE) against unnormalized ones (MMD), related by $\nabla\hat p=\hat p\nabla\log\hat p$.

*Prop. 4.3 (Epps-Pulley is MMD on a slice).* For 1D samples, $\int|\hat\varphi_N(t)-\varphi_0(t)|^2w(t)dt$ with Gaussian $w$ equals $\mathrm{MMD}^2$ with a Gaussian kernel between empirical and target laws; the Epps-Pulley statistic differs by a factor $N$.
*Proof.* Expand the squared modulus with $\hat\varphi_N(t)=\frac1N\sum e^{itx_l}$ and integrate against $w$; by Bochner $\int e^{it(x-x')}w(t)dt=k(x-x')$.
*Consequence.* SIGReg is a sum of MMD over random slices and SPHERE-JEPA the same with a uniform target. ==Expanding SPHERE-JEPA averages over all slices exactly, giving MMD on the sphere with an induced kernel.==

*Statement (repulsion of InfoNCE).* At $K\to\infty$ the uniformity term equals $\mathbb E_{p}\log(p*k_\tau)+\text{const}=-H(p)-\mathrm{KL}(p\,\|\,p*k_\tau)+\text{const}$, with $k_\tau$ the vMF density.
*Proof.* $\mathbb E_{x^-}e^{z^\top z^-/\tau}=C_\tau(p*k_\tau)(z)$, and $\mathbb E_p\log q=-H(p)-\mathrm{KL}(p\|q)$.
*Consequence.* It is KL to uniform only as $\tau\to0$, where $-H(p)=\mathrm{KL}(p\|\mathrm{unif})-\log|S^{D-1}|$. ==Whether this matches the KL variant of Expanding SPHERE-JEPA is unchecked.==

Three things follow. First, $V\equiv0$ on the whole space iff $\hat p^+=\hat p^-$, and a characteristic kernel then implies equal laws. Second, $\|V\|^2=\sigma^4\|\nabla\log(\hat p^+/\hat p^-)\|^2$ by Prop. 2.14, whose batch mean estimates $\sigma^4$ times a Fisher divergence, so a drift-norm monitor applies without labels to SIGReg and SPHERE-JEPA; ==its link to KSD is unchecked==. Third, the kernel width is the weight width in Epps-Pulley and the time in the heat kernel; ==the hypothesis is that positive-pair distance sets it, as in PPS==.

**What is MMD, simply?** Smear each point of two clouds with a Gaussian bump and measure how different the two resulting pictures are. For a characteristic kernel the number is zero exactly when the clouds are the same law.

**Can SIGReg and SPHERE-JEPA be described by drift?** Yes. By Props 4.2 and 4.3 their gradient is attraction to a sample of the target minus repulsion from the batch. They differ from InfoNCE in having an explicit target, and from $V$ in weighting by $\hat p$.

**Does drift show that SIGReg works?** No. That the SIGReg minimum is the target follows from Cramer-Wold. Why the target is good downstream is the minimax statement of 3.1.3 and 4.4, which drift does not reach.

**Why is collapse impossible while the drift is nonzero?** Nonzero drift does not forbid it. Collapse to a point is stationary for the batch field when the target is symmetric about that point (the uniform law on the sphere, a Gaussian at its mean) and for InfoNCE (Prop. 2.13), because the attraction and repulsion cancel there. What rules collapse out is instability: Prop. 2.13 for InfoNCE, ==unchecked for MMD to the uniform law==.

> ==(1) $V$ is computed only at batch points. By Arbel et al. (2019) convergence of the MMD flow to the target holds only under extra conditions, since MMD is not displacement-convex, so a state where the field vanishes at the points but the cloud has not matched the target cannot be excluded. Are there stable stalls for the vMF kernel on the sphere? A proof that there are none would be a closed piece of theory. (2) Can the three families be written once, with $V$ computed from the loss instead of defined per method?==

### 4.4. Shape and content

A homeomorphism can turn any smooth density into another, so shape and content can be separated. The repulsive part sets shape and has minimax theorems; the attractive part sets content and has none. ==What enforcing the shape costs in network capacity is open.==

*Prop. 4.4 (separation).* For positive smooth densities $p,q$ on one connected manifold $M$ of dimension $m$ ($\mathbb R^m$ or compact, e.g. a sphere), there is a homeomorphism $T:M\to M$ with $T_\#p=q$.
*Proof.* On $\mathbb R^m$: Knothe-Rosenblatt (successive conditional CDFs). On a compact manifold: Moser (two volume forms of equal total volume are related by a diffeomorphism).
So the cloud's shape can change without gluing points or losing information, and shape and content can be discussed separately. The assumption of one manifold $M$ of dimension $m$ matters: If data lie on a manifold of dimension below $D$, full isotropy in $\mathbb R^D$ is unreachable without gluing; extra axes fill with noise or stay empty.

- **Shape** is set by the repulsive part $\mu^-$: isotropic Gaussian (LeJEPA), uniform on the sphere (SPHERE-JEPA, InfoNCE at $K\to\infty$). Minimax theorems exist: if information is already in the embeddings, this shape hurts the probe least in the worst case.
- **Content** is set by the attractive part $\mu^+$: what counts as one object (augmentations, time, predictor). No theorems.

**Why call the target optimal if it only prevents collapse?** Only as a shape: "the target is optimal as shape, provided content is already learned." The network has finite capacity, so shape and content can conflict: positives want clusters, uniformity wants spread. This is the $\mu^+$ vs $\mu^-$ tension. If positives are bad, a perfect shape only prevents collapse.

> ==Where does shape start to displace content when the width or dimension of the network shrinks?==

### 4.5. Level of structure

A graph with one weight per edge stores only "similar by this much", so it teaches invariance. Equivariance needs an operator $R_{ij}$ per edge, and if the operators are consistent on cycles a global frame exists, which gives a penalty for learned operators. Hierarchy needs several scales.

*Prop. 4.5 (connection Laplacian).* Let $G$ be connected, take orthogonal $R_{ij}\in O(d)$ with $R_{ji}=R_{ij}^{-1}$, and the energy $\mathcal E(z)=\sum w_{ij}\|z_i-R_{ij}z_j\|^2$. If the product of $R$ along every cycle is $I$ (for a triangle, $R_{ij}R_{jk}R_{ki}=I$), then $R_{ij}=g_ig_j^{-1}$ for some $g_i\in O(d)$ and $\min\mathcal E=0$ at nontrivial $z$; at $R\equiv I$ this is the usual Laplacian.
*Proof.* Pick a spanning tree, set $g_{\rm root}=I$, extend $g_j=R_{ij}^{-1}g_i$ along edges; cycle consistency makes it well defined.
*Consequence.* Cycle consistency is the condition for a global coordinate system and can penalize learned $R_{ij}$ ("learned graph with cycle consistency", 9.4). Without consistency one gets Vector Diffusion Maps and sheaf networks; hierarchy is handled by diffusion wavelets (several scales).

### 4.6. Equivariance and learned symmetries

Exact equivariance helps when the symmetry is known; learned symmetries exist but need a safeguard against the trivial solution $E\equiv0$, $R\equiv I$.

- **4.6.1.** Exact equivariance (G-CNN, tensor field networks, NequIP): big gain if the symmetry is known and exact.
- ==**4.6.2.** Learned symmetries: LieGAN, Neural Isometries, Self-supervised Transformation Learning, Equivariance by Contrast. Nearby: ContextSSL (symmetries from context), AIL (invariance chosen during training).==
- **4.6.3. Trivial solution.** For $\sum\|F(x)-R\,F(Tx)\|^2$ with encoder $F$, the choice $F\equiv0$, $R\equiv I$ gives zero. Equivariance does not prevent collapse by itself; it needs variance or whitening terms, reconstruction, or a penalty against $R=I$.
- **4.6.4. Known in advance.** Each method fixes one of these in advance: the neighbourhood, the positive pairs, the symmetry family, the graph or the scale.

### 4.7. Where a method intervenes

A new paper changes one place in the pipeline, and in the form $V=\mu^+-\mu^-$ there are six places. A paper is located by the place it moves; a gap is a combination of places that no paper or theorem covers.

| Place | What it sets | Failure it answers | Methods |
| --- | --- | --- | --- |
| Positive pairs | content: what counts as one object | easy feature wins (Prop. 2.6); constant distractor wins as the slowest feature | augmentations, time windows, masks |
| Negatives and temperature | weights of the repulsion | false negatives, shortcuts, temperature mismatch | DCL, hard negatives, FNC/IFND, IFM, PPS |
| Target law | shape: the law the cloud is pushed toward | collapse | InfoNCE, VICReg, SIGReg (Gaussian), SPHERE-JEPA (uniform) |
| Domain of the repulsion | the function of $z$ the law is imposed on | the static part fills the variance budget | SIGReg on $z$; TC-LeWM: SIGReg on the temporally centered residual |
| Prediction target | what the attraction predicts | temporal feature collapse | next embedding, frame differences (MotionJEPA), temporal differences (TDV) |
| Task signal | which semantics counts | semantics is not in the loss | labels, downstream-guided SSL (chapter 7) |

The domain is the place that 4.4 leaves implicit: shape says what law, content says what is attracted, and the domain says which part of $z$ is repelled.

*Prop. 4.6 (domain of the repulsion).* Let $P$ be a linear map from the sequence $Z=(z_1,\dots,z_T)$ to the residuals $R=PZ$, for example $r_t=z_t-\bar z_t$ over a window. For any repulsive loss $\mathcal L(R)$, the gradient $\nabla_Z\mathcal L$ is orthogonal to $\ker P$, the sequences that are constant within every window.
*Proof.* $\nabla_Z\mathcal L(PZ)=P^\top\nabla_R\mathcal L$, and $\operatorname{range}(P^\top)=(\ker P)^\perp$.
*Consequence.* A repulsion on a sum constrains the sum; a repulsion on the residual has no force on the persistent part. This is why TC-LeWM frees the residual, and also why nothing in its loss keeps the persistent part from collapsing.

> ==(1) The persistent part of TC-LeWM carries 47-67% of the variance in the reported runs, so partial collapse of the persistent part is not seen, but no term excludes it. Which regularizer on which function of $z$ prevents both failures? (2) MotionJEPA changes the prediction target and TC-LeWM the domain of the repulsion, and both aim at the dynamic part. Are they the same fix in two places, and what does the unification say about it?==

## 5. Reachability of semantics and limits of SSL

Semantics is the choice of the task family. It enters through augmentations, time, modality, architecture or labels and never through the loss, and each theory assumes one of these sources.

### 5.1. No ideal representation without a task family

Without a task family there is no best representation. Perfect alignment on a connected graph gives a constant, so semantics lives in the weak cuts of the graph.

*Prop. 5.1.* Given $\mathcal F$, set $x\sim_{\mathcal F}x'$ if $f(x)=f(x')$ for all $f\in\mathcal F$. The minimal sufficient representation is the quotient $\mathcal X/\!\sim_{\mathcal F}$.
*Proof.* $z$ is sufficient if every $f\in\mathcal F$ is a function of $z$, i.e. $z$ separates all classes of $\sim_{\mathcal F}$; minimal means coarsest such $z$.
*Consequence.* If $\mathcal F$ is all functions, $\sim_{\mathcal F}$ is equality and the representation must be injective. Any compression chooses which differences do not matter (shape vs colour). Semantics is the choice of $\mathcal F$.

*Prop. 5.2 (what augmentations choose).* If the loss reaches perfect alignment, $z$ is constant on connected components of $G$.
*Proof.* Perfect alignment gives $z(x)=z(x')$ on every positive edge; equality propagates along paths.
*Consequence.* If $G$ is connected, perfect alignment gives a constant. If it is nearly discrete, as for real augmentations (2.4.2), perfect alignment constrains nothing across images. In neither case does the zero eigenvalue carry semantics; it comes from weak cuts of $G$ together with the architecture.

*Prop. 5.3 (Cheeger).* For the normalized Laplacian, $\lambda_2/2\le h(G)\le\sqrt{2\lambda_2}$, where $h(G)=\min_S w(S,\bar S)/\min(\mathrm{vol}\,S,\mathrm{vol}\,\bar S)$.
Small $\lambda_2$ means one low-conductance cut. For $k$ classes Prop. 2.10 turns $\rho$ into a spectral gap with a higher-order Cheeger inequality, $\rho_k\le O(k^2)\sqrt{\lambda_k}$ (Lee, Oveis Gharan, Trevisan), HaoChen et al. use a version of it (their Lemma B.4).

*Prop. 5.4 (InfoMin, Tian et al.).* Views optimal for task $y$ share exactly the information about $y$: $I(v_1;v_2)=I(v_1;y)=I(v_2;y)$, for views that are minimal sufficient for $y$.
Augmentations are a manual way to say what the views share, so a good choice needs knowledge of $y$.

**Is SSL solved, since a network compresses and SGD finds a good minimum?** No. A good minimum gives good geometry, and which semantics that geometry carries is set by $G$ (Prop. 5.2) and the architecture. Compression keeps the right differences only if it knows which ones matter, which is the task family (Prop. 5.1).

### 5.2. What each line assumes

Each theory proves one thing and takes another for granted.

| Work | Proves | Assumes |
| --- | --- | --- |
| Wang-Isola | minimum of the limit functional = alignment + uniformity | $K\to\infty$; uniform law reachable by the encoder |
| Wang-Liu | $\tau$ redistributes pressure among negatives | loss form; no optimal $\tau$ |
| Saunshi 2019 | bounds via latent classes | positives independent within a class |
| Saunshi 2022 | ==bounds are vacuous without architecture== | ==architecture bias is an assumption== |
| HaoChen | bounds via augmentation graph | graph reflects classes |
| Zimmermann | latent recovery up to rotation | vMF positive, injective generator |
| Ziyin | landscape structure, linear models | linearity |
| Koromilas, AnInfoNCE | ==finite losses, factor anisotropy== | ==data-generating form== |
| LeJEPA | isotropy is worst-case optimal over tasks | unknown task; ridge and kNN probes |
| SPHERE-JEPA | ==uniform on the sphere is optimal for a kNN probe== | ==same, plus kNN== |
| InfoMin (Tian) | optimal views share information about $y$ | task $y$ known |
| LeJEPA world model (Klindt et al.) | linear recovery of the world's latent variables; the Gaussian is the only latent law for which this holds | stationary additive-noise transitions |

The theories differ only in which source of semantics they take as given.

### 5.3. More gaps

Predictable is not useful, identifiability is not planning, and good manifolds have no agreed definition.

- **5.3.1. Predictable is not useful.** By Prop. 3.3 prediction learns the conditional mean. A wall is easier to predict than gripper fingers, but control needs the fingers. Under partial observability the target is a belief state, and for a multimodal target the conditional mean falls between the modes, so prediction loses what control needs.
- **5.3.2. Identifiability is not planning.** Props 2.11 and 2.12: recovering the world up to rotation does not give coordinates meaningful for the task.
- **5.3.3. Labels or structure.** Do we need labels, or a structure that links observations (time, motion, actions, modalities)? **Why augmentations, if time neighbours can be pulled together?** Time closeness is a free graph $G$, but it needs video, and it fails when a constant distractor is the slowest feature (3.4).
- **5.3.4. Good manifolds.** ==A good manifold has low local complexity, smoothness with respect to the learned geometry, repeatability and locality, several scales, robustness to transformations, high effective rank and easy readout of important factors.== No single property defines a useful geometry, since smoothness only makes sense with respect to the right metric.

**What if a point in $\mathbb R^d$ is only a special case of a representation?** ==Token fields, hyperbolic and product manifolds, subspaces and distributions are candidates; no theory compares them yet.==

### 5.4. Three scales of next work

Three directions follow from the gaps: self-calibrating SSL (health constraints instead of hand-set weights), latent geometry derived from data ($G$ from unlabeled data), and predictive sufficiency (sufficiency, minimality, convenient coordinates).

### 5.5. What SSL does not understand

Six holes remain: why instance discrimination gives semantics, augmentations as a prior, BYOL, the projector, loss not predicting quality, which distribution is good.

"Push and pull" explains optimization, not generalization: the loss is on augmentation pairs, but we evaluate a linear probe on clean images from another layer (before the projector).

| Question | Known explanation and its weak spot | Where |
| --- | --- | --- |
| Why instance discrimination gives semantics | ==augmentation graph (HaoChen); graph is nearly discrete, assumption violated== | 2.4.2, 5.1 |
| Augmentations as a hidden prior | ==all "semantics" lives in the augmentation choice; no theory of the choice== | 2.3.5, 5.1 |
| Why BYOL/SimSiam do not collapse | ==several competing explanations, none closes it== | 3.1.2 |
| Projector (guillotine regularization) | ==a name for the effect: why features before the projector beat the ones optimized== | 5.5 |
| Loss does not predict quality | ==two runs with equal InfoNCE give different probes; RankMe, effective rank are palliatives== | 2.3.8, 6 |
| Which embedding distribution is good | ==every anti-collapse term answers implicitly; downstream result only in the worst case== | 3.1.5, 4.4 |

Method: take one question (collapse or temperature), answer it from three theories, write down where predictions differ. The disagreements map the unknown and show where the thesis can add something (9.2).

## 6. Score matching and diffusion

By Prop. 2.14 the drift of the contrastive loss is a score difference, so the question here is which score-matching problem the contrastive loss solves. ==Whether this drift is a Wasserstein flow, and whether stop-gradient freezes its target, is open.==

==Related drifting models: Deng et al. train a generator with drift $\mu^+-\mu^-$ (positives are data, negatives are generator output); Turan-Ovsjanikov show such drift is score matching in a spectral and variational setting.== Prop. 2.14 is the same fact for the contrastive loss.

### 6.1. Hyvarinen identity and the flow $V$

The Hyvarinen identity lets the score be learned without knowing the density; the zero of the drift field is equality of densities.

*Prop. 6.1 (Hyvarinen).* For fast-decaying $p,q$: $\mathbb E_p\|\nabla\log q-\nabla\log p\|^2=\mathbb E_p[\|\nabla\log q\|^2+2\Delta\log q]+\mathrm{const}$.
*Proof.* Expand the square; integrate the cross term $-2\int\langle\nabla\log q,\nabla p\rangle$ by parts to $2\int p\Delta\log q$ (boundary terms vanish). The rest is constant in $q$.
*Consequence.* The score can be learned without knowing $p$.

*Prop. 6.2 (from 2.14).* $V/\sigma^2=\nabla_z\log(\hat p^+/\hat p^-)$, and $V\equiv0$ on the whole space iff $\hat p^+=\hat p^-$.
*Proof.* $\nabla\log(\hat p^+/\hat p^-)\equiv0$ makes the ratio constant; both are normalized, so it is 1. Training reaches $V=0$ only at batch points, which is the stall problem of Arbel et al. discussed in 4.3.

### 6.2. Stop-gradient and a fixed target

With a frozen target, one training step may be a step of a Wasserstein flow, and stop-gradient is the freezing.

*Conjecture 6.3.* The $W_2$ gradient flow of $\mathrm{KL}(\mu\|\pi)$ moves mass with velocity $\nabla\log(\pi/\mu)$ (Jordan-Kinderlehrer-Otto). ==With $\mu=\hat p^-$ and $\pi=\hat p^+$ this velocity is $V/\sigma^2$, so an SGD step on the embeddings is an explicit Euler step of this flow and stop-gradient holds $\pi$ fixed.== SIGReg compares to a fixed $\mathcal N(0,I)$ and needs no such trick.

> ==The DriftSSL loss without a BYOL-like scheme collapses (2.6.8), so the Fisher link of 4.3 has to be rechecked before 6.3 can rely on it.==

### 6.3. Kernel width

Silverman's rule gives a kernel width from sample size and geometry. It is an analogy to PPS and no derivation, because the contrastive loss has no analogue of its criterion.

*Prop. 6.4 (Silverman, classical).* KDE integrated error is minimal at $h^*\propto n^{-1/(D+4)}$; Silverman's rule $h=\hat s(4/((D+2)n))^{1/(D+4)}$; error rate $n^{-4/(D+4)}$.
*Consequence.* The "right" temperature $\tau^*=h^{*2}$ is set by sample and geometry, not a grid. At large $D$ the $n$-dependence nearly vanishes and the data scale $\hat s$ dominates, consistent with PPS taking the cluster scale from the data.

> ==What does the InfoNCE kernel width minimize, that is, what is the contrastive analogue of AMISE (2.6.1, 2.6.8)?==

### 6.4. Generator on top of JEPA

JEPA has no generator. A generator trained by denoising score matching on the same latent would supply one without a separate decoder.

*Prop. 6.5 (Vincent).* Denoising score matching $\mathbb E\|s_\theta(\tilde x)-\nabla_{\tilde x}\log q_\sigma(\tilde x|x)\|^2$ and explicit score matching to the noised marginal $q_\sigma(\tilde x)=\int q_\sigma(\tilde x|x)p(x)\,dx$ differ by a $\theta$-independent constant.
*Proof.* Expand both squares; the cross terms agree because $\nabla q_\sigma(\tilde x)=\int p(x)\nabla q_\sigma(\tilde x|x)\,dx$.

==Whether a JEPA with a generator on the same latent keeps the quality of its representation is open (9.2).==

### 6.5. Training dynamics

**Why does SSL train many times longer than supervised learning?** Three hypotheses: (1) the target moves with the encoder (BYOL, DINO) or depends on negatives; (2) weaker signal per example: a pair instead of a label; (3) semantics comes from weak spots of $G$, which are learned late (==easy features first, Prop. 2.6==). ==Comparing representation trajectories of SSL and supervised learning under a moving target would separate them (9.2).==

### 6.6. Distances instead of coordinates

==In DriftSSL, describing data points through pairwise distances made the link to score matching fit much better than coordinates did.== This agrees with the softmax kernel depending only on distances (2.1.2). Writing the loss with a distance matrix would show whether Props 2.14 and 6.2 carry over with no extra assumptions.

## 7. Semi-supervised and downstream-guided SSL

If semantics is the choice of the task family $\mathcal F$ (Prop. 5.1), a task or a few labels can state that choice directly. By 2.3.8 there is no general embedding quality, so the task must enter training or at least measurement. ==What to measure and how to use it is open.==

### 7.1. Families

Five families use a task: classical semi-supervised (manifold, smoothness, low-density assumptions; FixMatch, ==S4L==, ==PAWS==); supervised contrastive; pretraining aware of the downstream task (BiSSL, V-pretraining, task-customized pretraining); continual learning with gradient projection (A-GEM, ==PCGrad==); invariance chosen by task (==AIL, ContextSSL==). Common condition: the structure of $p(x)$ is tied to $p(y|x)$.

> ==What does each method assume, and where do reports on them disagree (7.6.2)?==

### 7.2. Predictive information as a measure of structure

Predictive information measures structure in time and space. For Gaussians the best linear code follows canonical correlations, so small features like a ball are dropped for capacity reasons.

- **7.2.1. Time.** $\mathrm{PI}=I(x_{\le t};x_{>t})$ (Bialek-Nemenman-Tishby). MotionJEPA splits a static term on $z$ from a dynamic term on frame differences, with hand-set weights $\lambda_z,\lambda_d$.
- **7.2.2. Space.** $I(x_A;x_B)$ for two regions of one frame. Flat background and noise give $I\approx0$, objects and texture give large $I$. I-JEPA, MAE and InfoNCE already estimate such a quantity.

*Prop. 7.1 (CCA).* For a Gaussian pair $(x_A,x_B)$ the best $k$-dim linear $z=Wx_A$ for $I(z;x_B)$ is the projection on the top $k$ canonical directions, with $I=-\frac12\sum_{i\le k}\log(1-\rho_i^2)$.
*Consequence ("cheap bits").* A feature with small $\rho$, like a ball of a few pixels, is dropped at small $k$. It is not chaotic, it just carries little information per unit capacity. Large predictable features (colour, lighting, texture continuation) take capacity first. Augmentations and masks exist to take that role away from cheap bits.
A universal metric decides what counts as structure; the canonical correlations $\rho_i$ and the budget $k$ decide how much capacity each feature gets.

### 7.3. Task-weighted information

Information weighted by the task: $I(z;y)$, sensitivity, and the task subspace $G_T$ from about 30 labels.

- **7.3.1.** $I(z;y)$ (information bottleneck) and sensitivity $\partial y/\partial z$.
- **7.3.2.** For a linear probe with vectors $w_j$ set $G_T=\frac1k\sum_jw_jw_j^\top$ (in general $v_j=\partial\hat y_j/\partial z$). It is a positive semidefinite matrix whose range is the subspace the task reads; the projector onto that subspace is $P_T=G_TG_T^{+}$. The cost is labels (about 30).

### 7.4. Base construction: ridge probe and gradient projection

Project the SSL gradient away from the probe gradient (A-GEM); the ridge probe has a closed form. Let $g_{\rm task}$ be the probe-loss gradient and $g_{\rm ssl}$ the SSL gradient.

*Prop. 7.2 (A-GEM).* If $g_{\rm task}^\top g_{\rm ssl}<0$, set $g'=g_{\rm ssl}-\frac{g_{\rm task}^\top g_{\rm ssl}}{\|g_{\rm task}\|^2}g_{\rm task}$; then $g_{\rm task}^\top g'=0$ and the step does not worsen the probe loss to first order.
*Proof.* Substitute.
The signal $g_{\rm task}^\top g_{\rm ssl}$ matches V-pretraining (Ke-Fanti), which uses 1,024 GSM8K examples only as feedback to a task designer; here it corrects the step. The assembly is a strong baseline more than a new method. The ridge probe has a closed form $W=(Z^\top Z+\lambda I)^{-1}Z^\top Y$, so $g_{\rm task}$ is computed through $Z$ with no inner optimization.

### 7.5. Three contribution directions

Three directions are open: 30 labels instead of a hand-made prior, a task subspace in $z$ as a plugin, and diagnostics of what SSL washes out. The first has a baseline and a clear criterion, so it goes first.

1. **Labels instead of a hand-made prior.** Can 30 labeled ball positions give the same effect as a special frame-difference regularizer (MotionJEPA, DISReg: Pong, ball NMSE 0.005 vs 1.26 for the forward-only baseline)? Criterion: "30 labels vs DISReg".
2. **Subspace $G_T$ in $z$, not in parameters.** One object that plugs into SimCLR, JEPA, DINO without changing their loss, plus composition for several tasks. The task is held by a lower bound on information in $P_Tz$, not by adversarially removing everything else.
3. **Diagnostics.** Measure when and which information each SSL method washes out, on tasks with known factors.

### 7.6. What to measure and how to use task semantics

Three pieces turn the task into something measurable: what to measure, what the existing methods assume, and whether they transfer.

- **7.6.1. Measure.** Downstream accuracy is not enough; what matters is how the task changes the representation: per-feature contribution (sensitivity $\partial y/\partial z$, $G_T$), the task-relevant share of information in $z$, and how much task information SSL washed out.
- **7.6.2. Critique.** For each downstream-guided method: what it assumes (task, number of labels, domain), where reports disagree, what critics say. The result is an assumption table like 5.2.
- **7.6.3. Reproduction.** ==Reproduce 2-3 methods on their data, then transfer them to other fields (medical images, time series, audio).==

## 8. Application: semantics on small samples

This chapter applies Part I to one practical case: 30 unlabeled images, a small network trained from scratch with no external data, then a frozen encoder with a linear probe trained on a large labeled set. The setup separates representation quality from label scarcity: if the probe is bad with unlimited labels, the encoder geometry is the problem. ==How much data suffices is open.==

### 8.1. What theory says about this regime

Three facts shape the regime: semantics must be built in, the rank is at most $n-1$, and SIGReg constrains only the training points.

1. **Semantics must be injected (Props 5.1, 5.2).** At $n=30$ neither $G$ nor $\mathcal F$ can be learned; they must be built in: architecture, patches, allowed transformations.
2. **Rank is bounded.** The centered embeddings of $n$ images span at most $n-1$ directions, so 30 images fix at most 29 axes and cannot match an isotropic Gaussian in 128 dimensions. ==Do augmented views add task-relevant directions beyond these 29?==
3. **A regularizer does not extend a function (Prop. 3.4).** SIGReg checks the law only on training points; two functions with the same $\mathcal N(0,I)$ on 30 points can differ on the 31st image.

### 8.2. How to apply

Patches and built-in representations give a large graph from few images, and prediction of a hidden part keeps the SIGReg axes alive. From 1: replace 30 images by $10^5$-$10^6$ correlated patches, which give a large $G$ from few images (Models Genesis, MAE, Swin-MAE, ZSSR, sparse coding, patch k-means). Alternative: built-in representations (wavelet scattering, Deep Image Prior, random features). From 3: add prediction of a hidden part, which learns internal dependencies while SIGReg only keeps the axes alive. Expectation: SIGReg + prediction beats SIGReg alone. Experiments: 9.6.

Context: for segmentation 30 images suffice with dense labels and strong deformation augmentation (U-Net, nnU-Net). With a yes/no tumor label the effective $n\approx30$ and a flexible model does not beat a simple one. Few-shot segmentation solves a different problem: the prior is already in the weights.

### 8.3. How much data: information and sufficiency

Mutual information cannot be estimated from a small sample, so usable information replaces it, and density estimation is hopeless in raw space, so only the intrinsic dimension matters. ==How many points, and what diversity, suffice is open.==

- **Question.** How many points, and how diverse, give a representation sufficient for a task family. Diversity is relative to the information in the examples, not to the image count.
- **Information.** Mutual information cannot be estimated reliably from a small sample: any distribution-free high-confidence lower bound on mutual information from $n$ samples is at most about $\log n$ (McAllester-Stratos). Use usable information (V-information, Xu et al.), which accounts for the probe class, and information bottleneck as the "sufficient and minimal" frame.
- **Geometry.** With the optimal width, the mean integrated squared error of a KDE falls as $n^{-4/(D+4)}$; at $D=128$ it barely improves with $n$. So geometry cannot be estimated in raw space, and only the intrinsic dimension $m$ works (minimax manifold estimation, Genovese et al.). The practical question is how to estimate $m$ and coverage.
- **Empirics.** Size, diversity and domain of the dataset (Cole et al.), SSL on one image (Asano et al.), are large datasets necessary (El-Nouby et al.), example selection (Joshi-Mirzasoleiman), data pruning and scaling laws (Sorscher et al.), pretraining data diversity.

### 8.4. Modern small-data methods

Small-data methods replace contrastive losses with masking and patch prediction, internal learning, or JEPA. ==Which of them works best across domains and data types (medicine, time series, audio, molecules) under one protocol is open (9.6).==

## 9. Experiments

The experiments check the propositions of chapters 2-8 and are grouped by what they test: the Robinson reproduction, theory checks, PPS in other regimes, unification, downstream-guided SSL, and the small-sample application.

### 9.1. Robinson et al. reproduction

SimCLR with ResNet-18 on data with known factors: Trifeature (colour, shape, texture; Hermann-Lampinen) and STL-digits. The goal is to reproduce Fig. 2 (InfoNCE vs per-factor error), Fig. 3 ($\tau$-$\beta$ trade-off) and Fig. 6 (Implicit Feature Modification).

All runs use $\tau=0.5$, 200 epochs, batch 512, one training seed. On STL-digits, SimCLR and IFM-SimCLR ($\varepsilon=0.1$, $\alpha=0.5$) are read out by logistic regression on 5000 train and 8000 test images. Against random init, SimCLR raises test accuracy from 19.4% to 35.4% on STL10 and from 12.5% to 75.1% on MNIST; IFM gives 35.4% and 75.9%, so the joint improvement claimed by Robinson et al. is not reproduced. A SimCLR run on Trifeature (64 px) ends with a probe error of about 4% on colour, shape and texture on 1000 validation images. ==Reproducing Fig. 3 and Fig. 6 needs a sweep over $\tau$ and $\beta$, three seeds and several encoders.==

### 9.2. Theory checks

These experiments test the propositions of chapters 2-4 on toys: a shortcut toy, the Hessian at collapse, the graph spectrum, the ridge probe and temperature schedules.

| Experiment | What | Checks |
| --- | --- | --- |
| ==Shortcut toy== | easy feature $s$, useful $t$, sweep $K$, $\delta$; compare loss gap to $K\delta$ | Prop. 2.6 |
| ==Hessian at collapse== | sign and size of the smallest InfoNCE Hessian eigenvalue vs $\tau$ | Prop. 2.13 |
| ==Graph spectrum vs embeddings== | eigenvectors of $\bar A$ on a small graph vs SimCLR, VICReg, spectral-loss embeddings | Props 2.9, 3.1, 4.1 |
| ==Ridge probe vs anisotropy== | probe on embeddings of varied anisotropy at fixed trace | Prop. 3.2 |
| ==Convergence under $\tau$ schedules== | training speed for fixed vs adaptive $\tau$ (PPS) | 2.6.3 |
| ==Embedding quality vs world== | linear probe and retrieval vs "similarity in the world" on toy factors | 2.3.8 |
| ==Three theories, one question== | collapse or $\tau$ through Wang-Isola, Huang/PPS, LeJEPA: where predictions differ | 3.1.2, 5.5 |
| ==Invariant vs random vs equivariant augmentations== | three types on one downstream task | 2.3.5 |
| ==Representation trajectory== | SSL vs supervised under a moving target | 6.5 |
| ==JEPA with a generator on the same latent== | JEPA + generator tied to the same latent | 6.4 |

### 9.3. PPS in other regimes

The PPS rule is tested across seeds, architectures, methods and domains.

| Experiment | What | Checks |
| --- | --- | --- |
| ==Seeds, architectures, batches== | several seeds, ResNet-50, ViT, batch sizes | whether PPS holds beyond one configuration |
| ==Other methods== | MoCo, DINO, SupCon; non-contrastive BYOL, VICReg | where "kernel width = positive cluster size" works |
| ==Other domains== | medical images, time series, audio, small samples | PPS when $p$ is estimated from few pairs |
| ==How other papers show transfer== | collect transfer protocols from SSL papers | how protocols differ |
| ==Measure $q^*(\sigma)$== | inter-object coordinate dynamics at several fixed $\sigma$ | closing the dynamics, safe-to-optimal schedule |
| ==PPS width rule in target regularizers== | Epps-Pulley weight width and heat-kernel time from positive-pair distance | hypothesis in 4.3 |

### 9.4. Unification

Unification is tested through drift fields, stalls of the MMD flow, a drift monitor, the number of moments, the domain of the repulsion and the cost of capacity.

| Experiment | What | Checks |
| --- | --- | --- |
| ==Log-drift vs MMD-drift== | InfoNCE, SIGReg, SPHERE-JEPA on one sphere toy; compare $V$ fields | Prop. 4.2 |
| ==MMD-flow stalls for vMF== | theory (no stable stalls) plus a numeric check on the sphere | open piece of 4.3 |
| ==Drift monitor on SIGReg, SPHERE-JEPA== | $\|V\|^2$ as a label-free diagnostic; compare with KSD and Fisher divergence | $\|V\|^2=\sigma^4\hat F$, KSD link |
| ==KL-to-uniform vs InfoNCE== | compare InfoNCE repulsion with the KL variant of Expanding SPHERE-JEPA | statement on InfoNCE repulsion (4.3) |
| ==How many moments== | VICReg, Weak-SIGReg, SIGReg on data with controlled non-Gaussianity | 3.1.5 |
| ==Capacity cost== | shrink width or dimension, see where shape displaces content | 4.4 |
| ==Domain of the repulsion== | SIGReg on $z$ vs on the temporally centered residual, on a toy with a static and a dynamic factor; variance of each part and probe error | 3.4, Prop. 4.6 |
| ==Learned graph with cycle consistency== | $R_{ij}=h(x_i,x_j)$ instead of hand-made augmentations, penalty against $R=I$ | 4.5, 4.6 |

### 9.5. Semi-supervised and downstream-guided SSL

Here a few labels replace a hand-made prior, a task subspace plugs into existing methods, and downstream-guided methods are reproduced.

| Experiment | What | Checks |
| --- | --- | --- |
| ==30 labels vs DISReg== | Pong, ball probe, DISReg baseline | 7.5, direction 1 |
| ==Slow-features failure and DISReg== | reproduce the JEPA failure on constant noise and the DISReg fix on a known-factor toy | 3.4 |
| ==$G_T$ as a plugin== | $G_T$ from probe gradients into SimCLR, JEPA, DINO; two-task composition | 7.5, direction 2 |
| ==Washout diagnostic== | which factors each SSL method washes out, on a 3D sandbox (points on a sphere, few latent factors) | 7.5, direction 3 |
| ==Assumption table== | from papers, critics, blogs | 7.6.2 |
| ==Reproduction and transfer== | 2-3 methods on their data, then other fields | 7.6.3 |

### 9.6. Application: small sample

The protocol uses 30 images: first an untrained architecture, then within-image prediction, then SIGReg on 30 points, then a linear-accessibility check, then labels and a sample-size sweep.

| Experiment | What | Checks |
| --- | --- | --- |
| ==Random conv, untrained== | frozen random conv, linear probe | what locality gives without SSL |
| ==MAE with feature field== | MAE autoencoder, field $E(x)$ of size $H'\times W'\times d$, $1\times1$ probe | reconstruction and localization |
| ==SIGReg/VICReg/LeJEPA on the whole image== | regularizer on 30 points | points spread, tumor not encoded |
| ==Linear probe vs small MLP== | on random conv, MAE, whole-image regularizer | MLP wins: feature present, not linearized |
| ==MAE + var/cov or SIGReg on the latent== | MAE with a latent regularizer | probe vs plain MAE |
| ==$1\times1$ probe for localization== | field $E(x)$ and probe | whether the spot disappears without a tumor crop |
| ==Effective rank and SIGReg diagnostic== | on the same $z$ | high rank, bad probe: coordinates alive, tumor absent |
| ==Sample-size sweep== | $n\in\{30,100,300,1000\}$ for MAE, MAE + regularizer, whole-image regularizer | where global SSL starts to work; does it match $\mathrm{rank}\le n-1$ (8.1) |
| ==Labels on top== | $m\in\{0,10,30\}$ labels over MAE + regularizer | link of chapters 7 and 8 |
| ==How many points suffice== | usable information and coverage on a known-factor toy vs the geometric bound | 8.3 |
| ==Modern small-data methods across domains== | masking, internal learning, JEPA, one protocol | 8.4 |

Extensions: wavelet scattering; patch k-means or sparse coding; InfoNCE, DCL and graph $G$ from crop, time or dropout; Deep Image Prior on a 3D render; a 4-dim bottleneck with a shared transformation operator (identifiability toy); LieGAN and version space.

### 9.7. Visualizations

One shared 3D sandbox produces the figures of the theory sections:

| Section | Figure |
| --- | --- |
| 2.1 | ==sphere normalization; cosine vs distance slider; softmax weights of negatives vs $\tau$; InfoNCE vs DCL; Gaussian kernel on the sphere== |
| 2.5, 2.6 | ==Hessian saddle; particle flow and Lyapunov function; $\tau(p)$ of PPS vs $\tau(A)$ of Huang== |
| 3, 4 | ==drift field $V$ for InfoNCE, MMD and SIGReg in one picture; SIGReg cloud; the loop $z\to$ geometry $\to$ regularizer $\to z$== |

## References

### Contrastive learning and InfoNCE

- Chen, Kornblith, Norouzi, Hinton. A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML 2020.
- van den Oord, Li, Vinyals. Representation Learning with Contrastive Predictive Coding. 2018.
- Gutmann, Hyvärinen. Noise-contrastive estimation: a new estimation principle for unnormalized statistical models. AISTATS 2010.
- Wang, Isola. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. ICML 2020.
- Wang, Liu. Understanding the Behaviour of Contrastive Loss. CVPR 2021.
- ==Koromilas et al. Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses. ICML 2024.==
- Yeh et al. Decoupled Contrastive Learning. ECCV 2022. arXiv:2110.06848
- Chuang et al. Debiased Contrastive Learning. NeurIPS 2020. arXiv:2007.00224
- Robinson, Chuang, Sra, Jegelka. Contrastive Learning with Hard Negative Samples. ICLR 2021. arXiv:2010.04592
- ==Huynh et al. Boosting Contrastive Self-Supervised Learning with False Negative Cancellation. WACV 2022.==
- Chen et al. Incremental False Negative Detection for Contrastive Learning. ICLR 2022. arXiv:2106.03719
- Robinson, Sun, Yu, Batmanghelich, Jegelka, Sra. Can Contrastive Learning Avoid Shortcut Solutions? NeurIPS 2021. arXiv:2106.11230
- Hermann, Lampinen. What Shapes Feature Representations? Exploring Datasets, Architectures, and Training. NeurIPS 2020. arXiv:2006.12433
- ==Huang, Chen, Zhang, Shan. Model-Aware Contrastive Learning: Towards Escaping the Dilemmas. ICML 2023.==
- ==Kukleva et al. Temperature Schedules for Self-Supervised Contrastive Methods on Long-Tail Data. ICLR 2023.==
- ==Manna et al. DySTreSS: Dynamically Scaled Temperature in Self-Supervised Contrastive Learning. 2024.==
- ==Qiu et al. Not All Semantics are Created Equal: Contrastive Self-Supervised Learning with Automatic Temperature Individualization. ICML 2023.==
- Yuan et al. Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance. ICML 2022.
- Kim, Kim. A Temperature-Free Loss Function for Contrastive Learning. 2025.
- Zhang et al. Temperature as Uncertainty in Contrastive Learning. 2021.
- Zhang et al. Dual Temperature Helps Contrastive Learning Without Many Negative Samples. CVPR 2022.
- Khaertdinov, Asteriadis, Ghaleb. Dynamic Temperature Scaling in Contrastive Self-Supervised Learning for Sensor-Based Human Activity Recognition. 2022.
- Wang, Koniusz, Gedeon, Zheng. Adaptive Multi-Head Contrastive Learning. ECCV 2024.
- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
- Nie, Zhang, Mao. On the Inadequacy of Optimizing Alignment and Uniformity in Contrastive Learning of Sentence Representations. ICLR 2023.
- Fang, Li, Sun, Wang. Rethinking the Uniformity Metric in Self-Supervised Learning. ICLR 2024.
- ==Deng et al. Generative Modeling via Drifting. 2026.==
- ==Turan, Ovsjanikov. Generative Drifting is Secretly Score Matching: A Spectral and Variational Perspective. 2026.==
- ==Arbel, Korba, Salim, Gretton. Maximum Mean Discrepancy Gradient Flow. NeurIPS 2019. arXiv:1906.04370==
- ==Jing, Vincent, LeCun, Tian. Understanding Dimensional Collapse in Contrastive Self-supervised Learning (DirectCLR). ICLR 2022. arXiv:2110.09348==
- ==Bordes, Balestriero, Garrido, Bardes, Vincent. Guillotine Regularization: Why removing layers is needed to improve generalization in Self-Supervised Learning. arXiv:2206.13378==
- Positive-Pair Distance Schedules Temperature in Contrastive Learning Without a Grid Search. ICOMP 2026 (own work).
- He et al. Momentum Contrast for Unsupervised Visual Representation Learning (MoCo). CVPR 2020.
- Dwibedi et al. With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations (NNCLR). ICCV 2021.
- Caron et al. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV). NeurIPS 2020.
- Ermolov et al. Whitening for Self-Supervised Representation Learning (W-MSE). ICML 2021.

### SSL theory

- Saunshi et al. A Theoretical Analysis of Contrastive Unsupervised Representation Learning. ICML 2019.
- ==Saunshi et al. Understanding Contrastive Learning Requires Incorporating Inductive Biases. ICML 2022. arXiv:2202.14037==
- ==Wen, Li. Toward Understanding the Feature Learning Process of Self-supervised Contrastive Learning. ICML 2021. arXiv:2105.15134==
- HaoChen, Wei, Gaidon, Ma. Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss. NeurIPS 2021. arXiv:2106.04156
- Ziyin, Lubana, Ueda, Tanaka. What Shapes the Loss Landscape of Self-Supervised Learning? ICLR 2023.
- Cui et al. An Augmentation-Aware Theory for Self-Supervised Contrastive Learning. ICML 2025.
- Rusak et al. InfoNCE: Identifying the Gap Between Theory and Practice (AnInfoNCE). arXiv:2407.00143
- Reizinger, Balestriero, Klindt, Brendel. Position: An Empirically Grounded Identifiability Theory Will Accelerate Self-Supervised Learning Research. ICML 2025.
- Zimmermann et al. Contrastive Learning Inverts the Data Generating Process. ICML 2021.
- Locatello et al. Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML 2019.
- Mei, Montanari, Nguyen. A Mean Field View of the Landscape of Two-Layer Neural Networks. PNAS 2018.
- Chizat, Bach. On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport. NeurIPS 2018.

### Regularizers, JEPA, prediction

- Grill et al. Bootstrap Your Own Latent (BYOL). NeurIPS 2020. arXiv:2006.07733
- Chen, He. Exploring Simple Siamese Representation Learning (SimSiam). CVPR 2021.
- Zbontar et al. Barlow Twins: Self-Supervised Learning via Redundancy Reduction. ICML 2021.
- Bardes, Ponce, LeCun. VICReg. ICLR 2022. arXiv:2105.04906
- Bardes, Ponce, LeCun. VICRegL: Self-Supervised Learning of Local Visual Features. NeurIPS 2022.
- Balestriero, LeCun. Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods. NeurIPS 2022. arXiv:2205.11508
- Balestriero, LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544
- ==Klindt, LeCun, Balestriero. When Does LeJEPA Learn a World Model? arXiv:2605.26379==
- ==Maes et al. LeWorldModel (2026)==
- ==Liu, Suo, Jin, Ping, Iwasawa, Matsuo, Zhu. Temporally Centered SIGReg Improves Le World Model Representations for Robot Policy Learning (TC-LeWM). arXiv:2607.26924==
- ==SPHERE-JEPA (uniform target on the sphere instead of Gaussian). arXiv:2605.26900==
- ==Expanding SPHERE-JEPA (MMD, KSD, KL, heat kernel). arXiv:2606.17603==
- ==Weak-SIGReg (sketch-space projection, covariance to identity). arXiv:2603.05924==
- ==JEPA failure on slow features. arXiv:2211.10831==
- ==TDV (temporal differences). arXiv:2606.15956==
- ==Tian, Chen, Ganguli. Understanding Self-Supervised Learning Dynamics without Contrastive Pairs. ICML 2021. arXiv:2102.06810==
- Caron et al. Emerging Properties in Self-Supervised Vision Transformers (DINO). ICCV 2021.
- Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024.
- Epps, Pulley. A test for normality based on the empirical characteristic function. Biometrika 1983.
- Elhage et al. Toy Models of Superposition. 2022.
- Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR 2023. arXiv:2301.08243
- He et al. Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.
- Baevski et al. data2vec. ICML 2022.
- Garrido et al. RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank. ICML 2023.
- Lehner et al. Contrastive Tuning: A Little Help to Make Masked Autoencoders Forget (MAE-CT). AAAI 2024.

### Semi-supervised and downstream-guided SSL

- Chapelle, Schölkopf, Zien. Semi-Supervised Learning. MIT Press, 2006.
- van Engelen, Hoos. A survey on semi-supervised learning. Machine Learning, 2020.
- Belkin, Niyogi, Sindhwani. Manifold Regularization. JMLR 2006.
- Sohn et al. FixMatch. NeurIPS 2020.
- Khosla et al. Supervised Contrastive Learning. NeurIPS 2020.
- Chaudhry et al. Efficient Lifelong Learning with A-GEM. ICLR 2019.
- Tishby, Pereira, Bialek. The Information Bottleneck Method. 1999.
- Bialek, Nemenman, Tishby. Predictability, Complexity, and Learning. Neural Computation 2001.
- ==MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space (DISReg regularizer). arXiv:2609.23881==
- ==Zhai et al. S4L: Self-Supervised Semi-Supervised Learning. ICCV 2019. arXiv:1905.03670==
- ==Assran et al. Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples (PAWS). ICCV 2021. arXiv:2104.13963==
- ==Yu et al. Gradient Surgery for Multi-Task Learning (PCGrad). NeurIPS 2020. arXiv:2001.06782==
- ==Amortised Invariance Learning for Contrastive Self-Supervision (AIL). arXiv:2302.12712==
- ==In-Context Symmetries: Self-Supervised Learning through Contextual World Models (ContextSSL). arXiv:2405.18193==
- Ke, Fanti. Learning What to Predict: Downstream-Guided Task Design for Continued Pretraining (V-pretraining). arXiv:2601.22108
- Zakarias, Hansen, Tan. BiSSL: Enhancing the Alignment Between Self-Supervised Pretraining and Downstream Fine-Tuning via Bilevel Optimization. arXiv:2410.02387
- Liu et al. Task-Customized Self-Supervised Pre-training with Scalable Dynamic Routing. arXiv:2205.13267
- Przewięźlikowski et al. Augmentation-aware Self-supervised Learning with Conditioned Projector (CASSLE). Knowledge-Based Systems, 2024. arXiv:2306.06082
- Ericsson, Gouk, Hospedales. Why Do Self-Supervised Models Transfer? Investigating the Impact of Invariance on Downstream Tasks. arXiv:2111.11398
- Purushwalkam, Gupta. Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases. NeurIPS 2020. arXiv:2007.13916
- Kalibhat et al. Disentangling the Effects of Data Augmentation and Format Transform in Self-Supervised Learning of Image Representations. arXiv:2312.02205

### How much data: information, diversity, scale

- Cole et al. When Does Contrastive Visual Representation Learning Work? CVPR 2022.
- Asano, Rupprecht, Vedaldi. A Critical Analysis of Self-Supervision, or What We Can Learn From a Single Image. ICLR 2020.
- El-Nouby et al. Are Large-Scale Datasets Necessary for Self-Supervised Pre-training? 2021.
- Joshi, Mirzasoleiman. Data-Efficient Contrastive Self-Supervised Learning: Most Beneficial Examples for Supervised Learning Contribute the Least. ICML 2023.
- Sorscher et al. Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning. NeurIPS 2022.
- Kaplan et al. Scaling Laws for Neural Language Models. 2020.
- Xu et al. A Theory of Usable Information Under Computational Constraints. ICLR 2020.
- McAllester, Stratos. Formal Limitations on the Measurement of Mutual Information. AISTATS 2020.
- Tschannen et al. On Mutual Information Maximization for Representation Learning. ICLR 2020.
- Tian et al. What Makes for Good Views for Contrastive Learning? NeurIPS 2020.
- Ericsson, Gouk, Hospedales. How Well Do Self-Supervised Models Transfer? CVPR 2021.
- Shwartz-Ziv, LeCun. To Compress or Not to Compress: Self-Supervised Learning and Information Theory: A Review. Entropy 2024.
- Poole, Ozair, van den Oord, Alemi, Tucker. On Variational Bounds of Mutual Information. ICML 2019.
- Bansal, Kaplun, Barak. For self-supervised learning, Rationality implies generalization, provably. arXiv:2010.08508
- Lee, Lei, Saunshi, Zhuo. Predicting What You Already Know Helps: Provable Self-Supervised Learning. NeurIPS 2021. arXiv:2008.01064
- Ericsson, Gouk, Loy, Hospedales. Self-Supervised Representation Learning: Introduction, Advances and Challenges. IEEE Signal Processing Magazine, 2022. arXiv:2110.09327
- Korchinski, Favero, Wyart. Learn from your own latents and not from tokens: A sample-complexity theory. arXiv:2605.27734
- Hammoud et al. On Pretraining Data Diversity for Self-Supervised Learning. ECCV 2024. arXiv:2403.13808
- Vélez García, Cazorla, Pomares. Escaping The Big Data Paradigm in Self-Supervised Representation Learning (SCOTT). Computer Vision and Image Understanding, 2026. arXiv:2502.18056
- Yang et al. A Self-Supervised Paradigm for Data-Efficient Medical Foundation Model Pre-training: V-information Optimization Framework. arXiv:2408.07107
- Silverman. Density Estimation for Statistics and Data Analysis. Chapman and Hall, 1986.
- Fukunaga, Hostetler. The estimation of the gradient of a density function, with applications in pattern recognition. IEEE Trans. Information Theory, 1975.
- Jordan, Kinderlehrer, Otto. The variational formulation of the Fokker–Planck equation. SIAM J. Math. Analysis, 1998.
- Azizi et al. Big Self-Supervised Models Advance Medical Image Classification. ICCV 2021.
- Yue et al. TS2Vec: Towards Universal Representation of Time Series. AAAI 2022.
- Baevski et al. wav2vec 2.0. NeurIPS 2020.

### Small samples and internal statistics

- Ronneberger, Fischer, Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. arXiv:1505.04597
- Isensee et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 2021.
- Zhou et al. Models Genesis. Medical Image Analysis 2021.
- ==Xu et al. Swin MAE: Masked Autoencoders for Small Datasets. 2023.==
- Zontak, Irani. Internal Statistics of a Single Natural Image. CVPR 2011.
- Shocher, Cohen, Irani. "Zero-Shot" Super-Resolution using Deep Internal Learning (ZSSR). CVPR 2018.
- Ulyanov, Vedaldi, Lempitsky. Deep Image Prior. CVPR 2018.
- Olshausen, Field. Sparse coding with an overcomplete basis set: a strategy employed by V1? Vision Research 1997.
- Coates, Ng, Lee. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. AISTATS 2011.
- Bruna, Mallat. Invariant Scattering Convolution Networks. TPAMI 2013.
- Mallat. A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction. arXiv:1512.06293
- Rahimi, Recht. Random Features for Large-Scale Kernel Machines. NIPS 2007.
- Rahaman et al. On the Spectral Bias of Neural Networks. ICML 2019.
- Genovese et al. Minimax Manifold Estimation. JMLR 2012.

### Graphs, equivariance, hierarchy

- Belkin, Niyogi. Laplacian Eigenmaps for Dimensionality Reduction and Data Representation. Neural Computation 2003.
- Coifman, Lafon. Diffusion Maps. ACHA 2006.
- Coifman, Maggioni. Diffusion Wavelets. ACHA 2006.
- Singer, Wu. Vector Diffusion Maps and the Connection Laplacian. CPAM 2012. arXiv:1102.0075
- Bodnar et al. Neural Sheaf Diffusion. NeurIPS 2022.
- Hinton, Krizhevsky, Wang. Transforming Auto-Encoders. ICANN 2011.
- Cohen, Welling. Group Equivariant Convolutional Networks. ICML 2016.
- Thomas et al. Tensor Field Networks. 2018.
- Batzner et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials (NequIP). Nature Communications 2022.
- Yang et al. Generative Adversarial Symmetry Discovery (LieGAN). ICML 2023.
- ==Mitchel et al. Neural Isometries: Taming Transformations for Equivariant ML. NeurIPS 2024.==
- ==Park et al. Self-supervised Transformation Learning for Equivariant Representations. NeurIPS 2024.==
- Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions. NeurIPS 2025. arXiv:2510.21706
- Nickel, Kiela. Poincaré Embeddings for Learning Hierarchical Representations. NeurIPS 2017.
- Gu et al. Learning Mixed-Curvature Representations in Product Spaces. ICLR 2019.

### Score matching and diffusion

- Liu, Lee, Jordan. A Kernelized Stein Discrepancy for Goodness-of-fit Tests. ICML 2016.
- Lee, Oveis Gharan, Trevisan. Multiway Spectral Partitioning and Higher-Order Cheeger Inequalities. JACM 2014.
- Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. JMLR 2005.
- Vincent. A Connection Between Score Matching and Denoising Autoencoders. Neural Computation 2011.
- Song, Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS 2019.
- Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.
- Song et al. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021.
