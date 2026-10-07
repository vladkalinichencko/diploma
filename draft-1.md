# Inductive Biases for Semantic Representation Learning in Self-Supervised Learning

## 0. Central question

Self-supervised learning trains an encoder without labels by declaring some inputs equivalent. In every joint-embedding method one relation says which inputs should land close together, such as two augmentations of an image, two neighbouring frames of a video, or a visible context and its masked target, and one mechanism stops the encoder from satisfying that relation by sending every input to the same point. Contrastive methods build the mechanism from negatives, regularizer methods from statistics of the batch, and siamese methods from an asymmetric architecture with a predictor and a stopped gradient. SSL theory takes the relation and the mechanism as given and describes what representation they optimize. Neither the data nor the loss says which differences between inputs to keep or which geometry a later task needs, so those answers come from outside the loss: from the augmentations, the architecture, the target law of a regularizer, or a model of the tasks to come.

The thesis asks which inductive biases turn unlabeled data into a semantically useful representation, and it answers through one object. The gradient of each family moves every embedding toward what the relation calls the same and away from the rest of the cloud, so training is a flow of points under attraction and repulsion, and the three families differ only in how they build the repulsion (chapter 2). This flow obeys a few regularities seen across methods: collapse is a saddle, the temperature decides where the push goes, dimensions die where augmentation noise exceeds data variance, and easy features are learned first (chapter 3). Published theorems each explain one link of the chain from loss to downstream accuracy under their own assumptions, and they cannot be stacked into one result because they speak about different objects (chapter 4). Within each family the methods came in waves, each patching a failure of the previous one (chapter 5), and in all of them the attraction is the only term through which the relation, and with it the semantics, enters the loss (chapter 6). The second part applies this picture to one failure that all three families share, to SSL guided by a downstream task, and to SSL on a few dozen images (chapters 7-9). Chapter 10 collects the experiments that test the open claims.

Semantics does not follow from the loss, and every theorem buys a stronger conclusion with stronger assumptions. The contribution of this work is to write those assumptions down and test them. To understand a method here means to predict, before the run, what happens when the temperature, the batch size, the augmentation strength or the projector size change, and to call the prediction wrong when the run disagrees.

## 1. Setting and notation

An encoder $f:\mathcal X\to\mathbb R^D$ maps an input to an embedding, and most methods normalize it, $z=f(x)/\|f(x)\|\in S^{D-1}$. On the sphere $\|z-z'\|^2=2-2z^\top z'$, so dot products and distances carry the same information. Contrastive and siamese methods compute the loss on the output of a small projector placed after the encoder, while a downstream probe reads the encoder output before it; that output is written $h$ when the distinction matters. A batch of $N$ embeddings forms the rows of $Z\in\mathbb R^{N\times D}$.

The relation that declares inputs equivalent is a weighted graph $G$ on observations. A positive pair $(x,x^+)\sim p^+$ is two augmentations of one input, two neighbouring frames, or a context and its target. The edge weight is $W_{xx'}=p^+(x,x')$ and the degree $d_x=p_X(x)$ is the data marginal, from which negatives are drawn when a method uses them. The normalized adjacency is $\bar A=\mathcal D^{-1/2}W\mathcal D^{-1/2}$ with $\mathcal D=\mathrm{diag}(d_x)$, and the Laplacian is $L=I-\bar A$.

Two batch statistics recur throughout: the mean squared distance between positives, $p=\mathbb E\|z_i-z_{i^+}\|^2$, and between different inputs, $q=\mathbb E_{i\ne j}\|z_i-z_j\|^2$, with gap $g=q-p$. At initialization the embeddings are nearly orthogonal and $p\approx q\approx2$. A collapsed encoder has $p,q\to0$.

A linear probe on a frozen encoder measures whether information is linearly accessible, and information can be present while the probe misses it. Other measures used below are per-factor probe error, effective rank (the exponent of the entropy of the normalized singular values) and uniformity. A task is a map $y:\mathcal X\to\mathcal Y$, and $\mathcal F$ is the set of tasks the representation will serve; chapter 6 argues that semantics is the choice of $\mathcal F$.

## 2. One flow for all joint-embedding losses

Loss values of different methods cannot be compared, since InfoNCE is a cross-entropy, VICReg a weighted sum of squared errors and BYOL a prediction error. Their gradients can, because a gradient step moves every embedding by some displacement, and displacements of points in one space are the same kind of object for every method. This chapter writes each family as a velocity field on the embeddings and stays at the level of algebra on the loss; assumptions about data, networks and limits enter in chapter 4.

### 2.1. Attraction alone collapses

The simplest loss that uses only the relation is alignment, $\mathbb E\|z-z^+\|^2$. What this term selects on its own decides whether anything else is needed.

*Proposition 2.1.* If a loss reaches perfect alignment, $z$ is constant on every connected component of $G$.
*Proof.* Perfect alignment gives $z(x)=z(x')$ on every edge of $G$, and equality propagates along paths.

A constant encoder therefore reaches the global minimum of alignment. The minimum is also stable: around a collapsed state $z_a=c+\varepsilon u_a$ alignment grows by $\varepsilon^2\sum_i\|u_i-u_{i^+}\|^2\ge0$ in every direction. Every working method adds a second force that makes collapse costly, and the families differ in where that force comes from. Proposition 2.1 also bounds what the relation can teach. If $G$ is connected, perfect alignment gives a constant; if $G$ is nearly discrete, as it is for augmentations of different images, perfect alignment constrains nothing across inputs. Chapter 6 returns to this.

### 2.2. Contrastive repulsion is a kernel density estimate

Contrastive methods build the second force from negatives. For anchor $z$, positive $z^+$ and $K$ negatives $z_k^-$, InfoNCE is the cross-entropy of finding the positive among $K+1$ candidates,

$$
\mathcal L=-\frac{s^+}{\tau}+\log\Big(e^{s^+/\tau}+\sum_{k=1}^K e^{s_k/\tau}\Big),\qquad s^+=z^\top z^+,\ s_k=z^\top z_k^-.
$$

The temperature $\tau$ looks like a free scale. On the sphere it has a geometric meaning, and that meaning ties contrastive learning to density estimation.

*Proposition 2.2.* On $S^{D-1}$, $e^{z^\top z'/\tau}=e^{1/\tau}e^{-\|z-z'\|^2/(2\tau)}$, so the softmax weights of InfoNCE are the weights of a Gaussian kernel of width $\sigma=\sqrt\tau$.
*Proof.* Substitute $z^\top z'=1-\tfrac12\|z-z'\|^2$. The factor $e^{1/\tau}$ is common to all terms and cancels in the softmax.

The temperature is thus the squared size of the neighbourhood the loss treats as close. The kernel $\propto e^{\kappa z^\top y}$ is the von Mises-Fisher (vMF) bump with sharpness $\kappa=1/\tau$, and the denominator of InfoNCE is a kernel density estimate (KDE) of the embeddings, evaluated at the anchor. The next question is how the loss moves the anchor at this width.

*Proposition 2.3.* Let $S=e^{s^+/\tau}+\sum_k e^{s_k/\tau}$, $w_k=e^{s_k/\tau}/S$ and $w^+=e^{s^+/\tau}/S$. Then $\partial\mathcal L/\partial z_k^-=\tfrac{w_k}{\tau}z$, $\partial\mathcal L/\partial z^+=-\tfrac{1-w^+}{\tau}z$, and $\partial\mathcal L/\partial z=-\tfrac1\tau\big((1-w^+)z^+-\sum_kw_kz_k^-\big)$.
*Proof.* Differentiate the log-sum-exp.

A close negative carries a large $w_k$ and is pushed hardest, and the pull toward the positive carries the factor $1-w^+$, which vanishes once the positive outweighs the negatives. Regrouping the anchor gradient gives the form used for every family below.

*Proposition 2.4 (drift identity).* With $\sigma^2=\tau$, $\nabla_z\mathcal L=-\frac{1-w^+}{\sigma^2}V$, where $V=\mu^+-\mu^-$, $\mu^+=z^+$ (the centroid of positives when there are several views) and $\mu^-=\sum_kw_kz_k^-/(1-w^+)$ is the softmax-weighted centroid of the negatives. Equivalently, $V/\sigma^2=\nabla_z\log\big(\hat p^+(z)/\hat p^-(z)\big)$, where $\hat p^\pm$ are Gaussian KDEs of width $\sigma$ over the positives and the negatives.
*Proof.* The first form is Proposition 2.3 with $\sum_kw_k=1-w^+$ factored out. For the second, a Gaussian KDE $\hat p$ with kernel-weighted mean $m(z)$ satisfies $\nabla\log\hat p(z)=(m(z)-z)/\sigma^2$ (mean shift, Fukunaga-Hostetler). By Proposition 2.2 the kernel weights over negatives are $w_k/(1-w^+)$, so $m=\mu^-$ for $\hat p^-$ and $m=\mu^+$ for $\hat p^+$; subtracting the two scores removes $z$.

The gradient is taken in ambient coordinates, before the projection onto the tangent space of the sphere, and the realized update also collects terms from the losses of other anchors; $V$ is the anchor-term drift. Each anchor moves toward its positive and away from a softmax-weighted mean of its near negatives, and the step is a difference of two scores, gradients of log-densities at scale $\sigma$. The other two families are now written in the same form.

### 2.3. Regularizer repulsion acts on the batch as a whole

VICReg, Barlow Twins and SIGReg have no negatives. They penalize statistics of the whole batch, and the question is whether their gradients still split into attraction and repulsion. For regularizers that compare the batch with a target law the split is exact.

*Proposition 2.5 (MMD gradient as drift).* Let $z_1..z_N$ be embeddings, $y_1..y_M$ a sample from the target law, $k(z,y)=e^{-\|z-y\|^2/(2\sigma^2)}$, $\hat p_P=\frac1N\sum_lk(\cdot,z_l)$ and $\hat p_Q=\frac1M\sum_jk(\cdot,y_j)$ (unnormalized). Then
$$
-\nabla_{z_i}\widehat{\mathrm{MMD}}^2=\tfrac2N\big(\nabla\hat p_Q(z_i)-\nabla\hat p_P(z_i)\big).
$$
*Proof.* Expand $\widehat{\mathrm{MMD}}^2=\frac1{N^2}\sum k(z_l,z_{l'})-\frac2{NM}\sum k(z_l,y_j)+\mathrm{const}$ and use $\nabla_zk(z,y)=k(z,y)(y-z)/\sigma^2$.

The gradient attracts each embedding to the target sample and repels it from its own batch, the same attraction minus repulsion as $V$. The difference is the weighting: InfoNCE and a KL over a KDE use normalized weights and give $\nabla\log\hat p$, MMD uses unnormalized ones and gives $\nabla\hat p$, and the two are related by $\nabla\hat p=\hat p\,\nabla\log\hat p$. SIGReg, the regularizer of LeJEPA, pushes the embedding law to $\mathcal N(0,I)$ through random one-dimensional projections, which determine the law by the Cramer-Wold theorem, and scores each projection with the Epps-Pulley statistic, a weighted distance between the empirical and the Gaussian characteristic functions. That statistic turns out to be an MMD.

*Proposition 2.6 (Epps-Pulley is MMD on a slice).* For one-dimensional samples, $\int|\hat\varphi_N(t)-\varphi_0(t)|^2w(t)\,dt$ with a Gaussian weight $w$ equals $\mathrm{MMD}^2$ with a Gaussian kernel between the empirical and the target laws; the Epps-Pulley statistic differs by the factor $N$.
*Proof.* Expand the squared modulus with $\hat\varphi_N(t)=\frac1N\sum e^{itx_l}$ and integrate against $w$. By Bochner's theorem $\int e^{it(x-x')}w(t)\,dt=k(x-x')$ with $k$ Gaussian.

SIGReg is therefore a sum of MMDs over random slices, and SPHERE-JEPA, which replaces the Gaussian target with the uniform law on the sphere, is the same with another target. Both fall under Proposition 2.5, with the kernel width set by the weight $w$. ==Expanding SPHERE-JEPA averages over all slices analytically and should give MMD on the sphere with an induced kernel; this is unchecked.==

VICReg and Barlow Twins compare no target sample. Their variance and covariance terms act on the spectrum of the batch, and the spectral picture shows what they compute.

*Proposition 2.7 (HaoChen et al.).* The spectral contrastive loss $\mathcal L_{\rm spec}(F)=-2\mathbb E_{x,x^+}F(x)^\top F(x^+)+\mathbb E_{x,x'}(F(x)^\top F(x'))^2$ equals $\|FF^\top-\bar A\|_F^2$ up to a constant, after rescaling the rows of $F$ by $\sqrt{d_x}$.
*Proof.* Expand $\sum_{x,x'}\big(\sqrt{d_xd_{x'}}f_x^\top f_{x'}-W_{xx'}/\sqrt{d_xd_{x'}}\big)^2$ into quadratic, linear and constant parts.

*Proposition 2.8 (Ky Fan).* Under $Z^\top Z=I$, $\min\mathrm{tr}(Z^\top LZ)$ is attained by the bottom eigenvectors of $L$.
*Proof.* Rayleigh-Ritz for the sum of the $k$ smallest eigenvalues.

By Eckart-Young-Mirsky the best rank-$k$ fit of $\bar A$ is its top eigenvectors, the bottom ones of $L$, so both propositions end in a spectral embedding of $G$. VICReg is a soft version of Proposition 2.8, with the constraint $Z^\top Z=I$ replaced by variance and covariance penalties.

*Proposition 2.9 (Balestriero-LeCun).* Under conditions on linearity, batch size and augmentations, VICReg recovers Laplacian Eigenmaps on the positive graph, SimCLR recovers Kernel ISOMAP, and Barlow Twins is close to CCA.
*Proof sketch.* Each loss reduces to a spectral problem, by Ky Fan for the Laplacian and by Eckart-Young for a kernel matrix; the exact conditions are in the paper.

The spectral picture still separates "contrastive" from "non-contrastive" by what the penalty touches: the second term of $\mathcal L_{\rm spec}$ penalizes similarities between samples, while VICReg penalizes correlations between dimensions. One identity shows the separation is thin.

*Proposition 2.10 (after Garrido et al.).* Let $\mathcal L_{\rm sam}=\|ZZ^\top-\mathrm{diag}(ZZ^\top)\|_F^2$ penalize similarity between samples and $\mathcal L_{\rm dim}=\|Z^\top Z-\mathrm{diag}(Z^\top Z)\|_F^2$ penalize correlation between dimensions. Then $\mathcal L_{\rm sam}-\mathcal L_{\rm dim}=\sum_d\|Z_{:,d}\|^4-\sum_n\|z_n\|^4$.
*Proof.* $\|ZZ^\top\|_F^2=\mathrm{tr}(ZZ^\top ZZ^\top)=\mathrm{tr}(Z^\top ZZ^\top Z)=\|Z^\top Z\|_F^2$ by cyclicity of the trace. Removing the diagonals subtracts $\sum_n\|z_n\|^4$ from the first and $\sum_d\|Z_{:,d}\|^4$ from the second.

With unit-norm samples, as on the sphere, and standardized dimensions, as enforced by the variance term of VICReg, both norm sums are constant and the two penalties coincide. Repulsion between samples and decorrelation of dimensions are one quantity, and the boundary between the contrastive and the regularizer families is a choice of which axis to normalize. ==Garrido et al. state the equivalence for their criteria with specific normalizations; the conditions under which it carries over to the full SimCLR and VICReg losses, and its drift form, are unread.==

### 2.4. Siamese repulsion lives in the dynamics

BYOL and SimSiam train an online branch with a predictor $g$ against a target branch under a stopped gradient, with loss $\|g(z)-\mathrm{sg}(z')\|^2$ on normalized vectors; in BYOL the target is a moving average of the online network. A constant encoder with $g$ the identity has zero loss, so by Proposition 2.1 the loss alone admits collapse, and whatever prevents it is a property of the training dynamics. ==Tian, Chen and Ganguli show on a linear model that with a stopped gradient the predictor aligns with the eigenspaces of the correlation matrix of the embeddings, and that weight decay is needed for a stable non-trivial solution.== In the language of this chapter the repulsion term $\mu^-$ is absent from the loss and present in the update rule. ==Whether the pair of predictor and stopped gradient can be written as an implicit repulsion term, so that all three families share one flow equation, is the open half of the unification.== DINO avoids collapse by centering and sharpening the teacher output, which acts as repulsion from the batch mean; its details stay outside this work.

### 2.5. What the three families share

With the three forms side by side, the role of $\mu^-$ can be read off for each family, and a few properties hold for all of them regardless of how the repulsion is built.

| Family | What plays the role of $\mu^-$ |
| --- | --- |
| Explicit negatives (SimCLR, MoCo) | softmax-weighted centroid of negatives |
| Regularizer with a target law (SIGReg, SPHERE-JEPA) | the batch itself (self-repulsion); attraction goes to a sample of the target |
| Redundancy reduction (VICReg, Barlow Twins) | ==directions already covered by other dimensions (a reading through Proposition 2.10, not derived)== |
| Siamese and distillation (BYOL, SimSiam, DINO) | ==implicit in the update rule; centering for DINO== |

The first shared property concerns what the repulsion can see.

*Proposition 2.11.* Variance, covariance, $\mathcal N(0,I)$-target and uniformity terms depend only on the multiset of rows of $Z$; any relabeling of the inputs leaves them unchanged.
*Proof.* Each is a function of the empirical distribution of the rows.

Which input lands close to which is therefore set by the attraction term, that is by $G$, and by the inductive bias of the architecture; the repulsion keeps the space from degenerating and gives it no meaning. A function that is injective on the batch and invariant to the augmentations satisfies perfect alignment and every multiset repulsion at once, whatever it encodes. This yields a prediction that needs no data: a shortcut feature that supplies such a function should defeat contrastive and regularizer losses alike. Chapter 7 tests it.

The second property separates the shape of the embedding cloud from its content, so that the two can be discussed apart.

*Proposition 2.12.* For positive smooth densities $\mu,\nu$ on one connected manifold $M$ of dimension $m$ ($\mathbb R^m$ or a compact manifold such as the sphere) there is a homeomorphism $T:M\to M$ with $T_\#\mu=\nu$.
*Proof.* On $\mathbb R^m$, the Knothe-Rosenblatt map built from successive conditional distribution functions. On a compact manifold, Moser's theorem: two volume forms of equal total volume are related by a diffeomorphism.

A cloud can therefore change its law without gluing points or losing information. The repulsion sets the shape: an isotropic Gaussian for LeJEPA, the uniform law on the sphere for SPHERE-JEPA and for InfoNCE as $K\to\infty$. For shapes there are minimax theorems (5.3), which say that if the information is already in the embeddings, this shape hurts a probe least in the worst case over tasks. The attraction sets the content, what counts as one object, and has no theorem of that kind. The assumption of one manifold of dimension $m$ matters: if the data lie on a manifold of dimension below $D$, full isotropy in $\mathbb R^D$ cannot be reached without gluing, and the extra axes fill with noise or stay empty. A network has finite capacity, so shape and content compete, positives asking for clusters and the target law asking for spread. ==Where shape starts to displace content as the width or dimension of the network shrinks is open.==

The third property says when the flow stops.

*Proposition 2.13.* $V\equiv0$ on the whole space if and only if $\hat p^+=\hat p^-$; with a characteristic kernel this means the two laws are equal.
*Proof.* $\nabla\log(\hat p^+/\hat p^-)\equiv0$ makes the ratio constant, and both densities are normalized, so the constant is one.

Training evaluates $V$ only at batch points, so the flow can stop at a state where the field vanishes on the points while the cloud differs from its target (4.6). The drift also gives a label-free monitor: by Proposition 2.4, $\|V\|^2=\sigma^4\|\nabla\log(\hat p^+/\hat p^-)\|^2$, whose batch mean estimates $\sigma^4$ times a Fisher divergence, and by Propositions 2.5 and 2.6 the same quantity exists for SIGReg and SPHERE-JEPA. ==Its link to the kernelized Stein discrepancy (KSD) is unchecked.== The kernel width is the softmax temperature in InfoNCE, the weight width in Epps-Pulley and the time of a heat kernel on the sphere. ==The hypothesis that the positive-pair distance sets it in all three, as it does in PPS (4.5), is untested.==

A unifying language earns its place only if it predicts something that is not known without it. Proposition 2.11 gives one such prediction, tested in chapter 7, and the drift monitor gives one tool. Collapse structure, which also follows from the field, is the subject of 3.1.

### 2.6. The flow is a score difference

By Proposition 2.4, $V/\sigma^2=\nabla\log\hat p^+-\nabla\log\hat p^-$ is a difference of scores, the object that score matching learns and diffusion models integrate. Mean shift moves points along $\nabla\log\hat p$ at a fixed width (Fukunaga-Hostetler). Denoising score matching learns the score of a density smoothed by Gaussian noise, which is a KDE of width $\sigma$ when the noise has variance $\sigma^2$ (Vincent, 4.6). Drifting generators move generated samples by $\mu^+-\mu^-$ with data as positives and their own samples as negatives (Deng et al.); Turan and Ovsjanikov show that this drift is score matching, and Cao et al. write it as a Wasserstein gradient flow of a KDE-approximated divergence. In generation the data density is fixed. In SSL both clouds are outputs of the encoder, so the density the embeddings move toward moves with them. That one difference explains why siamese methods need a stopped gradient and why a gradient-flow reading of SSL needs a frozen target (Conjecture 4.14). Which proofs carry over between these models is decided by which of their assumptions survive this difference, and 4.6 sorts them.

The kernel of Proposition 2.2 depends only on distances, so the whole flow can be written in terms of the pairwise distance matrix of the batch. ==In DriftSSL, describing points by pairwise distances made the link to score matching fit much better than coordinates did; whether Propositions 2.4 and 2.13 carry over to a distance-matrix form without extra assumptions is open.==

## 3. Laws of motion

Chapter 2 says what pushes each embedding at one step. Training composes many steps through one shared encoder, and the same regularities appear across methods and datasets. Each has a local explanation and none has a global one. They are the facts a theory of SSL has to reproduce, and chapter 4 checks which theorem reproduces which.

### 3.1. Collapse is a saddle when repulsion is in the loss

Section 2.1 showed that collapse is a stable minimum of alignment. Adding negatives changes its type, and the change can be computed at the collapsed state itself.

*Proposition 3.1 (this work).* The state with all embeddings equal to $c$ is a critical point of InfoNCE with a negative Hessian eigenvalue.
*Proof.* Perturb $z_a=c+\varepsilon u_a$ with $u_a\perp c$ and write $d_{ab}=\|u_a-u_b\|^2$. On the sphere $s_{ab}\approx1-\tfrac{\varepsilon^2}2d_{ab}$. For anchor $i$ with candidate set $C_i$ (the positive and $K$ negatives)
$$
\mathcal L_i\approx\text{const}+\frac{\varepsilon^2}{2\tau}\Big(d_{ii^+}-\frac1{K+1}\sum_{j\in C_i}d_{ij}\Big).
$$
The first order vanishes. Take $u_i=u_{i^+}=v_i$ with independent $v_i$: then $d_{ii^+}=0$, the mean distance to the candidates is positive, and the quadratic form is negative.

Collapse is a strict saddle. The escape direction separates different inputs and keeps the two views of each input together, and its rate scales as $1/\tau$. A nonzero drift does not by itself forbid collapse: the collapsed state is stationary for InfoNCE and, for a target law symmetric about the collapse point (the uniform law on the sphere, a Gaussian at its mean), for the MMD field of Proposition 2.5, because attraction and repulsion cancel there. Instability is what rules it out. ==For MMD to the uniform law on the sphere the instability is unchecked.== Ziyin et al. find the same structure on linear models: contrastive and non-contrastive losses share their collapse critical points and differ in what stabilizes the non-trivial minimum.

### 3.2. The temperature decides where the push goes and when the pull stops

Proposition 2.3 gives the weights; how they redistribute with $\tau$ is the first law of the contrastive family.

*Proposition 3.2 (after Wang-Liu).* The entropy of the weights $w_k$ over negatives is non-decreasing in $\tau$.
*Proof.* For a Gibbs distribution $w_\beta\propto e^{\beta s}$, $dH/d\beta=-\beta\,\mathrm{Var}_{w_\beta}(s)\le0$, and with $\beta=1/\tau$, $dH/d\tau=\mathrm{Var}_w(s)/\tau^3\ge0$.

A small $\tau$ puts the push on the nearest negatives and a large $\tau$ spreads it. One might expect the softmax to repel similar inputs less, blurring them like an augmentation; the gradient says the opposite, because the weight of a negative grows with its similarity, and tolerance of close pairs comes only from a large $\tau$. Wang and Liu call the resulting trade-off the uniformity-tolerance dilemma: a small temperature spreads the cloud and also pushes apart different inputs of one class, which a class-level task wants close.

The pull has its own law. The factor $1-w^+$ of Proposition 2.3 vanishes once the positive outweighs the negatives, so the contrastive gradient switches itself off when the relative gap is won. Nie et al. call this gradient dissipation and show that optimizing alignment and uniformity as separate terms, which lacks it, keeps pushing absolute distances and underperforms InfoNCE on sentence embeddings. Decoupled contrastive learning (DCL) removes the same factor from the positive term and does better at small batch. ==The two observations pull in opposite directions; when switching off the pull helps and when it hurts is unreconciled.==

The batch is finite, and the loss computed on it is a biased estimate of the limit that most theory studies.

*Proposition 3.3 (delta method).* The bias of the batch loss relative to its $K\to\infty$ limit is $\approx-\mathrm{Var}(e^{s/\tau})/\big(2K(\mathbb Ee^{s/\tau})^2\big)$.
*Proof.* Second-order expansion of $\log$ around the mean (the Jensen gap).

The variance of $e^{s/\tau}$ grows as $\tau$ falls, so a small temperature needs a large batch, and gradient accumulation does not help because the logarithm sits inside the batch.

### 3.3. Positives contract, and the spread between inputs decides collapse

The two statistics $p$ and $q$ of chapter 1 behave differently in every PPS run on CIFAR-10, CIFAR-100 and ImageNet-100. The positive distance $p$ falls toward a floor $p_{\min}>0$ under every temperature schedule tried, because the two views are distinct augmentations and the encoder has finite capacity. The inter-input distance $q$ is the coordinate that can collapse, and the gap $g=q-p$ opens early only if the kernel is wide while the gap is still small. Section 4.5 derives the first half of this law in a reduced model and models the second half.

### 3.4. Repulsion does not keep dimensions alive

Spreading points on the sphere was expected to use the whole space, and Jing et al. show it does not. SimCLR embeddings avoid complete collapse while several singular values of their covariance fall to zero, so the cloud lives in a lower-dimensional subspace. In a one-layer linear model the weight dynamics is driven by a matrix equal to a weighted data covariance minus a weighted augmentation covariance; along directions where the augmentation varies more than the data, its eigenvalues are negative and the corresponding singular values decay. In a two-layer linear model collapse appears even with mild augmentations: gradient descent aligns adjacent weight matrices, after which singular values grow in proportion to themselves, small ones lag, and the product becomes effectively low-rank.

The uniformity of Wang and Isola does not register this. Fang et al. state four properties a uniformity metric should have (invariance to permuting inputs, to duplicating them, sensitivity to cloned features and to added constant dimensions) and prove that the Gaussian-potential uniformity violates three of them, while a Wasserstein-based metric satisfies all four. RankMe shows that the effective rank of the embeddings tracks downstream accuracy well enough to select hyperparameters without labels. Rank is a necessary condition for quality and falls short of a sufficient one: random features have full rank, and once the rank exceeds the intrinsic dimension of the task, accuracy saturates and extra rank can hold nuisance factors. ==Simon et al. find that Barlow Twins, SimCLR and VICReg learn eigen-directions one at a time in the order of their eigenvalues, so a weak direction may never be reached within the schedule.==

### 3.5. Easy features are learned first and can suppress the rest

Uniformity says the inputs should be spread and says nothing about which features spread them. When two features both separate inputs, the loss can be indifferent between them.

*Proposition 3.4 (after Robinson et al.).* Let $x$ carry an easy feature $s$ and a useful feature $t$, and let positives agree on both. A random negative shares the anchor's $s$ with probability $\delta$. If $g(s)^\top g(s')\le0$ for $s\ne s'$, the encoder $f=g(s)$ has loss at most $\log(1+K\delta+K(1-\delta)e^{-1/\tau})$. Using $t$ as well removes the $K\delta$ term and lowers the loss by $\log\frac{1+K\delta+Ke^{-1/\tau}}{1+Ke^{-1/\tau}}\le K\delta$.
*Proof.* Alignment is perfect. A negative with the same $s$ adds $e^0=1$ to the sum, and every other negative adds at most $e^{-1/\tau}$.

If $K\delta\ll1$, the gradient does not force the encoder to learn $t$, and a lower InfoNCE need not mean a lower per-factor error. Chen, Luo and Li observe the extreme case: a few bits of a feature shared by both views suppress the image features entirely, temperature and batch size barely help, BYOL suffers as much, and a VAE barely suffers. Xue et al. trace suppression to the simplicity bias of SGD and find that a larger embedding dimension and better augmentations reduce it. Wen and Li show that augmentations decorrelate dense features between positives and leave sparse ones intact, so contrastive learning extracts sparse features and needs stronger augmentations than supervised learning. ==Littwin et al. find that JEPA prefers features with a large regression coefficient and MAE features with large variance.== A narrow model has to drop something, and by Proposition 3.4 what it drops is decided by which features are cheapest; ==whether a small width alone selects semantics is untested (chapter 9).==

### 3.6. The loss shapes one space and the probe reads another

The loss acts on the projector output $z$, while the probe reads $h$ before the projector, and features before the projector are consistently better (guillotine regularization, Bordes et al.). DirectCLR removes the trainable projector and applies the loss to a subvector of $h$. In the PPS ablation the controller that maximizes the gradient norm collapses the projected geometry almost at once while kNN accuracy on $h$ still rises, because the encoder keeps extracting weak structure under a degenerate projection. Every statement of chapter 2 is about $z$ and reaches $h$ only through the projector, ==for which no theory exists beyond linear models.==

### 3.7. An adaptive temperature closes a feedback loop

For a fixed loss, gradient flow decreases it, $\dot{\mathcal L}=-\|\nabla\mathcal L\|^2\le0$, so the loss is a Lyapunov function of training; as $N\to\infty$ the particles become a PDE for the density on the sphere (Mei-Montanari-Nguyen, Chizat-Bach). An adaptive temperature computes $\tau$ from the current embeddings, so the embeddings set the temperature, the temperature sets the gradient, and the gradient moves the embeddings. The temperature and the embeddings thus form a closed loop in the sense of control theory, and the Lyapunov argument for a fixed loss does not survive it.

*Proposition 3.5.* Let $\tau=\tau(z)$ be computed from the batch and held fixed in the backward pass, so that $\dot z=-\nabla_z\mathcal L(z,\tau)$. Then $\frac{d}{dt}\mathcal L(z,\tau(z))=-\|\nabla_z\mathcal L\|^2+\partial_\tau\mathcal L\,\dot\tau$, and the second term has no fixed sign.
*Proof.* Chain rule.

Under an adaptive schedule the loss can rise during training without anything going wrong, and a stability argument has to use another function of the geometry. PPS uses $\Phi=p-p_{\min}$ (4.5). Every published adaptive-temperature rule is a controller in this loop, and 5.2 compares them as such. Analogies with physical particles (Debye or Yukawa potentials, phase transitions) help intuition and fail as models, since the particles are embeddings of one shared encoder and cannot move independently (4.1).

### 3.8. SSL trains longer than supervised learning

SSL needs many more epochs than supervised training on the same data, and three explanations compete. The target moves with the encoder (BYOL, DINO) or depends on the negatives. Each example carries a weaker signal, a pair instead of a label. Semantics comes from the weak cuts of $G$, which are learned late, ==as the order of features in 3.4 and 3.5 suggests.== ==Comparing representation trajectories of SSL and supervised learning under a moving target would separate the three (10.1).==

## 4. Theorems and their assumptions

The laws of chapter 3 are observations with local explanations. A theorem turns an explanation into a guarantee by fixing assumptions, and the strength of the guarantee depends on how far the assumptions are from a trained network. This chapter places each published result on one scale, states what it assumes, and shows why the results do not combine into one account.

### 4.1. How strong a proof is

A statement about SSL connects two points of the chain features, loss, gradient, trajectory, geometry, downstream accuracy. The weakest link between two points is a heuristic or an analogy. Next comes an exact identity, which holds at every step and says nothing about where training goes, such as the drift identity of Proposition 2.4. Above it are local stability of one state (Proposition 3.1), a Lyapunov region of a reduced model (the mean-field model of 4.5), and global convergence of a full model. The strongest level, a guarantee for a real network trained by SGD that ends at a representation optimal for a task, no published result reaches. The state of training has its own ladder: a configuration can be stationary, locally stable, globally optimal for the loss, and semantically useful, and each step up needs new assumptions.

Most theorems about SSL treat the embeddings as free particles that the gradient moves independently. A network moves them together, and the relation between the two flows is exact.

*Proposition 4.1.* Let the batch embeddings be $Z=F_\theta(X)$ and train $\theta$ by gradient flow on $\mathcal L(Z)$. Then $\dot Z=-\Theta\,\nabla_Z\mathcal L$, where $\Theta=JJ^\top$, $J=\partial\,\mathrm{vec}Z/\partial\theta$, is the empirical neural tangent kernel on the batch.
*Proof.* $\dot\theta=-J^\top\nabla_Z\mathcal L$ by the chain rule, and $\dot Z=J\dot\theta$.

The free-particle flow is the case $\Theta=I$. Since $\Theta$ is positive semidefinite, the loss still decreases, $\dot{\mathcal L}=-\nabla_Z\mathcal L^\top\Theta\nabla_Z\mathcal L\le0$, and every stationary point of the particle flow is stationary for the network. The converse fails: the network also stops where the gradient lies in the null space of $\Theta$, it moves fastest along the top eigenvectors of $\Theta$, and an escape direction from a saddle helps only if $\Theta$ does not annihilate it. The kernel is fixed during training only in the lazy regime (Jacot et al.); with feature learning it changes along the trajectory. A theorem about particles reaches a network only through an assumption on $\Theta$, so a minimizer of the loss over distributions is a statement about what the encoder would do if it could reach that distribution. The distance between such a minimizer and what the network reaches is the realizability gap, and the order in which features are learned (3.4, 3.5) lives inside it.

The ladder also explains why published theorems cannot be stacked into one result. Each proves one link of the chain for its own object: Wang and Isola a distribution on the sphere under a limit functional, HaoChen et al. a function minimizing a population loss on a graph, Tian et al. the weights of a linear network, PPS two scalars of a mean-field model, Saunshi et al. a classifier over latent classes. Stacking two theorems needs the conclusion of one to satisfy the assumptions of the next, and none of them does. The uniform minimizer of Wang and Isola and the spectral minimizer of HaoChen et al. are minimizers of two idealizations of the same SimCLR loss, and they are different points. The table in 4.7 lists, for each result, the link it proves and the assumptions it pays for it.

### 4.2. Wang and Isola: correct in the limit, silent about the path

Wang and Isola's analysis is the most influential reading of InfoNCE, and it still fails to cover all properties of the original loss. What the limit keeps, and what it drops, decides how much of chapter 3 it can explain.

*Proposition 4.2 (Wang-Isola, Thm 1).* As $K\to\infty$,
$$
\mathcal L-\log K\to-\tfrac1\tau\,\mathbb E\,z^\top z^+\;+\;\mathbb E_x\log\mathbb E_{x^-}e^{z^\top z^-/\tau}.
$$
*Proof.* Divide the denominator by $K$. The sum over negatives converges by the law of large numbers, since the integrand is bounded ($|s|\le1$), and $\log$ is continuous. The positive term $e^{s^+/\tau}/K$ vanishes. The random loss converges at rate $O_P(K^{-1/2})$ and its expectation at $O(1/K)$ (Proposition 3.3).

The first term is alignment. The second is uniformity, the log-mean of a Gaussian kernel, which is smallest for spread-out embeddings. Which configuration minimizes the pair is the next question.

*Proposition 4.3.* If an encoder exists with perfect alignment ($z=z^+$ a.s.) and $z$ uniform on the sphere, it is a global minimum of the limit functional (Wang-Isola). The second variation at the uniform law is positive in every direction (this work).
*Proof of the second part.* Take $\mu=(1+\varepsilon\varphi)\sigma_{\rm unif}$ with $\int\varphi=0$ and expand $\varphi$ in spherical harmonics. The kernel operator multiplies degree $\ell$ by $a_\ell$, and the increment of the uniformity term is $\varepsilon^2\sum_\ell(a_\ell/a_0-a_\ell^2/(2a_0^2))\|\varphi_\ell\|^2>0$, because $0<a_\ell\le a_0$ for the kernel $e^{t\,u^\top v}$ ($a_\ell\propto I_{\ell+(D-2)/2}(t)>0$ by Funk-Hecke, decreasing in $\ell$).

The statement is conditional on reachability, and Proposition 4.1 says this condition concerns $\Theta$, which the theorem does not model. The repulsion term also has an exact density form.

*Proposition 4.4.* At $K\to\infty$ the uniformity term equals $\mathbb E_p\log(p*k_\tau)+\text{const}=-H(p)-\mathrm{KL}(p\,\|\,p*k_\tau)+\text{const}$, where $p$ is the law of the embeddings and $k_\tau$ the vMF density.
*Proof.* $\mathbb E_{x^-}e^{z^\top z^-/\tau}=C_\tau(p*k_\tau)(z)$, and $\mathbb E_p\log q=-H(p)-\mathrm{KL}(p\|q)$.

The term is a KL divergence to the uniform law only as $\tau\to0$, where $-H(p)=\mathrm{KL}(p\|\mathrm{unif})-\log|S^{D-1}|$. ==Whether it matches the KL variant of Expanding SPHERE-JEPA is unchecked.==

The work that followed located six places where InfoNCE does more than its limit says. The limit keeps the minimizer and changes the gradient field, so the switch-off of the pull that Nie et al. found (3.2) is invisible to it. Uniformity separates all inputs, including inputs of one class that a task wants close, and only the temperature trades spread against this tolerance (Wang-Liu). Pairwise spread is blind to how many directions the cloud uses, so a representation with a good uniformity score can be low-rank (Jing et al.), and Fang et al. prove that the uniformity metric misses added zero dimensions and cloned features. Where augmentation noise exceeds data variance, the dynamics remove a direction whatever the limit prefers (3.4). The loss has many minimizers, and the one SGD finds is selected by simplicity bias (3.5). The loss acts on $z$ and the probe reads $h$ (3.6). Each of these gaps is a property of the finite batch, the temperature, the augmentations, the architecture or the optimizer, and none is a property of the limit functional.

The kernel width follows the same pattern. Read as functionals of kernel density estimates of the embeddings, alignment equals $p$, the first moment of the positive cluster, and contains no width; uniformity with the usual $t=2$ is the log Rényi-2 energy of the embeddings at the fixed width $h=1/\sqrt8$; and the drift acts at the training width $\sigma$ (PPS, Appendix C). Effective rank is a functional of the covariance matrix and lies outside this family, and in the PPS runs it is the metric that separates schedules while the density functionals move together. The theorem therefore fixes neither the training width nor the semantics. Wang and Isola is a zero-order model of contrastive learning: an equilibrium, scalar description that is right about which final state the loss rewards and silent about the path, the spectrum and the content. ==Koromilas et al. compare finite and asymptotic losses and kernel losses, and AnInfoNCE shows that the number of recovered factors does not fix downstream accuracy; both are unread in detail.==

### 4.3. Downstream guarantees pay for themselves with assumptions about classes

Wang and Isola describe what the loss prefers and never mention a task. Three lines of work obtain a downstream guarantee by assuming where the classes are.

*Proposition 4.5 (Saunshi et al. 2019).* If positives come from one latent class, negatives from the marginal, and $\theta^+$ is the probability that a negative has the anchor's class, the supervised loss of the class-mean classifier is $\lesssim\frac1{1-\theta^+}(\mathcal L_{\rm un}-\theta^+)$.
*Proof.* Convexity of the logistic loss (Jensen), a correction for class collisions, and a Rademacher bound.

The bound assumes that positives are independent draws from one class, which two augmentations of one image are not. HaoChen et al. replace the classes with the graph $G$ of chapter 1.

*Proposition 4.6 (HaoChen et al., Thm 3.8).* If labels are recoverable from augmentations with error $\alpha$, so that few edges of $G$ cross classes, and $\rho_{\lfloor k/2\rfloor}$ is the conductance of the sparsest partition of $G$ into $\lfloor k/2\rfloor$ parts, the linear-probe error of the population minimizer of $\mathcal L_{\rm spec}$ is $\tilde O(\alpha/\rho_{\lfloor k/2\rfloor}^2)$.

If classes are connected components of $G$, the zero eigenvectors of $L$ are class indicators and the probe is perfect; weak edges between classes perturb this (Davis-Kahan). The theorem assumes that the graph reflects the classes, which is the outside information of chapter 6. Augmentations of different images almost never overlap, so the real graph is nearly discrete and the method still works: the theorem survives and its explanation does not. ==Whether a guarantee exists for a soft graph, where closeness in a learned metric replaces a shared edge, is open.==

The third line asks whether the latent variables that generated the data can be recovered at all.

*Proposition 4.7 (Zimmermann et al.).* If latents $z$ are uniform on $S^{d-1}$, $p(\tilde z|z)\propto e^{\kappa z^\top\tilde z}$, the generator $g$ is injective and the encoder $h$ maps to the sphere, every minimizer of the contrastive loss at $K\to\infty$ satisfies $h\circ g=R$ with $R$ orthogonal.
*Proof.* At $K\to\infty$ the loss is, up to constants, the cross-entropy between the true vMF conditional and the model conditional $\propto e^{h(x)^\top h(\tilde x)/\tau}$ (Proposition 4.2). At its minimum $h\circ g$ preserves inner products, and such a map of the sphere is orthogonal.

*Proposition 4.8 (Locatello et al.).* Without assumptions on the model or the data, disentanglement is impossible.
*Proof sketch (Gaussian case).* If $z\sim\mathcal N(0,I)$, then $Rz$ has the same law for every rotation $R$, so the data cannot distinguish the coordinates of $z$ from those of $Rz$. For any factorized prior the same holds with nonlinear bijections.

Recovery up to a rotation is the best a loss without further assumptions can give, and Proposition 4.8 says even that requires an assumption. ==Saunshi et al. (2022) show that the graph-based guarantees become vacuous without the architecture, because a rich enough function class minimizes the loss without useful features; the architecture then belongs in the theory next to the graph, which is the subject of this thesis.== Wen and Li (3.5) give the complementary result for features: which ones survive depends on the augmentations. ==Whether both assumptions can be stated as weak cuts of $G$ (6.1) is open.==

### 4.4. Theorems about dynamics stop at linear models

Chapter 3 described the trajectory, and theorems that cover a trajectory exist only for reduced models. Tian, Chen and Ganguli analyse BYOL and SimSiam on a linear network and show that the predictor aligns with the eigenspaces of the embedding correlation matrix (2.4). Jing et al. derive dimensional collapse on one- and two-layer linear networks (3.4). Ziyin et al. map the loss landscape of linear models for contrastive and non-contrastive losses (3.1). Simon et al. solve the dynamics of Barlow Twins in a linear kernel setting and find stepwise learning of eigen-directions. ==Tian (2022) writes contrastive learning as coordinate-wise optimization of a max player over pair weights and a min player over the network, and for a deep ReLU network relates the min player to the leading eigenvector of a contrastive covariance matrix.== Mean-field theory (Mei-Montanari-Nguyen, Chizat-Bach) describes wide two-layer networks as a flow of a density and has not been applied to an SSL loss in the work reviewed here. All of these reach the Lyapunov or convergence level for a reduced model, and Proposition 4.1 marks what they leave out: the kernel $\Theta$ of a deep nonlinear encoder.

### 4.5. PPS: a chain of questions about one temperature

The Positive-Pair Schedule (PPS) is the author's own result, written up as a paper for ICOMP 2026, and the one piece of this thesis that goes from an identity to a rule tested in training. Its derivation is a sequence of questions, each forced by the answer to the previous one, and each answer sits at its own level of the ladder of 4.1.

The first question is what the temperature is, and Proposition 2.2 answers that $\tau=\sigma^2$ is the squared width of the kernel of a density estimate of the embeddings. The second is what the gradient does at that width, and Proposition 2.4 answers exactly that it moves each anchor by the drift $V=\mu^+-\mu^-$, a difference of scores at width $\sigma$. A width that follows the current geometry should then replace a grid search, and the natural target is the width that maximizes the gradient, since the gradient is what moves the representation.

*Proposition 4.9.* For an anchor-gradient proxy with isotropic Gaussian clusters and fixed centroids, the width that maximizes the gradient norm is $\sigma^{*2}=\max\big(0,((d^-)^2-(d^+)^2)/c_N-d_{\rm clust}^2\big)$ with $c_N=2(1+W(N_{\rm eff}/e))$, where $d^\pm$ are the distances from the anchor to the positive and negative centroids, $d_{\rm clust}$ the common cluster width, $N_{\rm eff}$ the effective number of negatives and $W$ the Lambert function.
*Proof.* PPS, Appendix A.

At initialization $d^+\approx d^-$, so $\sigma^*=0$, and an online controller built on this optimum collapses. A rule that optimizes the current step ignores that the width chosen now changes the geometry every later step sees. The third question is therefore which schedule $\sigma(t)$ keeps training away from collapse, and answering it needs a model of the trajectory.

*Proposition 4.10 (mean-field model).* Assume that the two views of an image induce nearly equal weighted negative terms, $(1-w_i^+)\mu_i^-\approx(1-w_{i^+}^+)\mu_{i^+}^-$ (A1), that the batch is spread on the sphere with mean near the origin (A2), and that the positive distance stays above a floor $p_{\min}>0$ (A3). Then $\dot p=-a\,\phi\,(p-p_{\min})$ with $a>0$ and $\phi=(1-w^+)/\sigma^2$, and $\Phi=p-p_{\min}$ is a Lyapunov function. The inter-input coordinate $q$ has a fixed point $q^*(\sigma)$ that rises continuously from zero with $\sigma$, and a schedule stays in the healthy region when its length scale is free of $\sigma$, large while the gap is small, and above zero.
*Proof.* PPS, Appendix B. Under A1 the shared negative terms cancel in $\dot p$. The evolution of $q$ has no closed form, so only its sign structure is derived and the remaining constants are modelled.

The model thus derives the first half of the law of 3.3, that positives contract, and models the second, that $q$ decides collapse. The model also gives the function that replaces the loss as a Lyapunov candidate under an adaptive temperature (Proposition 3.5). The fourth question is which length meets the safety condition, and four arguments give the same answer. The loss splits into alignment, which equals $p$ and contains no $\sigma$, and uniformity, which does; asking the uniformity kernel to resolve at the alignment scale sets $\sigma=\sqrt p$. Data-driven bandwidth rules of density estimation scale the width with the local spread of the data, and the spread of a positive cluster is $\sqrt p$. In mid-training, where the distances to the positive and negative centroids share a scale, the optimum of Proposition 4.9 is a multiple of $\sqrt p$. The only label-free length the objective defines that is free of $\sigma$ and still carries signal is $\sqrt p$, since the negative scale saturates at orthogonality and kernel-weighted scales bring $\sigma$ back. The schedule is $\tau(t)=p(t)$, large at initialization, where $p\approx2$, and above zero late in training because of the floor $p_{\min}$.

The density-estimation argument has a classical counterpart, and it transfers only as an analogy.

*Proposition 4.11 (Silverman).* The asymptotic mean integrated squared error of a KDE is minimal at $h^*\propto n^{-1/(D+4)}$; Silverman's rule is $h=\hat s(4/((D+2)n))^{1/(D+4)}$, and the error falls as $n^{-4/(D+4)}$.

At large $D$ the dependence on $n$ nearly vanishes and the data scale $\hat s$ dominates, which agrees with taking the scale from the positive cluster. The contrastive loss has no analogue of this error criterion. ==What the InfoNCE kernel width minimizes, the contrastive analogue of AMISE, is open.==

With SimCLR and ResNet-18 on CIFAR-10, CIFAR-100 and ImageNet-100, PPS reaches the useful regime of a tuned constant temperature without a search: on CIFAR-10 it matches the best constant (probe 75.62, kNN 80.74) and the adaptive schedules of Kukleva, Huang, Manna and Qiu, and it stays competitive on the larger datasets. In the rigor ladder, the identity is exact, the contraction of $p$ is a Lyapunov result inside a reduced model whose assumptions were checked empirically, the safety condition rests on a modelled sign structure for $q$, and $\sqrt p$ is a heuristic supported by four arguments. The model locates a safe region and leaves the optimum inside it to the encoder, whose effect on $q$ has no closed form. All runs used one seed, one architecture and one batch size. ==Whether PPS transfers to other seeds, architectures, batch sizes, methods and domains is untested (10.2), and whether the schedule changes convergence speed as well as the final accuracy is unmeasured.==

### 4.6. What transfers from score matching and generative drifting

Section 2.6 showed that the SSL flow is a difference of scores. Score matching has a mature theory, and the question is which of its results hold when the target density moves with the model.

*Proposition 4.12 (Hyvärinen).* For densities $p,q$ that decay fast enough, $\mathbb E_p\|\nabla\log q-\nabla\log p\|^2=\mathbb E_p[\|\nabla\log q\|^2+2\Delta\log q]+\mathrm{const}$.
*Proof.* Expand the square and integrate the cross term $-2\int\langle\nabla\log q,\nabla p\rangle$ by parts to $2\int p\,\Delta\log q$; the boundary terms vanish, and the remaining term does not depend on $q$.

*Proposition 4.13 (Vincent).* Denoising score matching $\mathbb E\|s_\theta(\tilde x)-\nabla_{\tilde x}\log q_\sigma(\tilde x|x)\|^2$ and explicit score matching to the noised marginal $q_\sigma(\tilde x)=\int q_\sigma(\tilde x|x)p(x)\,dx$ differ by a constant that does not depend on $\theta$.
*Proof.* Expand both squares; the cross terms agree because $\nabla q_\sigma(\tilde x)=\int p(x)\nabla q_\sigma(\tilde x|x)\,dx$.

A score can thus be learned without the density, and a Gaussian smoothing of width $\sigma$ is the same object as the kernel of Proposition 2.2. Both results are algebra on kernels and transfer to SSL unchanged. Results about where a flow ends need more.

*Conjecture 4.14.* The $W_2$ gradient flow of $\mathrm{KL}(\mu\|\pi)$ moves mass with velocity $\nabla\log(\pi/\mu)$ (Jordan-Kinderlehrer-Otto). With $\mu=\hat p^-$ and $\pi=\hat p^+$ this velocity is $V/\sigma^2$. ==The conjecture is that an SGD step on the embeddings is an explicit Euler step of this flow, and that a stopped gradient holds $\pi$ fixed.== SIGReg compares the batch with a fixed $\mathcal N(0,I)$ and needs no such device.

Convergence of such flows is proved for a fixed target, and even then not always. MMD is not displacement-convex, and Arbel et al. show that the MMD flow converges to its target only under extra conditions, so a state where the field vanishes at the particles while the cloud differs from the target cannot be excluded (Proposition 2.13). ==Whether stable stalls exist for the vMF kernel on the sphere is open; a proof that they do not would close one piece of the theory.== Drifting generators sit between the two settings: Deng et al. train a generator with the drift $\mu^+-\mu^-$, data as positives and generated samples as negatives; Turan and Ovsjanikov show that the drift is score matching; Cao, Wei and Liu write it as the Wasserstein gradient flow of a divergence approximated by KDE. In generation the data density is fixed and these convergence arguments apply. In SSL both clouds move, so exact identities transfer, convergence results transfer only with a frozen target (a stopped gradient, a moving-average teacher, or a fixed law as in SIGReg), and bandwidth rules transfer as analogies because the error criteria differ. ==JEPA has no generator, and a generator trained by denoising score matching on the same latent would supply one without a separate decoder; whether it keeps the quality of the representation is open (10.1).==

The DriftSSL project tested the transfer in practice and recorded where it fails. A loss written directly from the drift collapses almost at once without a BYOL-like scheme, because shrinking the norm of every vector pulls all points together, which is locally optimal; the drift identity holds throughout, so an identity alone does not prevent collapse, as Proposition 3.1 already says for any stationary point. Deriving $\sigma$ from the loss collapses at initialization for the reason of Proposition 4.9, and a moving average for stable statistics has not fixed it. A controller on the drift norm is unstable. The kernel argument extends from the Laplace law to all symmetric exponential laws, so Laplace is not special. The parts that preserve local structure match baselines without hand-tuned hyperparameters and do not beat them.

### 4.7. What each result assumes

Each line of theory proves one link of the chain of 4.1 and takes the rest as given.

| Work | Link of the chain | Proves | Assumes |
| --- | --- | --- | --- |
| Wang-Isola | loss → geometry | minimum of the limit functional is perfect alignment plus uniformity | $K\to\infty$; the uniform law is reachable |
| Wang-Liu | loss → gradient | $\tau$ redistributes the push among negatives | the loss form; no optimal $\tau$ |
| Saunshi 2019 | geometry → downstream | bounds through latent classes | positives independent within a class |
| Saunshi 2022 | features → downstream | ==graph bounds are vacuous without the architecture== | ==an architecture bias== |
| HaoChen | loss → downstream | bounds through the augmentation graph | the graph reflects classes |
| Zimmermann | loss → geometry | latent recovery up to rotation | vMF positives, injective generator |
| Ziyin; Jing; Tian 2021; Simon | gradient → trajectory | landscape, dimensional collapse, predictor alignment, stepwise learning | linear networks |
| Tian 2022 | gradient → trajectory | ==min player learns the leading eigenvector of a contrastive covariance== | ==coordinate-wise optimization; ReLU network== |
| Koromilas; AnInfoNCE | loss → geometry | ==finite losses; anisotropy of factors== | ==a data-generating form== |
| PPS | gradient → trajectory | Lyapunov contraction of $p$; a safe width schedule | A1-A3; a modelled closure for $q$ |
| LeJEPA | geometry → downstream | isotropy is worst-case optimal over tasks | unknown task; ridge and kNN probes |
| SPHERE-JEPA | geometry → downstream | ==the uniform law on the sphere is optimal for a kNN probe== | ==the same, plus kNN== |
| Klindt-LeCun-Balestriero | loss → geometry | linear recovery of world latents; the Gaussian is the only latent law for which it holds | stationary additive-noise transitions |
| InfoMin (Tian et al.) | features → downstream | optimal views share exactly the information about $y$ | the task $y$ is known |

No row covers the link from features to trajectory for a deep network, which is where Proposition 4.1 places the realizability gap, and no row says which semantics the representation carries. The rows differ in which source of semantics they take as given.

## 5. Three groups of implementations

Chapters 2-4 describe one flow and its theory. The methods that implement it arrived in waves, each fixing a failure of the previous one, and they fall into three groups by how they build the repulsion. Sorting them this way shows which failure each wave answered and which law of chapter 3 it was fighting.

### 5.1. Families and waves

The first wave (CPC, SimCLR, MoCo, 2018-2020) used explicit negatives. The second (BYOL, SimSiam, DINO, 2020-2021) removed negatives and relied on an asymmetric architecture. The third (Barlow Twins, VICReg, 2021-2022) replaced the asymmetry with statistics of the batch, and the fourth (I-JEPA, LeJEPA, SPHERE-JEPA, 2023-2026) moved the attraction to prediction in latent space and the repulsion to a target law. Each method has three parts: where $G$ comes from (augmentations, time, masks), what prevents collapse, and what is predicted.

| Family | Methods | What prevents collapse | Negatives |
| --- | --- | --- | --- |
| Explicit negatives | SimCLR, MoCo, NNCLR | InfoNCE denominator | batch, momentum queue, memory bank |
| Siamese and distillation | BYOL, SimSiam, DINO, I-JEPA | asymmetry: moving-average teacher and predictor (BYOL), centering and sharpening (DINO) | implicit |
| Regularizers | VICReg, Barlow Twins, LeJEPA, SPHERE-JEPA | variance and covariance terms; a target law | implicit |
| Related | SwAV (online clustering, Sinkhorn), W-MSE (whitening and MSE), DirectPred and DirectCLR (spectral control instead of a trained predictor) | partition constraint, whitening, spectrum | implicit |

Which of these count as contrastive depends on the definition. With an explicit negative denominator, SimCLR, MoCo, NNCLR and DirectCLR are contrastive. With attraction and repulsion both present, nearly every method is, since DINO repels through centering, VICReg and Barlow Twins through decorrelation and W-MSE through whitening, and Proposition 2.10 makes the boundary a matter of normalization. Pure BYOL and SimSiam remain outside, because their repulsion is not in the loss. One representative per group is enough for experiments: SimCLR, BYOL, VICReg or SIGReg.

### 5.2. Contrastive methods

The first wave justified InfoNCE through mutual information.

*Proposition 5.1 (CPC).* For any critic, $I(x;x^+)\ge\log(K+1)-\mathcal L$; the bound is tightest for a critic $\propto p(x^+|x)/p(x^+)$.
*Proof.* InfoNCE is the Barber-Agakov bound with a critic self-normalized over $K+1$ samples (Poole et al.).

The bound saturates at $\log(K+1)$ (McAllester-Stratos), and maximizing information does not by itself give good representations (Tschannen et al.), so the later waves dropped this justification and kept the loss. The patches inside the family each answer a law of chapter 3. Decoupled contrastive learning (Yeh et al.) removes the positive from the denominator, which removes the factor $1-w^+$ from the pull (3.2) and keeps the $K\to\infty$ limit; it does better at small batch, where the factor weakens the pull exactly when the negatives are easy. Debiased contrastive learning (Chuang et al.) assumes a fraction $\theta^+$ of the negatives shares the anchor's class and subtracts their estimated contribution with a floor of $e^{-1/\tau}$. Hard-negative sampling (Robinson et al.) draws negatives $\propto e^{\beta s}p(x^-)$ by importance sampling. FNC (Huynh et al.) and IFND (Chen et al.) use the current encoder to mark negatives close to the anchor as false negatives and drop or attract them, so a shared background can pass for a shared class. Implicit feature modification (Robinson et al.) perturbs embeddings to remove the shortcut of Proposition 3.4.

The temperature rules are the latest patches, and by Proposition 3.5 each is a controller in a feedback loop. Read through $\sigma=\sqrt\tau$, each tracks one length, and the safety condition of Proposition 4.10 sorts them by whether that length is free of $\sigma$ and large early.

| Work | Length tracked | Signal | Level | Free of $\sigma$ | Large early |
| --- | --- | --- | --- | --- | --- |
| Kukleva et al. | cosine in the epoch | time only | global | yes | yes, then periodically small |
| Manna et al. (DySTreSS) | bounded function of the pair cosine | pair similarity | pair | yes | no |
| Huang et al. (MACL) | $\sqrt{\tau_0(1+A/2)}$, $A=1-p/2$ | alignment | global | yes | no; grows as positives tighten |
| Qiu et al. (iSogCLR) | per-anchor $\tau_i$ holding the softmax entropy at $\log k$ | negative distribution | anchor | no | no |
| Khaertdinov et al. | ==$\tau$ from an external autoencoder== | external model | pair | ==unchecked== | ==unchecked== |
| Wang et al. (AMCL) | ==$\tau_{ij}^{(c)}$ per head and pair, by regularized likelihood== | likelihood over heads | head and pair | ==unchecked== | ==unchecked== |
| PPS | $\sqrt p$ | positive-pair geometry | global | yes | yes |

Huang et al. move in the opposite direction to PPS: their temperature grows as positive pairs tighten. Qiu et al. derive $\tau_i$ as the dual variable of a distributionally robust objective, which makes them the closest in spirit to PPS, since both derive the temperature from a principle; they differ in the principle (robustness against kernel geometry) and the level (anchor against global). Because Qiu's signal passes through the current softmax, it is the one rule that can feed back on itself. Kim's temperature-free loss and the learned temperature of CLIP stand apart. ==Whether any work sets the width inside the uniformity term by density estimation, separately from the softmax temperature, is unknown.==

### 5.3. Regularizer methods

The regularizer waves replaced negatives with statistics of the batch, and the question each wave answers is how much of the law to constrain. VICReg minimizes $\lambda s(Z,Z')+\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')]$, where $s$ is the MSE between views, $v$ a hinge on the standard deviation of each coordinate and $c$ the sum of squared off-diagonal covariances; Barlow Twins pushes the cross-correlation matrix of the two views to the identity. By Propositions 2.8 and 2.9 these are soft spectral embeddings of $G$. Both fix two moments and leave everything above free, so embeddings may have heavy tails or several modes at zero VICReg loss. LeJEPA constrains the whole law through SIGReg (2.3) and justifies the Gaussian target by the downstream risk of a probe on an unknown task.

*Proposition 5.2 (simplified LeJEPA argument).* Take a ridge probe with parameter $\lambda$, an unknown task direction $w$ uniform on the sphere, and embedding covariance eigenvalues $s_j$ with fixed $\sum_js_j$. The expected bias $\frac{\|w\|^2}{d}\sum_j\big(\frac{\lambda}{s_j+\lambda}\big)^2$ is minimal at $s_1=\dots=s_d$.
*Proof.* $s\mapsto(\lambda/(s+\lambda))^2$ is convex; apply Jensen.

The variance term of the ridge risk, $\sum_js_j/(s_j+\lambda)^2$, is concave for $s_j<2\lambda$, so the Jensen step covers only the bias. ==The full LeJEPA argument uses Fisher information instead.== The assumption behind it is that no direction is more useful than another: with $\eta(z)=\mathbb E[y|z]$, LeJEPA assumes $\mathbb E[\nabla\eta\nabla\eta^\top]\propto I$, and Proposition 5.2 assumes the same through $w$ uniform on the sphere. The result is a minimax statement about shape (2.5), optimal in the worst case over tasks; when the task is known, an anisotropic law can be better, and isotropy erases global scale structure. Against contrastive methods, uniformity on the sphere is also a full target law, but it comes from the asymptotics of the loss, has no downstream derivation, and needs negatives or large batches. LeJEPA drops the stopped gradient, the teacher and the schedules and keeps one hyperparameter. ==Its reported results are 79% linear probe on ImageNet-1k with ViT-H/14, stability across dozens of architectures and a shorter schedule than I-JEPA, comparable to DINOv2 with fewer components; these are unreproduced here.== The augmentations, the architecture and the choice of layer stay outside the argument.

==SPHERE-JEPA replaces the Gaussian target with the uniform law on the sphere, with the same random projections. Its argument is that a Gaussian density is non-uniform, so kNN neighbourhoods are anisotropic, and the worst case over tasks for a kNN probe gives uniform density. In high dimension a projection of the uniform law is close to $\mathcal N(0,1/D)$, so per direction SIGReg nearly equals uniformity and the difference is mostly the radius. Expanding SPHERE-JEPA integrates the projections analytically into a family of sphere regularizers (MMD, KSD, KL over a KDE; heat and band-limited kernels), in which MMD and KSD give local clusters and KL gives instance separation.== Klindt, LeCun and Balestriero ask when LeJEPA learns a world model and find linear recovery of the latents only for Gaussian latents with isotropic stationary transitions. ==How many moments a regularizer needs is open. If only two matter, the gap between VICReg and LeJEPA should vanish where embeddings are nearly Gaussian; Weak-SIGReg, which projects into a sketch space and pushes the covariance to the identity, was tested only as a supervised stabilizer (10.3).==

### 5.4. Siamese and distillation methods

The siamese wave removed the repulsion from the loss and kept collapse away through the training procedure (2.4). BYOL trains an online network with a predictor against a moving-average target under a stopped gradient, SimSiam drops the moving average, DINO uses a teacher whose outputs are centered and sharpened, and I-JEPA and data2vec predict masked targets in latent space against a moving-average encoder. Three explanations of why these methods do not collapse compete, and none closes the question. In the first, the predictor and the stopped gradient align the predictor with the eigenspaces of the correlation matrix (Tian et al.), and DirectPred sets the predictor from that spectrum directly and matches a trained one. In the second, normalization layers center the batch implicitly and act as a hidden repulsion. In the third, the moving-average teacher changes slowly enough that the online network chases a non-collapsed target. DirectCLR (Jing et al.) applies the loss to a subvector of $h$ with no trainable projector and keeps the representation full-rank, which ties this group to the projector gap of 3.6. ==Where the three explanations make different predictions is unmapped; the disagreements would show what SSL does not understand (10.3).==

### 5.5. Where a method intervenes

A new paper changes one place in the pipeline, and in the form $V=\mu^+-\mu^-$ there are six places. A paper is located by the place it moves; a gap is a combination of places that no paper or theorem covers.

| Place | What it sets | Failure it answers | Methods |
| --- | --- | --- | --- |
| Positive pairs | content: what counts as one object | easy feature wins (Proposition 3.4); constant distractor wins as the slowest feature | augmentations, time windows, masks |
| Negatives and temperature | weights of the repulsion | false negatives, shortcuts, temperature mismatch | DCL, hard negatives, FNC/IFND, IFM, PPS |
| Target law | shape: the law the cloud is pushed toward | collapse | InfoNCE, VICReg, SIGReg (Gaussian), SPHERE-JEPA (uniform) |
| Domain of the repulsion | the function of $z$ the law is imposed on | the static part fills the variance budget | SIGReg on $z$; TC-LeWM: SIGReg on the temporally centered residual |
| Prediction target | what the attraction predicts | temporal feature collapse | next embedding, frame differences (MotionJEPA), temporal differences (TDV) |
| Task signal | which semantics counts | semantics is not in the loss | labels, downstream-guided SSL (chapter 8) |

The domain is the place that 2.5 leaves implicit: shape says what law, content says what is attracted, and the domain says which part of $z$ is repelled.

*Proposition 5.3 (domain of the repulsion).* Let $P$ be a linear map from the sequence $Z=(z_1,\dots,z_T)$ to the residuals $R=PZ$, for example $r_t=z_t-\bar z_t$ over a window. For any repulsive loss $\mathcal L(R)$, the gradient $\nabla_Z\mathcal L$ is orthogonal to $\ker P$, the sequences that are constant within every window.
*Proof.* $\nabla_Z\mathcal L(PZ)=P^\top\nabla_R\mathcal L$, and $\operatorname{range}(P^\top)=(\ker P)^\perp$.

A repulsion on a sum constrains the sum; a repulsion on the residual has no force on the persistent part. Proposition 5.3 thus explains why TC-LeWM frees the residual, and also why nothing in its loss keeps the persistent part from collapsing. ==The persistent part of TC-LeWM carries 47-67% of the variance in the reported runs, so partial collapse of the persistent part is not seen, but no term excludes it. Which regularizer on which function of $z$ prevents both failures is open. MotionJEPA changes the prediction target and TC-LeWM the domain of the repulsion, and both aim at the dynamic part; whether they are the same fix in two places, and what the unification says about it, is open.==

## 6. Where semantics enters

Every method of chapter 5 shapes the cloud, and by Proposition 2.11 the repulsion cannot decide which inputs end up close. This chapter asks where that decision comes from, and the answer is that it comes from outside the loss: from the relation $G$, the architecture, or a task.

### 6.1. No ideal representation without a task family

A representation is good only for some tasks, and the set of tasks fixes what it may discard.

*Proposition 6.1.* Given $\mathcal F$, set $x\sim_{\mathcal F}x'$ if $f(x)=f(x')$ for all $f\in\mathcal F$. The minimal sufficient representation is the quotient $\mathcal X/\!\sim_{\mathcal F}$.
*Proof.* A representation $z$ is sufficient if every $f\in\mathcal F$ is a function of $z$, that is, if $z$ separates all classes of $\sim_{\mathcal F}$; minimal means the coarsest such $z$.

If $\mathcal F$ contains all functions, $\sim_{\mathcal F}$ is equality and the representation must be injective. Any compression chooses which differences do not matter, shape over colour or the reverse, and semantics is the choice of $\mathcal F$. Proposition 2.1 says what the relation $G$ chooses: perfect alignment makes $z$ constant on the components of $G$, so on a connected graph it gives a constant and on a nearly discrete graph it constrains nothing across images. In neither case does the zero eigenvalue carry semantics, which therefore comes from the weak cuts of $G$ together with the architecture.

*Proposition 6.2 (Cheeger).* For the normalized Laplacian, $\lambda_2/2\le h(G)\le\sqrt{2\lambda_2}$, where $h(G)=\min_S w(S,\bar S)/\min(\mathrm{vol}\,S,\mathrm{vol}\,\bar S)$.

A small $\lambda_2$ means one cut of low conductance. For $k$ classes the higher-order Cheeger inequality $\rho_k\le O(k^2)\sqrt{\lambda_k}$ (Lee, Oveis Gharan, Trevisan) turns the conductance of Proposition 4.6 into a spectral gap, and HaoChen et al. use a version of it (their Lemma B.4). Augmentations are the manual way to place these cuts, and the best placement depends on the task.

*Proposition 6.3 (InfoMin, Tian et al.).* Views that are minimal sufficient for a task $y$ share exactly the information about $y$: $I(v_1;v_2)=I(v_1;y)=I(v_2;y)$.

A good choice of augmentations therefore needs knowledge of $y$. A network that compresses well and an optimizer that finds a good minimum give good geometry, and which semantics that geometry carries is still set by $G$ (Proposition 2.1), by the architecture, and by the task family (Proposition 6.1).

### 6.2. Augmentations are hidden supervision

Requiring $f(x)\approx f(Tx)$ for every augmentation $T$ declares $x$ and $Tx$ equivalent, and the transitive closure of this relation partitions $G$ into the components of Proposition 2.1. Cui et al. write the downstream risk as a function of the augmentation choice: strong augmentations reduce within-class variance and also erase task features. By Proposition 6.3 the optimal views share exactly the information about the task, so a rule for choosing augmentations needs the task. ==Invariant, random and equivariant augmentations have not been compared on one task (10.1).==

### 6.3. Predicting a target does not choose what is useful

JEPA, MAE and data2vec replace the second view with a target to predict, and what a squared prediction loss learns is classical.

*Proposition 6.4.* The minimizer of the squared prediction loss of a target $y$ from a context $c$ is $\mathbb E[y|c]$.
*Proof.* $\mathbb E\|y-g(c)\|^2=\mathbb E\|y-\mathbb E[y|c]\|^2+\mathbb E\|\mathbb E[y|c]-g(c)\|^2$.

In pixel space with a multimodal target the prediction is the mean of the modes, which is the blur of MAE. In latent space (I-JEPA, data2vec) the target encoder can drop what is unpredictable, and then a regularizer or a moving average is needed to keep the target from collapsing. Reconstruction keeps information that need not be linearly readable (MAE-CT, superposition), and VICRegL applies the same losses to local features. Prediction learns what the context makes predictable, and predictable is not the same as useful. A wall is easier to predict than gripper fingers and control needs the fingers; under partial observability the target is a belief state, and for a multimodal target the conditional mean falls between the modes, so prediction loses what control needs. Sobal et al. show the extreme case: a constant distractor is the slowest and most predictable signal, and a JEPA with a slow-feature objective prefers it to the useful one (chapter 7).

Recovering the generating factors does not help either. By Propositions 4.7 and 4.8, the best a loss gives is the world up to a rotation, and a rotation of the latents is not a set of coordinates meaningful for a task. The open question is whether labels are needed or a structure that links observations is enough: time, motion, actions, modalities. Closeness in time is a free graph $G$ and needs video, and it fails when a constant distractor is the slowest feature.

### 6.4. A graph stores invariance, and structure needs more

A graph with one weight per edge stores only how similar two inputs are, so it can teach invariance. Equivariance needs an operator $R_{ij}$ per edge, and a consistency condition decides whether the operators define a global frame.

*Proposition 6.5 (connection Laplacian).* Let $G$ be connected, take orthogonal $R_{ij}\in O(d)$ with $R_{ji}=R_{ij}^{-1}$, and the energy $\mathcal E(z)=\sum w_{ij}\|z_i-R_{ij}z_j\|^2$. If the product of the $R$ along every cycle is $I$ (for a triangle, $R_{ij}R_{jk}R_{ki}=I$), then $R_{ij}=g_ig_j^{-1}$ for some $g_i\in O(d)$ and $\min\mathcal E=0$ at a non-trivial $z$; at $R\equiv I$ this is the usual Laplacian.
*Proof.* Pick a spanning tree, set $g_{\rm root}=I$ and extend $g_j=R_{ij}^{-1}g_i$ along its edges; cycle consistency makes the extension well defined.

Cycle consistency is thus the condition for a global coordinate system, and it can serve as a penalty on learned operators (10.3). Without consistency one gets Vector Diffusion Maps and sheaf networks, and hierarchy needs several scales, as in diffusion wavelets. Exact equivariance (G-CNN, tensor field networks, NequIP) gives a large gain when the symmetry is known and exact. ==Learned symmetries (LieGAN, Neural Isometries, Self-supervised Transformation Learning, Equivariance by Contrast, and nearby ContextSSL and AIL) are unread in detail.== A learned equivariance has its own collapse: for $\sum\|F(x)-R\,F(Tx)\|^2$ the choice $F\equiv0$, $R\equiv I$ gives zero, so it needs variance or whitening terms, reconstruction, or a penalty against $R=I$. Each method fixes something in advance, whether the neighbourhood, the positive pairs, the symmetry family, the graph or the scale.

Graph theory is a second language for the same questions and a side topic of this thesis. HaoChen et al. read contrastive learning as spectral clustering on the augmentation graph. ==Tan et al. (2024) prove that InfoNCE performs spectral clustering on a similarity graph, and Wang et al. (2022) explain the success of contrastive learning on a nearly discrete graph through augmentation overlap between images of one class ("chaos is a ladder"); both are unread in detail, and whether they close the soft-graph question of 4.3 is open.==

### 6.5. Quality is defined through the task

There is no general quality of an embedding. Proxies such as alignment, uniformity and RankMe are many, none suffices (4.2, 3.4), and quality is the match between the representation and the task family $\mathcal F$. Measuring it against the world requires fixing $\mathcal F$ on a toy with known factors and comparing the proxies with probe error and retrieval ==(10.1)==. ==A good manifold may have low local complexity, smoothness with respect to the learned geometry, repeatability and locality, several scales, robustness to transformations, high effective rank and easy readout of important factors; no single property defines a useful geometry, since smoothness depends on the metric.== ==Token fields, hyperbolic and product manifolds, subspaces and distributions are candidate alternatives to a point in $\mathbb R^d$, and no theory compares them yet.==

### 6.6. What SSL does not yet explain

"Push and pull" explains optimization and leaves generalization unexplained: the loss acts on augmentation pairs, and evaluation is a linear probe on clean images from another layer.

| Question | Known explanation and its weak point | Where |
| --- | --- | --- |
| Why instance discrimination gives semantics | ==augmentation graph (HaoChen); the graph is nearly discrete, so the assumption fails== | 4.3, 6.1 |
| Augmentations as a hidden prior | ==all semantics lives in the augmentation choice, and there is no theory of the choice== | 6.2 |
| Why BYOL and SimSiam do not collapse | ==several competing explanations, none closes it== | 2.4, 5.4 |
| The projector | ==guillotine regularization names the effect and does not explain why features before the projector are better== | 3.6 |
| The loss does not predict quality | ==two runs with equal InfoNCE give different probes; RankMe and effective rank are partial remedies== | 3.4, 6.5 |
| Which embedding law is good | ==every anti-collapse term answers implicitly; downstream results exist only in the worst case== | 2.5, 5.3 |

Each question can be answered from several theories at once, and the places where their predictions differ map what is unknown (10.3). Three directions follow from the gaps. Self-calibrating SSL replaces hand-set weights with health constraints on the geometry, as PPS does for the temperature. Latent geometry derived from data builds $G$ from unlabeled data instead of hand-made augmentations. Predictive sufficiency asks for a representation that is sufficient for a task family, minimal, and readable in convenient coordinates. The second part of the thesis applies this picture to three cases where the needed semantics must be known in advance.

## 7. One failure shared by the three families

Proposition 2.11 predicts, before any run, that a feature supplying an injective, augmentation-invariant function of the input should defeat contrastive and regularizer losses alike, because such a function satisfies perfect alignment and every multiset repulsion without encoding anything useful. If the prediction holds, one task can show where all three families lose semantics, and the same task can test whether a downstream signal repairs the loss. Such a task would also touch the three conditions under which semantics is hardest to obtain: few samples, structure that lives between samples, and a downstream goal that is unknown during pretraining.

### 7.1. Searching for the case

The first candidate was the shortcut setting of Robinson et al.: SimCLR with ResNet-18 on Trifeature (colour, shape, texture; Hermann-Lampinen) and STL-digits, with $\tau=0.5$, 200 epochs, batch 512 and one seed. It did not isolate a failure. On STL-digits, SimCLR raised logistic-regression accuracy against random initialization from 19.4% to 35.4% on STL-10 and from 12.5% to 75.1% on MNIST, so both features were learned; implicit feature modification gave 35.4% and 75.9%, and the joint improvement claimed by Robinson et al. did not reproduce. On Trifeature the probe error ended near 4% for all three factors, and colour is linearly available already at initialization. A useful test needs a shortcut whose strength can be set exactly and whose presence does not help the probe.

RandBit (Chen, Luo, Li) provides it. Each STL-10 image (32×32) receives $b$ extra channels holding a fixed random $b$-bit code of the image, constant over pixels. The augmentations change only the RGB channels, so both views carry the same bits. The bits are useless for the class and solve instance discrimination once $2^b$ is large compared with the batch, so $b$ sets one quantity, how much information the shortcut supplies. SSL runs on 20,000 unlabeled STL-10 images, and a probe is trained on 5,000 labeled images and evaluated on 8,000 test images, each with its own random bits. The encoder has four convolutional blocks (64, 128, 256 and 512 channels) and global average pooling, $h\in\mathbb R^{512}$, with a projector 512→512→128 whose output the losses read. The augmentations are those of SimCLR. The losses are InfoNCE ($\tau=0.2$), VICReg (weights 25, 25, 1) and SIGReg in the LeJEPA form (Epps-Pulley over 256 random projections, $\lambda=0.05$), trained with Adam at learning rate $10^{-3}$, batch 256, for 15 epochs with seed 0. The readouts are logistic regression on the class from standardized $h$ and the fraction of bits recoverable from $h$ by ridge regression.

### 7.2. All three losses erase the image, at different thresholds

Linear accuracy on the STL-10 test set without labels during pretraining:

| $b$ | 0 | 2 | 4 | 6 | 8 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| initialization | 0.391 | 0.358 | 0.316 | 0.285 | 0.258 | 0.191 |
| InfoNCE | 0.622 | 0.575 | 0.525 | 0.409 | 0.106 | 0.097 |
| VICReg | 0.601 | 0.534 | 0.123 | 0.105 | 0.097 | 0.103 |
| SIGReg | 0.569 | 0.499 | 0.398 | 0.217 | 0.102 | 0.100 |

At large enough $b$ all three losses reach the chance level of 0.1 and fall below the untrained encoder at the same $b$, so training actively removes the image from $h$, while the bits are read from $h$ with 97-100% accuracy at every $b>0$. The prediction of Proposition 2.11 holds for the contrastive and the regularizer families. The threshold differs between losses, and it agrees with how many distinct values each repulsion needs. VICReg breaks at 4 bits, since sixteen codes placed along the axes already satisfy the variance hinge and the decorrelation. SIGReg falls below the untrained encoder at 6 bits and reaches chance at 8, because each projection must look Gaussian, which needs more distinct values than unit variance. InfoNCE stays above the untrained encoder at 6 bits and reaches chance at 8. In a batch of 256 images each anchor has 510 negatives, and a negative carries the anchor's code with probability $2^{-b}$, so about 8 negatives share the code at 6 bits, 2 at 8 bits and 0.008 at 16 bits; these are the $K\delta$ term of Proposition 3.4, and only they make the image worth learning. ==The order of the thresholds, VICReg first, then SIGReg, then InfoNCE, rests on one seed and 15 epochs.==

### 7.3. Labels help only when the shortcut cannot satisfy them

The same task tests downstream guidance (chapter 8). At $b=16$, 100 labeled images (10 per class) enter pretraining either jointly, as a cross-entropy of a linear head on $h$ added to the SSL loss, or through A-GEM (Proposition 8.2), which projects the SSL gradient so that it does not oppose the label gradient. A control trains on the labels alone. In a further variant the labeled images receive fresh random bits at every step, so the cross-entropy cannot be satisfied by memorizing the code of each of the hundred images.

| | no labels | joint | A-GEM | joint, fresh bits |
| --- | --- | --- | --- | --- |
| InfoNCE | 0.097 | 0.104 | 0.098 | 0.386 |
| VICReg | 0.103 | 0.103 | 0.099 | 0.097 |
| SIGReg | 0.100 | 0.098 | 0.100 | 0.508 |
| labels only | – | 0.112 | – | 0.518 |

With fixed bits neither joint training nor A-GEM recovers anything, and the control explains why: labels alone give 0.112. A hundred labeled images are memorized through their unique 16-bit codes, so the cross-entropy is itself satisfied through the shortcut and its gradient tells the SSL gradient nothing about the image. A-GEM is useless here by construction, since it projects one shortcut gradient against another. When the labels cannot be satisfied through the bits, the losses separate. SIGReg with labels reaches 0.508, close to labels alone (0.518), so the regularizer neither hurts nor helps. InfoNCE with labels reaches 0.386, so the SSL loss pulls against the labels. VICReg with labels stays at 0.097, because the regularizer fills $h$ with the bits so that a linear head on a hundred images cannot recover the class. Labels alone at $b=0$ give 0.562. Guidance works only when the downstream signal is not itself solved by the shortcut. On RandBit this condition had to be created by hand; on real data it means that the labeled sample must cover the variation of the shortcut feature.

### 7.4. The same failure without artificial channels

Natural data show the same failure wherever one feature is cheaper than the useful one. Sobal et al. place a moving dot on a noise background: with a fixed background, VICReg and SimCLR learn the background and lose the dot. ==In MotionJEPA on Pong, Dino and Golf, SIGReg in LeWorldModel keeps the score and the paddles and loses the ball (ball NMSE 1.001 against 0.005 for the DISReg regularizer, which predicts the embedding of frame differences), and the loss does not show the collapse. Strohm et al. report that on SLIM, where 0.37% of pixels change per step, LeWorldModel with SIGReg solves 0.3% of tasks, and an auxiliary inverse-dynamics loss raises this to 34.8%. In TC-LeWM, SIGReg on the whole latent leaves the temporally centered residual with little variance, because the persistent part fills the unit variance budget of each projection and prediction favours it; applying SIGReg to $r_t=z_t-\bar z_t$ over a window of four frames raises the success of a policy on frozen features from 63.6% to 83.8% on LIBERO (suite-wise, 10 tasks), and the variance argument is a Monte Carlo toy with independent Gaussian parts. TDV learns from temporal differences and reaches kNN top-5 of 17.05 on ImageNet-1k after pretraining on SSv2, against 40.19 for DINO with full augmentations, which suggests that dynamics alone do not give semantics.== In every case the cheap feature is a static part that the repulsion can spread at no cost, the analogue of the bits.

### 7.5. Is one failure case possible?

The answer is yes for the two families tested, and RandBit touches all three conditions named at the start of the chapter. The bits are structure between samples: an instance code that distinguishes inputs and says nothing about their content. The number of samples enters through the threshold, since the shortcut wins once it can separate everything the loss compares, a batch for InfoNCE, and with $n$ images $\log_2n$ bits identify every image. ==The threshold should therefore fall as the dataset shrinks, which the sample-size sweep of 10.4 tests.== The unclear goal is the reason guidance failed: the labels did not specify that the class must not be read from the code. Structure within a sample, the static and dynamic parts of a frame, appears in the natural cases of 7.4. Each condition has its own remedy. Structure within samples is handled by changing the prediction target or the domain of the repulsion (Proposition 5.3), or by reconstruction, which suppression barely affects (a VAE in Chen et al., PrCL in Li et al.). Structure between samples is handled by augmentations that randomize the shortcut, as the fresh bits do, or by augmentations in feature space (Hamidieh et al.). Few samples are handled by building the graph from patches (chapter 9), and an unclear goal by labels that cover the variation of the shortcut (chapter 8).

==The siamese family is untested, and Chen et al. report that BYOL suffers as much as SimCLR. The next runs are three seeds near the thresholds ($b=4,6,8$), a BYOL row, guidance at the threshold ($b=6$ for SIGReg and InfoNCE, $b=4$ for VICReg) where suppression is partial and labels have something to hold on to, a reconstruction control, a natural shortcut, and a sweep over the number of images (10.4).==

## 8. Downstream-guided SSL

If semantics is the choice of the task family $\mathcal F$ (Proposition 6.1), a task or a few labels can state that choice directly, and since there is no general embedding quality (6.5), the task has to enter training or at least the measurement. Chapter 7 adds a condition: the task signal helps only when the shortcut cannot satisfy it. ==What to measure, and how to use the task, is open.==

### 8.1. Families of methods that use a task

Five families use a task. Classical semi-supervised learning relies on the manifold, smoothness and low-density assumptions (FixMatch, ==S4L==, ==PAWS==). Supervised contrastive learning uses labels to define positives. Pretraining aware of the downstream task includes BiSSL, V-pretraining and task-customized pretraining. Continual learning projects gradients (A-GEM, ==PCGrad==). Some methods choose the invariance by task (==AIL, ContextSSL==). All of them assume that the structure of $p(x)$ is tied to $p(y|x)$. ==What each method assumes, and where reports on them disagree, is uncollected (8.5).==

### 8.2. Predictive information measures structure

A measure of structure that needs no task is predictive information. In time it is $\mathrm{PI}=I(x_{\le t};x_{>t})$ (Bialek-Nemenman-Tishby); MotionJEPA splits a static term on $z$ from a dynamic term on frame differences, with hand-set weights $\lambda_z,\lambda_d$. In space it is $I(x_A;x_B)$ for two regions of one frame: flat background and noise give $I\approx0$, objects and texture give a large $I$, and I-JEPA, MAE and InfoNCE already estimate such a quantity. For Gaussians the capacity it assigns to each feature is explicit.

*Proposition 8.1 (CCA).* For a Gaussian pair $(x_A,x_B)$ the best $k$-dimensional linear code $z=Wx_A$ for $I(z;x_B)$ is the projection on the top $k$ canonical directions, with $I=-\frac12\sum_{i\le k}\log(1-\rho_i^2)$.

A feature with a small canonical correlation $\rho$, like a ball of a few pixels, is dropped at small $k$. Such a feature is not chaotic; it carries little information per unit of capacity, and large predictable features (colour, lighting, continuation of texture) take capacity first. Augmentations and masks exist to take that role away from cheap features. A universal measure decides what counts as structure, and the correlations $\rho_i$ with the budget $k$ decide how much capacity each feature receives, which is the mechanism of chapter 7 in its simplest form.

### 8.3. Information weighted by the task

A task weights information through $I(z;y)$ (information bottleneck) or through the sensitivity $\partial y/\partial z$. For a linear probe with vectors $w_j$, $G_T=\frac1k\sum_jw_jw_j^\top$ (in general with $v_j=\partial\hat y_j/\partial z$) is a positive semidefinite matrix whose range is the subspace the task reads, and $P_T=G_TG_T^{+}$ projects onto it. Its cost is about 30 labels.

### 8.4. Gradient projection with a ridge probe

The simplest way to let a task correct SSL is to remove from the SSL step the component that hurts the task. Let $g_{\rm task}$ be the gradient of the probe loss and $g_{\rm ssl}$ the SSL gradient.

*Proposition 8.2 (A-GEM).* If $g_{\rm task}^\top g_{\rm ssl}<0$, set $g'=g_{\rm ssl}-\frac{g_{\rm task}^\top g_{\rm ssl}}{\|g_{\rm task}\|^2}g_{\rm task}$; then $g_{\rm task}^\top g'=0$, and the step does not worsen the probe loss to first order.
*Proof.* Substitute.

The signal $g_{\rm task}^\top g_{\rm ssl}$ matches V-pretraining (Ke-Fanti), which uses 1,024 GSM8K examples only as feedback to a task designer; here it corrects the step. The ridge probe has the closed form $W=(Z^\top Z+\lambda I)^{-1}Z^\top Y$, so $g_{\rm task}$ is computed through $Z$ without an inner optimization. The construction is a strong baseline more than a new method, and chapter 7 shows its limit: when the labels are satisfied through a shortcut, it projects one shortcut gradient against another.

### 8.5. Three directions and what to measure

Three directions are open. The first replaces a hand-made prior with labels: whether 30 labeled ball positions give the effect of the frame-difference regularizer of MotionJEPA (on Pong, ball NMSE 0.005 against 1.26 for the forward-only baseline). It has a baseline and a clear criterion, so it goes first. The second puts the task subspace $G_T$ in $z$ instead of the parameters, as one object that plugs into SimCLR, JEPA or DINO without changing their loss and composes over several tasks; the task is held by a lower bound on the information in $P_Tz$, and nothing else has to be removed adversarially. The third is a diagnostic of when and which information each SSL method washes out, on tasks with known factors.

Downstream accuracy alone does not measure how the task changes the representation. The quantities to measure are the contribution of each feature (the sensitivity $\partial y/\partial z$ and $G_T$), the share of task-relevant information in $z$, and the amount of task information SSL removed. For each downstream-guided method, an assumption table like 4.7 records what it assumes (task, number of labels, domain), where reports disagree and what critics say. ==Reproducing two or three methods on their own data and transferring them to other fields (medical images, time series, audio) is planned (10.5).==

## 9. Semantics on small samples

This chapter applies chapters 2-6 to one practical case: 30 unlabeled images, a small network trained from scratch with no external data, then a frozen encoder with a linear probe trained on a large labeled set. The setup separates the quality of the representation from the scarcity of labels: if the probe is bad with unlimited labels, the geometry of the encoder is the problem. ==How much data suffices is open.==

### 9.1. What the theory says about this regime

Three facts from chapters 2-6 shape this regime. At $n=30$ neither $G$ nor $\mathcal F$ can be learned, so by Propositions 6.1 and 2.1 they must be built in through the architecture, patches or allowed transformations. The centered embeddings of $n$ images span at most $n-1$ directions, so 30 images fix at most 29 axes and cannot match an isotropic Gaussian in 128 dimensions. ==Whether augmented views add task-relevant directions beyond these 29 is open.== A regularizer constrains the law only at the training points (Proposition 2.11), so two functions with the same $\mathcal N(0,I)$ on 30 points can differ on the 31st image. Chapter 7 adds a fourth fact: with 30 images, five bits of any cheap feature identify every image, so the shortcut threshold is low. ==This is a prediction, untested.==

### 9.2. How to apply it

The first fact suggests replacing 30 images by $10^5$-$10^6$ correlated patches, which give a large graph $G$ from few images (Models Genesis, MAE, Swin-MAE, ZSSR, sparse coding, patch k-means). An alternative is a built-in representation such as wavelet scattering, Deep Image Prior or random features. The third fact suggests adding prediction of a hidden part, which learns internal dependencies while SIGReg only keeps the axes alive. The expectation is that SIGReg with prediction beats SIGReg alone (10.6).

For segmentation, 30 images suffice with dense labels and strong deformation augmentation (U-Net, nnU-Net). With a yes-or-no tumor label the effective $n$ is about 30, and a flexible model does not beat a simple one. Few-shot segmentation solves a different problem, since its prior is already in the weights.

### 9.3. How much data: information and sufficiency

The question is how many points, and how diverse, give a representation sufficient for a task family, where diversity is measured by the information in the examples. Mutual information cannot be estimated reliably from a small sample: any distribution-free high-confidence lower bound on mutual information from $n$ samples is at most about $\log n$ (McAllester-Stratos). Usable information (V-information, Xu et al.), which accounts for the probe class, replaces it, and the information bottleneck gives the frame of a sufficient and minimal representation. Geometry has the same problem: with the optimal width, the error of a KDE falls as $n^{-4/(D+4)}$ (Proposition 4.11), which at $D=128$ barely improves with $n$, so geometry cannot be estimated in the raw space, and only the intrinsic dimension $m$ is workable (minimax manifold estimation, Genovese et al.). The practical question becomes how to estimate $m$ and the coverage of the data. Empirical work covers size, diversity and domain of the dataset (Cole et al.), SSL on one image (Asano et al.), whether large datasets are necessary (El-Nouby et al.), example selection (Joshi-Mirzasoleiman), data pruning and scaling laws (Sorscher et al.), and pretraining data diversity (Hammoud et al.).

### 9.4. Small-data methods

Small-data methods replace contrastive losses with masking and patch prediction, internal learning, or JEPA. ==Which of them works best across domains and data types (medicine, time series, audio, molecules) under one protocol is open (10.6).==

## 10. Experiments

The experiments test the open claims of chapters 2-9. Each row names the proposition or section it checks, and every row is planned unless chapter 7 reports it.

### 10.1. Theory checks

| Experiment | What | Checks |
| --- | --- | --- |
| ==Shortcut toy== | easy feature $s$, useful $t$; sweep $K$ and $\delta$; compare the loss gap with $K\delta$ | Proposition 3.4 |
| ==Hessian at collapse== | sign and size of the smallest InfoNCE Hessian eigenvalue against $\tau$ | Proposition 3.1 |
| ==Graph spectrum against embeddings== | eigenvectors of $\bar A$ on a small graph against SimCLR, VICReg and spectral-loss embeddings | Propositions 2.7-2.9 |
| ==Ridge probe against anisotropy== | probe on embeddings of varied anisotropy at fixed trace | Proposition 5.2 |
| ==Free particles against a network== | the same loss on free embeddings and on an encoder; spectrum of $\Theta$ and final geometry | Proposition 4.1 |
| ==Sign of $d\mathcal L/dt$ under PPS== | loss trajectory and $\Phi=p-p_{\min}$ under a fixed and an adaptive temperature | Proposition 3.5, 4.5 |
| ==Convergence under $\tau$ schedules== | epochs to a fixed probe accuracy for constant and adaptive $\tau$ | 4.5 |
| ==Embedding quality against the world== | linear probe and retrieval against similarity of known factors on a toy | 6.5 |
| ==Invariant, random and equivariant augmentations== | three types on one downstream task | 6.2 |
| ==Representation trajectory== | SSL against supervised learning under a moving target | 3.8 |
| ==JEPA with a generator on the same latent== | JEPA and a denoising-score generator tied to one latent | 4.6 |

### 10.2. PPS in other regimes

| Experiment | What | Checks |
| --- | --- | --- |
| ==Seeds, architectures, batches== | several seeds, ResNet-50, ViT, batch sizes | whether PPS holds beyond one configuration |
| ==Other methods== | MoCo, DINO, SupCon; BYOL, VICReg | where "kernel width equals the size of the positive cluster" works |
| ==Other domains== | medical images, time series, audio, small samples | PPS when $p$ is estimated from few pairs |
| ==Transfer protocols== | collect the transfer protocols of SSL papers | how protocols differ |
| ==Measure $q^*(\sigma)$== | dynamics of the inter-input coordinate at several fixed $\sigma$ | closing the model of 4.5; a schedule from safe to optimal |
| ==Width rule in target regularizers== | Epps-Pulley weight width and heat-kernel time from the positive-pair distance | hypothesis of 2.5 |

### 10.3. Unification

| Experiment | What | Checks |
| --- | --- | --- |
| ==Log-drift against MMD-drift== | InfoNCE, SIGReg and SPHERE-JEPA on one sphere toy; compare the fields | Propositions 2.4-2.6 |
| ==Sample- against dimension-contrastive== | the two penalties of Proposition 2.10 under the same normalization; probe and geometry | Proposition 2.10 |
| ==Siamese as implicit repulsion== | estimate the effective $\mu^-$ of BYOL from its update and compare with explicit repulsion | 2.4 |
| ==Three theories, one question== | collapse or $\tau$ through Wang-Isola, Huang and PPS, LeJEPA, and the three explanations of BYOL; where predictions differ | 5.4, 6.6 |
| ==MMD-flow stalls for vMF== | theory (no stable stalls) and a numerical check on the sphere | 4.6 |
| ==Drift monitor on SIGReg and SPHERE-JEPA== | $\|V\|^2$ as a label-free diagnostic; compare with KSD and Fisher divergence | 2.5 |
| ==KL to uniform against InfoNCE== | InfoNCE repulsion against the KL variant of Expanding SPHERE-JEPA | Proposition 4.4 |
| ==How many moments== | VICReg, Weak-SIGReg and SIGReg on data with controlled non-Gaussianity | 5.3 |
| ==Capacity cost== | shrink width or dimension and find where shape displaces content | 2.5 |
| ==Domain of the repulsion== | SIGReg on $z$ against the temporally centered residual, on a toy with a static and a dynamic factor; variance of each part and probe error | 7.4, Proposition 5.3 |
| ==Learned graph with cycle consistency== | $R_{ij}=h(x_i,x_j)$ instead of hand-made augmentations, with a penalty against $R=I$ | 6.4 |

### 10.4. The shared failure

| Experiment | What | Checks |
| --- | --- | --- |
| RandBit suppression | InfoNCE, VICReg, SIGReg at $b\in\{0,2,4,6,8,16\}$, one seed | Proposition 2.11; reported in 7.2 |
| RandBit with labels | joint, A-GEM and fresh bits at $b=16$ | 7.3 |
| ==Seeds near the thresholds== | three seeds at $b=4,6,8$ and on the runs with labels | order of thresholds |
| ==Siamese row== | BYOL on RandBit | whether the failure covers all three families |
| ==Guidance at the threshold== | $b=6$ for SIGReg and InfoNCE, $b=4$ for VICReg | whether labels help where suppression is partial |
| ==Reconstruction control== | a decoder variant (PrCL, VAE) | whether reconstruction escapes suppression |
| ==Sample-size knob== | the threshold $b$ against the number of images | 7.5, 9.1 |
| ==Natural shortcut== | slow-features scene and a Pong-like scene; the same three losses and DISReg | 7.4 |

### 10.5. Downstream-guided SSL

| Experiment | What | Checks |
| --- | --- | --- |
| ==30 labels against DISReg== | Pong, ball probe, DISReg baseline | 8.5, first direction |
| ==$G_T$ as a plugin== | $G_T$ from probe gradients in SimCLR, JEPA, DINO; composition of two tasks | 8.5, second direction |
| ==Washout diagnostic== | which factors each SSL method removes, on a 3D sandbox with few latent factors | 8.5, third direction |
| ==Assumption table== | from papers, critiques and reports | 8.5 |
| ==Reproduction and transfer== | two or three methods on their data, then other fields | 8.5 |

### 10.6. Small sample

The protocol uses 30 images and moves from an untrained architecture to prediction within an image, then SIGReg on 30 points, a check of linear accessibility, and finally labels and a sweep over the sample size.

| Experiment | What | Checks |
| --- | --- | --- |
| ==Random convolution, untrained== | frozen random convolution, linear probe | what locality gives without SSL |
| ==MAE with a feature field== | MAE autoencoder, field $E(x)$ of size $H'\times W'\times d$, $1\times1$ probe | reconstruction and localization |
| ==SIGReg, VICReg and LeJEPA on the whole image== | regularizer on 30 points | points spread, tumor not encoded |
| ==Linear probe against a small MLP== | on random convolution, MAE and the whole-image regularizer | the MLP wins when a feature is present and not linear |
| ==MAE with variance and covariance or SIGReg on the latent== | MAE with a latent regularizer | probe against plain MAE |
| ==$1\times1$ probe for localization== | field $E(x)$ and probe | whether the spot disappears without a tumor crop |
| ==Effective rank and SIGReg diagnostic== | on the same $z$ | high rank and a bad probe: coordinates alive, tumor absent |
| ==Sample-size sweep== | $n\in\{30,100,300,1000\}$ for MAE, MAE with a regularizer, the whole-image regularizer | where global SSL starts to work; agreement with $\mathrm{rank}\le n-1$ (9.1) |
| ==Labels on top== | $m\in\{0,10,30\}$ labels over MAE with a regularizer | link of chapters 8 and 9 |
| ==How many points suffice== | usable information and coverage on a known-factor toy against the geometric bound | 9.3 |
| ==Small-data methods across domains== | masking, internal learning, JEPA under one protocol | 9.4 |

Further variants are wavelet scattering; patch k-means or sparse coding; InfoNCE, DCL and a graph $G$ from crops, time or dropout; Deep Image Prior on a 3D render; a 4-dimensional bottleneck with a shared transformation operator as an identifiability toy; LieGAN and a version space.

### 10.7. Visualizations

One shared 3D sandbox produces the figures of the theory chapters.

| Section | Figure |
| --- | --- |
| 2.2 | ==sphere normalization; cosine against distance; softmax weights of negatives against $\tau$; InfoNCE against DCL; the Gaussian kernel on the sphere== |
| 2.3-2.5 | ==the drift field $V$ for InfoNCE, MMD and SIGReg in one picture; the SIGReg cloud; the loop $z\to$ geometry $\to$ regularizer $\to z$== |
| 3.1, 3.7, 4.5 | ==the saddle at collapse; particle flow and the Lyapunov function; $\tau(p)$ of PPS against $\tau(A)$ of Huang== |
| 7 | suppression curves of RandBit (`experiments/randbit/reports/randbit_suppression.png`) |

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
- Chen, Luo, Li. Intriguing Properties of Contrastive Losses. NeurIPS 2021. arXiv:2011.02803
- Li et al. Addressing Feature Suppression in Unsupervised Visual Representations (PrCL). WACV 2023. arXiv:2012.09962
- Xue, Joshi, Gan, Chen, Mirzasoleiman. Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression. ICML 2023. arXiv:2305.16536
- ==Hamidieh, Zhang, Sankaranarayanan, Ghassemi. Views Can Be Deceiving: Improved SSL Through Feature Space Augmentation. ICLR 2024. arXiv:2406.18562==
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
- ==Cao, Wei, Liu. Gradient Flow Drifting: Generative Modeling via Wasserstein Gradient Flows of KDE-Approximated Divergences. 2026.==
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
- ==Tian. Understanding Deep Contrastive Learning via Coordinate-wise Optimization. NeurIPS 2022.==
- ==Simon, Knutins, Ziyin, Geisz, Fetterman, Albrecht. On the Stepwise Nature of Self-Supervised Learning. ICML 2023. arXiv:2303.15438==
- ==Garrido, Chen, Bardes, Najman, LeCun. On the Duality Between Contrastive and Non-Contrastive Self-Supervised Learning. ICLR 2023.==
- Jacot, Gabriel, Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. NeurIPS 2018.
- Roy, Vetterli. The Effective Rank: A Measure of Effective Dimensionality. EUSIPCO 2007.
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
- ==Sobal, Jyothir S V, Jalagam, Carion, Cho, LeCun. Joint Embedding Predictive Architectures Focus on Slow Features. 2022. arXiv:2211.10831==
- ==Strohm et al. Keeping JEPA World Models Plannable When Little of the Frame Moves. 2026. arXiv:2610.03137==
- ==Littwin et al. How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks. NeurIPS 2024. arXiv:2407.03475==
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

- ==Tan, Zhang, Yang, Yuan. Contrastive Learning is Spectral Clustering on Similarity Graph. ICLR 2024.==
- ==Wang, Zhang, Wang, Yang, Lin. Chaos is a Ladder: A New Theoretical Understanding of Contrastive Learning via Augmentation Overlap. ICLR 2022.==
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
