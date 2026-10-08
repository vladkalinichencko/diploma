# Ledger: diploma draft

Target: `draft-1.md`, whole file.

## Constraints

- Audience: readers outside the author's life who want to see the result: a supervisor, a committee, a researcher in SSL. They know ML and have not seen any chat.
- Where the text goes: a diploma thesis draft. It may promise planned experiments only as open items marked with `==...==`.
- Length: about 60 pages for the final thesis. The current draft is a skeleton, about 35-40 pages of Markdown.
- Math syntax: `$...$` and `$$...$$` in Markdown.
- Language and register: English, formal, single author, so no "we" and no "I" in the text; the work and its objects are the subjects.
- Structure: hierarchy and sequence first. Every section opens by saying why it is there and how it follows from the previous one. Headings alone must read as a plan of the work.
- Highlights: `==...==` marks what is not done, not read, not checked, or open, and every connecting claim of this work that no cited paper states. Proven statements with proofs in the text stay plain; own propositions carry "(this work)" in their label.

## Sources

- Primary artifact: `~/VSCodeProjects/Bandwidth Research/Research/paper/submissions/icomp-2026/main.tex` and `references.bib` (PPS paper; read: yes).
- RandBit experiment: `experiments/randbit/README.md`, `experiments/randbit/reports/randbit_table.md` (read: yes).
- Robinson reproduction notes: `~/VSCodeProjects/ssl-robinson-ifm/notes`.
- Critiques of Wang-Isola: ChatGPT dialogue https://chatgpt.com/share/6a45a21f-c0fc-83ea-8f67-112af747a159 (read: yes). Nie et al. 2023, Wang-Liu 2021, Jing et al. 2022, Fang et al. 2024, Xue et al. 2023, RankMe.
- Obsidian notes: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Vlad Otpad's Vault/SSL` ("SSL Theory.md", "SSL on Tiny Sample with a Tiny Model.md").
- Previous version before the restructure: git commit f98badf (includes the author's edits of 7 Oct, 15:00). The mapping below translates its section and proposition numbers. Repository: https://github.com/vladkalinichencko/diploma (private).
- Theory papers read in full for their theorems and measurements: Luthra, Bryant, Zhu, Galanti 2026 (arXiv:2609.38393); Luthra, Yang, Galanti 2025 (2506.04411); Luthra, Salunkhe, Galanti 2026 (2603.03530); Gretton et al. 2026 (2605.05118); Zhu et al. 2026 (2609.04264); the Zotero collection "SSL bandwidth scheduling": Tian, Chen, Ganguli 2021 (2.4, 4.7); Tian 2022 $\alpha$-CL (4.4); Garrido et al. 2023 duality (2.3); Tao et al. UniGrad (2.4); Huang et al. (4.3); Wang 2024 (4.3); van den Oord et al. CPC (5.2); Saremi and Hyvärinen NEB (4.5, 4.6); Zhang, Wang, Wang U-MAE (9.3); Turan, Dufour, Ovsjanikov (4.5, 4.6); Cao, Wei, Liu Gradient Flow Drifting (4.6); Mittal et al. (9.3); Hyvärinen 2007 (4.6); Vincent 2011 (4.6); Liu's blog post on drifting models (4.6); the SSL Cookbook as a coverage check (5.1). Read only at the level of abstracts, so highlighted in 8.1: MCL, LooC, CASSLE, DivDis, lens, Viewmaker, Steerable, task-robust pretraining, Uzan-Weinberger, Bordes et al. 2023.
- Figures: `figures/atlas.py` writes `figures/system.svg` (Figure 1) and `figures/atlas.svg` (Figure 2); run `python3 figures/atlas.py`. The level names in the script, in Figure 1 and in the table of 1.2 must stay identical.

## Points

- P1: Joint-embedding SSL = a relation that says which inputs are the same (graph $G$) plus a mechanism against collapse. Contrastive, regularizer and siamese families differ in the mechanism. (chat; §0, §2)
- P2: One object for all three families: the gradient as a flow of embeddings, attraction minus repulsion, $V=\mu^+-\mu^-$; for kernel losses a score difference. (PPS §3; §2)
- P3: Invariants: attraction alone collapses; multiset repulsion sets shape and never content; sample- and dimension-contrastive losses differ by norms (Garrido). (§2.5, §2.3)
- P4: Laws of motion: collapse saddle, temperature sets push and pull, positives contract while $q$ decides collapse, dimensional collapse, easy features first, projector gap, adaptive $\tau$ breaks the Lyapunov property of the loss. (§3)
- P5: Rigor ladder, realizability (free particles vs network, NTK), why theorems cannot be stacked. (images; §4.1)
- P6: Wang-Isola is correct in its limit and misses the gradient field, tolerance, spectrum, augmentation variance, the SGD solution and the projector; it fixes no kernel width and no semantics. (ChatGPT dialogue; PPS App. C; §4.2)
- P7: PPS as a chain of questions with its rigor level. (PPS §3-4; §4.5)
- P8: Score matching, diffusion, drifting generators share the score-difference object; exact identities transfer, convergence results need a fixed target. (§2.6, §4.6)
- P9: Three groups of implementations arrived in waves. (§5)
- P10: Semantics enters only through the attraction, the architecture or the task. (§6)
- P11: One failure case shared by all three families: RandBit; Robinson reproduction folded in as the first attempt. (RandBit README; §7)
- P12: Downstream guidance works only when the labels cannot be satisfied through the shortcut. (RandBit README; §7, §8)
- P13: Small-sample SSL is an application, not the main work. (C6; §9)
- P14: Graph ML is a side formalism, highlighted. (chat; §6.4)
- P15: Every result looks at a few cells of one map of the training system (data, network, sample, objective order, time, control, use). A theorem that is sound in its paper fails in the next one because the test asks about a cell the theorem leaves blank. (chat; §1.2, Figure 1, Figure 2, §4.1, §4.7)
- P16: Axes of non-fixedness: score matching has a fixed target, SSL moves the target law, the kernel width, the relation, the batch, the parametrization and the second branch; each moving axis turns a convergence theorem with a fixed target into a statement about a sequence of problems. (chat; §4.6 table)
- P17: New theory papers, each with what it measures, on which object, how it is derived, how it agrees with the rest, and what happens at finite samples: two-view operator and captured energy (4.3, Prop 4.7), NSCL gap $e^{2/\tau}$ (Prop 4.8), directional CDNV on $h$ (4.3), first variation of uniformity (Prop 4.16) against Gretton's WGF velocity (Prop 4.17), Zhu's laziness numbers (7.4), RandBit counts (7.2). (chat; §4.3, §4.6, §7.2, §7.4, §8.3, §8.5)

## User phrasings

- U1: "Why are we even talking about it": every section and proposition opens with its reason. Used as a structural rule, not as a phrase.
- U2: "unify contrastive, regularizers and siamese approaches": used in §0 and §2.
- U3: "waves of different approaches, grouped into 3 groups": used in §5 opening.
- U4: Wang-Isola "a very cool and influential paper" that "still fails to cover all properties of original infonce loss": used in §4.2 opening, in register.
- U5: "оси нефиксированности": used as "axes" of the moving target in §4.6, in English.
- U6: "нулевого порядка — это к чему вообще относится?": answered in place, "a zero-order model of contrastive learning at the level of the embedding law" (§4.2) and the order paragraph of §1.2.
- U7: goal of the work: "чётко определяет, где в какой теории есть конкретные условности и предположения и как они на одной карте друг с другом соотносятся ... проверить на практике, что из этого сходится, что нет, и предложить несколько своих методов, которые закрывают многие из этих вещей наиболее общим способом". Used in the last paragraph of §0.
- U8: "вместо доказательств писать выводы. То есть именно каким образом мы к этому приходим, вместо того, чтобы писать, как это доказывается". Used in chapters 2-8: the step that leads to each proposition is in the text, the proofs are in Appendix A.1-A.6 (A.1 chapter 2, A.2 chapter 3, A.3 chapter 4, A.4 chapter 5, A.5 chapter 6, A.6 chapter 8).
- U9: chapter 9 "должен быть типа всякие варианты и информация внутри небольших сэмплов данных"; "SSL on 30 images" was a test setup and stays only as the running test case. Used in the chapter 9 title and structure.

## Corrections

- C1: bad: long Russian notes, emojis. fix: English, KISS, less text.
  rule: English, no emojis, cut what the reader does not need.
  pattern: [\U0001F300-\U0001FAFF]
- C2: bad: "[DONE]", "[WIP]", "[UNREAD]", "NEEDS WORK". fix: highlight the unfinished content itself with `==...==`.
  rule: no status tags.
  pattern: \[(DONE|WIP|UNREAD|TODO)\]|NEEDS WORK
- C3: bad: sections titled "Summary", "Done / Not done". fix: the document reads as a finished, consecutive work.
  rule: no report labels.
  pattern: (?im)^#+\s*(summary|done|not done|status)\b
- C4: bad: "Experiment A", "Experiment B". fix: name experiments by what they test.
  rule: no letter names for experiments.
  pattern: \bExperiment [A-Z]\b
- C5: bad: bare statement lists. fix: main statements and proofs with verbal transitions, trivial steps and assumptions named.
  rule: every proposition carries its proof or a cited source and its assumptions.
- C6: bad: small-sample chapter as the main contribution. fix: it is a practical application of knowing which semantics is needed.
  rule: small samples sit in Part II as an application.
- C7: bad: "SSL theory takes the positive pair and the loss as given". fix: the positive pair is contrastive-specific; state the relation and the anti-collapse mechanism for all joint-embedding SSL.
  rule: claims about "SSL" must hold for contrastive, regularizer and siamese families, or name the family.
  pattern: SSL theory takes the positive pair
- C8: bad: "### Where the work stands" table with status per section. fix: delete; status lives in highlights and in this ledger.
  rule: no author-facing status sections in the document.
  pattern: Where the work stands|is finished\.|The PPS paper \(ICOMP 2026\) is finished
- C9: bad: propositions as a dictionary of statements. fix: each proposition sits in a paragraph that says why it is needed and what it gives to the next step.
  rule: a proposition is preceded by its question and followed by its consequence for the argument.
- C10: bad: "The symbols below are used throughout; the last two paragraphs fix...", bold Q&A headers, colon-heavy explanatory lists. fix: natural prose with soft transitions, no text about how to read the document.
  rule: no meta text about the document, no FAQ blocks.
  pattern: (?i)the (symbols|sections?|paragraphs?) below|this (chapter|section) (describes|explains|is organized)
- C11: bad: "9.1. Robinson et al. reproduction" as its own section. fix: a small experiment; fold it into the shared failure case.
  rule: no section for a side experiment; it supports a point.
  pattern: (?m)^#+.*Robinson
- C12 (global): no em dashes; no "not X, but Y"; no hard line wraps inside paragraphs.
  pattern: —
- C13: bad: "Wang and Isola is a zero-order model" without saying zero order of what. fix: every claim names its object (embedding law, individual embeddings, batch, loss, parameters or neurons) inside the sentence, without a separate sentence for it.
  rule: each statement carries its level from the table of 1.2; a symbol has one meaning in the whole text ($h$ is only the representation).
- C14: bad: "Three theories, one question" (10.3). fix: a heading says what the section contains, with as many spoilers as fit; no pathos, no wordplay.
- C15: bad: Kernel ISOMAP, CDNV, kernel $\Theta$ used before any gloss; paragraphs with no link to the previous one. fix: a term is explained by its mechanism at first mention, and each paragraph opens from the previous one.
- C16: bad: $g$ for the gap, the predictor, the generator and the encoder; $G_T$ beside the graph $G$. fix: one symbol per object across the text, as few subscripts as possible (see Decisions).
- C17: bad: a proposition with no source. fix: every proposition and proof names where it comes from, "(Author)", "(after Author)" or "(this work)".
- C18: bad: "To understand a method here means to predict ..." in §0. fix: no rules for the writer inside the text; §0 states the goal in U7.
- C19: bad: "easy features are learned first" with "easy" undefined. fix: a vague adjective gets its operational meaning at first use (3.5).
- C20: bad: ", and it answers through one object." fix: no announcing or summarizing tails that add no content.
  pattern: answers through one object
- C21: bad: "embedding law", "target law". fix: "distribution" everywhere ($p_z$, $P_X$, target distribution), in text and figures.
  pattern: \b(embedding|target|data) law\b
- C22 (global): KISS. The shortest wording that keeps the argument; more figures from real runs and redrawn paper schematics.
- C23: bad: "## 9. SSL on 30 images", "## 10. Planned experiments". fix: a heading names the content the chapter holds, and a test setup is never a chapter title; chapter 10 says where each result goes.
  pattern: SSL on 30 images
- C24: figures come from the cited papers and from the PPS paper as well as from own runs; every caption ends with its source ("From X et al. (year)." or "From the PPS paper, Figure N.").
- C25 (repeats C15, user: "в начале каждой новой мысли должен быть ПЕРЕХОД"): bad: "## 9. Information inside and between small samples / A small sample carries information in two places." fix: the first sentence of every chapter, section, paragraph and table lead-in names how it follows from the text just before it (what the previous part left open, which object it moves to, or why the case changes). A heading followed directly by a table or by another heading gets a lead sentence. Check by reading the last sentence before each joint and the first after it, for the whole draft.
- C26 (user: "это нейрослоп"): bad: "With only a few dozen unlabeled images, the information has to come from the images themselves, and a small sample carries it in two places." fix: ", and" never glues a second independent claim onto the first as a fake connective. End the sentence; the next one says how it follows. ", and" stays for lists and for two actions of one subject. The checker flags it as `and-chain?`; every flag is either rewritten or a list.
- C27 (user: "указание кол-ва мест/объектов перед тем как о них рассказать"): bad: "in two places", "Three kinds recur.", "four facts from chapters 2-7 bound", "a distortion can enter the loss in two ways", "Three directions are open." fix: name the first item and why it comes up, then the next; no count announced before the items.
  pattern: \b(two|three|four|five|six|several) (\w+ )?(places|ways|kinds|facts|directions|families|questions|steps|routes|lessons)\b
- C28 (user: "у всего должна быть причина, ПЕРЕД самим текстом"; "Причём здесь нижние слои? Ты как-то вводишь вообще это понятие?"): bad: "Asano, Rupprecht and Vedaldi match the first layers of a network trained on a million images" with layers never introduced. fix: every concept is introduced with what it is and why the argument needs it before it is used, and every claim or proposition is preceded by the reason it comes up and the step that leads to it. Proofs go to Appendix A, the derivation chain stays in the text (extends U8 to chapters 3-8).
- C29 (user: "вопрос был вообще само искажение должно быть инвариантным или нет? ... либо случайный шум добавляем, либо конкретно там какой-то поворот на какой-то градус"): bad: 9.3 asked whether the representation should be invariant to a distortion. fix: 9.3 contrasts a random distortion whose parameter nobody tracks (noise, random crop) with a specific transformation whose parameter is known (rotation by 90°), and says what each lets the loss ask for.
- C30 (user: "а всем изображениям добавь подписи"): every figure has a visible caption line under the image; the alt text alone may not render.
- C31 (user: "как ты вообще определяешь эти частоты ... на уровне графа датасета или батча, или на уровне самого изображения?"): bad: "the weak cuts of this graph follow low frequencies, not objects". fix: every spectrum, frequency or eigenvector names its object (Fourier modes of one image, principal components of the dataset covariance, the graph $G$ on the dataset, the matrix of one batch), and a link between two of them gets its reason (stationary image statistics make principal components close to Fourier modes, 9.3).
- C32 (user: "Все мои идеи ... надо записывать ... в эксперименты, чтобы протестировать"): every idea of the author that the text can test becomes a highlighted row in chapter 10 and, where it belongs to an argument, a highlighted sentence in that chapter. Derivable claims of this work stay highlighted until their proof is in Appendix A. The running list with status is `notes/ideas.md`.
- C33 (user: "не знаю ничего про deepcluster. ты это как-то в диплом пишешь, или как? одно упоминание мне ничего не объяснит"): bad: "DeepCluster (Caron et al. 2018) relies on this." fix: a cited work gets what it did, the number that matters here and why the argument needs it at this point, or it is not cited there.
- C34 (user: "ты сравниваешь детерминированный knn и выученную модель"): a comparison sets like against like (a learned encoder against a learned encoder); a deterministic baseline enters only as the predicted ceiling or floor of a learned model, said as such.
- C35 (user: "мои мысли пока как hypotheses добавлять, ну или как наш paper contribution"; "связывать параграфы абзацы и выводы формул не забывай"): an idea of the author enters as a highlighted paragraph "*Hypothesis Hn (this work).*" next to the argument it follows from, with a lead sentence that links it to that argument and the chapter 10 row that can refute it; chapter 0 lists all of them.

## Section map from the previous structure

Old 2.1 → 2.2 and 5.2; 2.2 → 4.2; 2.3.1 → 3.5; 2.3.2 → 3.2; 2.3.3-2.3.4 → 5.2; 2.3.5 → 6.2; 2.3.7 → 4.2; 2.3.8 → 6.5; 2.4 → 4.3; 2.5 → 3.1, 3.7, 4.1; 2.6 → 4.5, 5.2; 3.1 → 2.3, 2.4, 5.3, 5.4; 3.2 → 6.3; 3.3 → 2.5; 3.4 → 7; 4.1 → 2.3; 4.2 → 5.1; 4.3-4.4 → 2.3, 2.5; 4.5-4.6 → 6.4; 4.7 (author's "Where a method intervenes") → 5.5, its Prop. 4.6 → Prop. 5.3; 5 → 6; 6.1-6.4 → 4.5, 4.6; 6.5 → 3.8; 6.6 → 2.6; 7 → 8; 8 → 9; 9.1 → 7; 9.2-9.7 → 10.1-10.7.

Old proposition numbers used outside the draft: 3.4 (multiset invariance) → 2.11, 6.2 (A-GEM) → 8.2. `experiments/randbit/README.md` cites the new numbers.

Renamed headings (draft of 2026-10-08): 2.5 "Shared properties: ..." (was "Three shared properties: ..."); 4.2 "Wang and Isola: the minimizer of the limit loss and what the limit misses"; 5.1 "Waves of methods and their families"; 5.4 "Siamese and distillation methods: competing explanations of why they do not collapse"; 5.5 "Places in the pipeline where a method intervenes"; 7.5 "RandBit and the hard conditions: few samples, structure between samples, an unclear goal"; 8.5 "Research directions and the quantities to measure". Chapter 9 order: 9.1 distortion, 9.2 inside and between samples, 9.3 random distortion against a specific transformation, 9.4 thirty images.

Chapter 4 renumbering after the new theory papers (commit 454901b → next): 4.7 Zimmermann → 4.9; 4.8 Locatello → 4.10; 4.9 gradient-optimal width → 4.11; 4.10 mean-field → 4.12; 4.11 Silverman → 4.13; 4.12 Hyvärinen → 4.14; 4.13 Vincent → 4.15; Conjecture 4.14 → 4.18. New: 4.7 two-view operator (Luthra 2026), 4.8 NSCL (Luthra 2025), 4.16 first variation of uniformity, 4.17 Gretton et al.

## Decisions

- Symbols: $K$ is the number of negatives everywhere; the RandBit bit count is $b$.
- TC-LeWM: the draft keeps the author's 63.6% → 83.8% (LIBERO, suite-wise, 10 tasks); the RandBit README still says 53.2% → 73.6%, so one of the two needs checking against the paper.
- MotionJEPA ball NMSE: 1.001 for SIGReg (LeWM) against 0.005 for DISReg in 7.4; 1.26 for the forward-only baseline in 8.5. Each number names its baseline.
- Two-view operator: $\mathcal T=\mathcal D^{-1}W$, same eigenvalues as $\bar A$, eigenfunctions $\psi=\mathcal D^{-1/2}u$. Luthra et al. do not link it to HaoChen; the link is this work's.
- Zimmermann and Locatello: latents are $c$, the encoder is $f$, so $z$ and $h$ keep their meaning.
- Symbols: Silverman's bandwidth is $\sigma$; Cheeger's conductance is $\mathrm{cond}_k(G)$ for $k$ parts; a learned edge operator is $\mathcal R(x_i,x_j)$.
- Numbers checked against the papers: Luthra 2026 median absolute difference below 0.05 is $\hat B$ against its label-free spectral reconstruction; Zhu et al. one-ball planning at $H=4$ goes from 1.2% to 90.0%, and Spearman $\rho$ is $-0.0008$ (controlled ball) against $0.40$ (environment ball); Gretton's $\tau=2\sigma^2$; NSCL gap 0.60/0.007 nats at $\tau=1$ and 7.8/3.1 at $\tau=0.2$ for 10/1000 classes.
- Notation: gap $g=q-p$; the predictor has no symbol, $\mathrm{pred}(z)$; generator $\gamma$ with $x=\gamma(c)$; easy feature $a$, useful feature $t$; Saunshi's gap $\delta$; CDNV $\nu$, directional $\tilde\nu$; data density $P$, model $Q$ (score matching, Gretton); noise $\xi$, step $\varepsilon$; Langevin step $\epsilon$ in Hyvärinen 2007; EMA ratio $\beta$, augmentation variance $\omega^2$ and weight decay $\eta$ in the per-mode dynamics of 2.4; concentration $(s,\delta_A)$ for Huang et al.; frequency $K_\xi$ for Turan et al.; kernel bandwidth $\sigma$ for Cao et al.; linear map $M$ in Prop 5.3; task matrix $\Gamma$ with projection $\Pi=\Gamma\Gamma^+$; A-GEM gradients $u$ (task), $v$ (SSL); predictor $\hat y$ in Prop 6.4.
- RandBit counts ($N=256$, $D=128$, $K=510$): whitening needs $2^b-1\ge128$, so $b=8$; instance discrimination needs $2^b\ge N$, so $b=8$; $K2^{-b}\approx2$ at $b=8$; VICReg's invariance and variance terms (weights 25, 25) hold with 16 codes. SIGReg below the untrained encoder at $b=6$ is left as a highlighted open point.

- Figures 2-19 in order of appearance: 2 VICReg comparison, 3 BYOL, 4 Jing collapse, 5 Robinson shortcut, 6 PPS gradient-optimal, 7 Wang-Isola CIFAR-10 on $S^1$, 8 HaoChen graph, 9 PPS principle and circle check, 10 PPS temperature, 11 atlas, 12 InfoMin, 13 I-JEPA architectures, 14-15 RandBit, 16 MAE, 17 jigsaw, 18 ZSSR, 19 Asano single images. Each figure has a visible bold caption line under the image; the alt text is only "Figure N". Paper figures live in `figures/papers/`, converted from the arXiv sources; the PPS ones from `Bandwidth Research/Research/paper/submissions/icomp-2026/figures/`.

## Next steps for the author

- Fill highlighted claims after reading Simon et al. 2023, Tan et al. 2024, Wang et al. 2022.
- RandBit: three seeds near the thresholds, BYOL as the siamese row, guidance at the threshold, reconstruction control, sample-size knob, dimension and batch knobs for the counts of 7.2; MCL and adversarial views as the first remedies to try.
- Read in full the abstract-level papers of 8.1 before removing their highlight.
- Check whether task directions of trained encoders are nearly orthogonal (8.3), which decides whether $\Gamma$ composes as a sum.
- Decide on the references cited nowhere in the text: Ericsson et al. CVPR 2021 and the SPM review, Shwartz-Ziv and LeCun, Bansal, Kaplun and Barak, Korchinski et al., SCOTT, Hinton's transforming autoencoders.
- The remaining `and-chain?` flags are lists, propositions with several assumptions and captions; the remaining `count-announce` flags refer back to the three families of chapter 0.
- Prove or drop Conjecture 4.18 with the anchor-drift experiment of 10.3.
- The spectral scheduling paper behind the end of 4.5 is Turan, Dufour, Ovsjanikov (École Polytechnique, arXiv:2603.09936), not a paper by Chinese authors. Its schedule is proved in $\mathbb R^d$ for a fixed target; the spherical-harmonic version for SSL is a highlighted link of this work in 4.5 and a row of 10.1.
- Cookbook topics the draft does not cover: multi-crop, the uniform prior and unbalanced data, learning-rate schedules and optimizers, weight decay beyond Tian's threshold, batch size, ViT-specific tricks, MIM techniques, evaluation beyond classification, speedups, other modalities, dense prediction, pretraining data. Status in `notes/ideas.md`.
- Reproduce MCL on RandBit before the SIGReg variant (H5, 10.4).
- Read SIE (Garrido et al. 2023) and EquiMod (Devillers and Lefort 2023) before the paragraph on ways to keep the parameter in 9.3 loses its highlight.
