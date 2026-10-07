# Ledger: diploma draft

Target: `draft-1.md`, whole file.

## Constraints

- Audience: readers outside the author's life who want to see the result: a supervisor, a committee, a researcher in SSL. They know ML and have not seen any chat.
- Where the text goes: a diploma thesis draft. It may promise planned experiments only as open items marked with `==...==`.
- Length: about 60 pages for the final thesis. The current draft is a skeleton, about 35-40 pages of Markdown.
- Math syntax: `$...$` and `$$...$$` in Markdown.
- Language and register: English, formal, single author, so no "we" and no "I" in the text; the work and its objects are the subjects.
- Structure: hierarchy and sequence first. Every section opens by saying why it is there and how it follows from the previous one. Headings alone must read as a plan of the work.
- Highlights: `==...==` marks what is not done, not read, not checked, or open. Proven statements with proofs in the text stay plain.

## Sources

- Primary artifact: `~/VSCodeProjects/Bandwidth Research/Research/paper/submissions/icomp-2026/main.tex` and `references.bib` (PPS paper; read: yes).
- RandBit experiment: `experiments/randbit/README.md`, `experiments/randbit/reports/randbit_table.md` (read: yes).
- Robinson reproduction notes: `~/VSCodeProjects/ssl-robinson-ifm/notes`.
- Critiques of Wang-Isola: ChatGPT dialogue https://chatgpt.com/share/6a45a21f-c0fc-83ea-8f67-112af747a159 (read: yes). Nie et al. 2023, Wang-Liu 2021, Jing et al. 2022, Fang et al. 2024, Xue et al. 2023, RankMe.
- Obsidian notes: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Vlad Otpad's Vault/SSL` ("SSL Theory.md", "SSL on Tiny Sample with a Tiny Model.md").
- Previous version before the restructure: `draft-1.previous.md` (includes the author's edits of 7 Oct, 15:00); delete once the new structure is accepted. The mapping below translates its section and proposition numbers.

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

## User phrasings

- U1: "Why are we even talking about it": every section and proposition opens with its reason. Used as a structural rule, not as a phrase.
- U2: "unify contrastive, regularizers and siamese approaches": used in §0 and §2.
- U3: "waves of different approaches, grouped into 3 groups": used in §5 opening.
- U4: Wang-Isola "a very cool and influential paper" that "still fails to cover all properties of original infonce loss": used in §4.2 opening, in register.

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

## Section map from the previous structure

Old 2.1 → 2.2 and 5.2; 2.2 → 4.2; 2.3.1 → 3.5; 2.3.2 → 3.2; 2.3.3-2.3.4 → 5.2; 2.3.5 → 6.2; 2.3.7 → 4.2; 2.3.8 → 6.5; 2.4 → 4.3; 2.5 → 3.1, 3.7, 4.1; 2.6 → 4.5, 5.2; 3.1 → 2.3, 2.4, 5.3, 5.4; 3.2 → 6.3; 3.3 → 2.5; 3.4 → 7; 4.1 → 2.3; 4.2 → 5.1; 4.3-4.4 → 2.3, 2.5; 4.5-4.6 → 6.4; 4.7 (author's "Where a method intervenes") → 5.5, its Prop. 4.6 → Prop. 5.3; 5 → 6; 6.1-6.4 → 4.5, 4.6; 6.5 → 3.8; 6.6 → 2.6; 7 → 8; 8 → 9; 9.1 → 7; 9.2-9.7 → 10.1-10.7.

Old proposition numbers used outside the draft: 3.4 (multiset invariance) → 2.11, 6.2 (A-GEM) → 8.2. `experiments/randbit/README.md` cites the new numbers.

## Decisions

- Symbols: $K$ is the number of negatives everywhere; the RandBit bit count is $b$.
- TC-LeWM: the draft keeps the author's 63.6% → 83.8% (LIBERO, suite-wise, 10 tasks); the RandBit README still says 53.2% → 73.6%, so one of the two needs checking against the paper.
- MotionJEPA ball NMSE: 1.001 for SIGReg (LeWM) against 0.005 for DISReg in 7.4; 1.26 for the forward-only baseline in 8.5. Each number names its baseline.

## Next steps for the author

- Fill highlighted claims after reading Tian 2022, Garrido 2023 duality conditions, Simon et al. 2023, Tan et al. 2024, Wang et al. 2022.
- RandBit: three seeds near the thresholds, BYOL as the siamese row, guidance at the threshold, reconstruction control, sample-size knob.
