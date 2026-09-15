# Roots and patterns in the Hebrew H-Net: research questions, experiments, results

Living document for the mechanistic-interpretability follow-up to the probing sweep. Model: the 300M 2-stage
H-Net trained on Hebrew (`runs/pretrain/h300m_he/model.pt`, config `configs/hnet_2stage_300M.json`). Data:
29,402 UD verb tokens (`data/morph/`, built by `mi_experiments/morph_data.py`, split by
`mi_experiments/morph_splits.py`).

Status key: ☐ planned · ◐ running / partial · ☑ done · ✂ cut (kept here for the record). Update the status, the results log (§6) and the decision
log (§7) after every run.

## 0. Current scope (trimmed 2026-10-02, time-limited)

| step | experiment | status | answers |
|---|---|---|---|
| 1 | E0 on saved features | ☑ | RQ1 (relative), RQ2 dissociation |
| 2 | Sweep v2, slimmed (trained only, extraction only, ~15 GPU min): `after` readouts, isolated-word features (`trained_iso`), per-token chunk positions; then E0 on 27 new site.readouts appended to `runs/mi/e0` (CPU, 4 tasks) | ☐ submit `bash scripts/mi_submit_v2.sh` (gpu-research account) | RQ1: where the `e1.L0` peak is built; word vs context (root AUC and probes, in context vs alone); filter tokens whose stage-2 `next` saw extra text |
| 3 | E1, slimmed: LDA root metric, strong → weak, supervised letters-only baseline | ☐ | RQ1/H2, RQ2 (second measure), subspace for E4 |
| 4 | E4, slimmed: remove root subspace at `m.L4.next`, next-word loss vs 10 random subspaces | ☐ needs E1 | RQ3/H4 |

Cut: standalone letter-control fix (E1's supervised letters baseline replaces it; E0 numbers are reported only as
paired contrasts), E2, E3b, E3c, E4b, E4 preposition breakdown. Reported as limitations.

---

## 1. Research questions

**RQ1: Does the model represent the root as an abstraction, and where?**
Is there a site where tokens of one root are close *beyond what their shared letters explain*, and does that
code generalise to roots the analysis never fit, in particular weak roots, where root letters drop out of the
surface form?

**RQ2: Does the hierarchy split root from pattern?**
The sweep suggests the main-network path down to the stage-1 decoder (`d1.dechunk`) carries root information and
little pattern information, and the skip path (`d1.resid`) carries the reverse. Is that split real, and are root
and pattern stored as separable (roughly additive) components that the decoder combines?

**RQ3: Does the model use root information to predict upcoming text?**
Decodable doesn't mean used. Does removing the root code make the model worse at predicting what follows the
verb, more than removing a random code of the same size, and specifically where the verb's lexeme determines the
continuation?

Hypotheses as falsifiable claims:

| | Claim | Fails if |
|---|---|---|
| H1 (RQ1) | Some site encodes root identity beyond surface letters | Root similarity falls to the letter floor (§3.2) once letters are controlled |
| H2 (RQ1) | The root code is abstract: it transfers from strong to weak roots | A root metric fit on strong roots does not beat the letter floor on held-out weak roots |
| H3 (RQ2) | Root and pattern are routed separately (`d1.dechunk` vs `d1.resid`) and are linearly separable | No dissociation once CIs are added, or no additive root × pattern structure |
| H4 (RQ3) | The root code is causally used for prediction after the verb | Ablating it hurts next-word loss no more than a random subspace of the same rank |

---

## 2. Architecture and readouts (what a "site" is)

Layout `m4 [T1m4 [T17] m4T1] m4`, so:

| prefix | module | layers | resolution |
|---|---|---|---|
| `e0` | stage-0 encoder | 4 Mamba | character |
| `e1` | stage-1 encoder | L0 = attention, L1–4 = Mamba | stage-1 chunk |
| `m` | main network | 17 attention | stage-2 chunk |
| `d1` | stage-1 decoder | `dechunk` (upsampled main output), `resid` (skip path), `in` = sum; L0–3 Mamba, L4 attention | stage-1 chunk |
| `d0` | stage-0 decoder | same structure, 4 Mamba, feeds `lm_head` | character |

The model is causal, and a chunk vector is the state at the chunk's first character. So:

- **`cur`** (the chunk containing the word) has not seen the word. At `m.*` it is at chance for every task. **Use
  `next`** (the first chunk after the word) for all chunk sites.
- Stage-1 boundaries fall right after the verb in 94% of tokens, so `e1/d1 .next` is close to the state at the space
  after the word.
- Consequence for RQ3: root information at `m.*.next` can only affect predictions **after** the verb, not the verb's
  own letters. Within-word use would have to go through the character path (`e0`/`d0`).

---

## 3. What the sweep established (job 956463, 2026-09-30) and later checks (2026-10-01)

Full tables: `runs/mi/sweep/report.md`. Code: `mi_experiments/sweep.py`, `scripts/mi_sweep.sbatch`.

### 3.1 Root similarity curve (`next`, AUC same-root > different-root, 1-D surface-binned; strong / weak roots)

| site | trained | random | cell AUC (pattern) | binyan probe acc |
|---|---|---|---|---|
| char n-gram baseline | .528 / .537 | | .577 | .695 |
| e0.emb.mean (letter floor, see 3.2) | .868 / .810 | .850 / .790 | .639 | .557 |
| **e1.L0** (attention) | **.953 / .922** | .522 | .755 | .850 |
| e1.L3 | .860 / .818 | .522 | **.826** | .831 |
| e1.out = m.in | .827 / .786 | .52 / .51 | .809 | .835 / .824 |
| **m.L4** (main peak, L5 equal) | **.931 / .875** | .511 | .720 | .766 |
| m.L8 | .913 / .857 | .511 | .667 | .739 |
| m.out | .827 / .766 | .511 | .587 | .683 |
| **d1.dechunk** | .841 / .785 | .534 | **.589** | .692 |
| **d1.resid** | .775 / .721 | .500 | **.826** | .803 |
| d1.in | .867 / .806 | .534 | .745 | .798 |

Shape: root similarity peaks entering stage 1, drops through the stage-1 Mamba layers (while pattern information
rises), recovers in the first third of the main network, then fades. Pattern information falls steadily through the
main network. On the decoder side, `dechunk` is root-heavy and pattern-poor, `resid` the reverse.

### 3.2 Caveats (all must be handled before any claim)

1. **The surface control leaks.** Binning on n-gram cosine removes only n-gram similarity. A mean of *random-init*
   character embeddings still scores .87 / .81. Stricter 2-D binning (n-gram cosine × bag-of-letters, 10 × 10) brings
   random controls to about .50–.58. Under it, trained `e0.emb.mean` = .78 / .74 (**this is the floor to beat**),
   and `e1.L0.next` = .94 / .90, but only 49% of same-root pairs fall in usable bins. Prototype code lived in a
   scratchpad. It must be ported into the repo (E0).
2. **Same-sentence context is ruled out**: only 0.1% of same-root pairs share a sentence. Similar-but-different
   contexts (distributional similarity of the lexemes) are **not** ruled out.
3. Hardest weak classes: hollow (.86) and pe-nun (.91).
4. The binyan "ambiguous > unambiguous" gap is a class-mix artefact: 53% of ambiguous forms are PAAL vs 36% of
   unambiguous.
5. Single seed, single split, no confidence intervals. Differences under ~.02 between neighbouring sites are not
   interpretable yet.
6. IAHLT treebanks overlap the pretraining domain (Knesset, Wikipedia). HTB is the only out-of-domain treebank,
   so report HTB separately where n allows.

---

## 4. Sites selected for follow-up

| role | site (`next` unless noted) | why |
|---|---|---|
| letter floor | `e0.emb.mean` | no contextual processing; anything not above it is letters |
| char-encoder summary | `e0.L3/out` **at the space after the word** (new readout `after`) | not measured by the sweep; decides whether `e1.L0`'s peak is built by the char encoder |
| overall peak | `e1.L0` | .953 / .922 |
| main-network input | `m.in` | trough; baseline for what the main network adds |
| main-network peak | `m.L4` | .931 / .875, pattern already fading |
| main-network output | `d1.dechunk` | root-heavy, pattern-poor signal into the decoder |
| skip path | `d1.resid` | pattern-heavy counterpart |
| merge | `d1.in` | root + pattern combined |

Every site is also run on the random-init model as a control.

---

## 5. Experiments

Conventions: code in `mi_experiments/<name>.py`, Slurm script `scripts/mi_<name>.sbatch` (self-contained and
resumable: someone else submits the jobs), outputs in `runs/mi/<name>/`. Reuse `sweep.py`
(`load_model`, `Recorder`, `token_vectors`, `fit_logreg`, `binned_auc`) and the saved features in
`runs/mi/sweep/feats/` where possible. Bootstrap CIs resample **roots**, not pairs or tokens (pairs that share a
root are not independent). 1,000 resamples, 95% percentile intervals.

### E0. Measurement hardening (prerequisite for RQ1–3) ◐

- **Goal:** a root-similarity measure that random features cannot pass, with error bars and the missing readout.
- **Changes:**
  1. Port 2-D surface binning (n-gram cosine × bag-of-letters) into `sweep.py`, keep 1-D as a secondary column.
  2. Report coverage (fraction of same-root pairs in usable bins) per group **and per root class**. Check whether
     the dropped pairs are mainly weak-root / low-overlap pairs, since those are the interesting ones.
  3. New char readout `after` = state at the character after the word (the space), for all `e0.*` and `d0.*` sites.
  4. Bootstrap CIs over roots for root AUC and cell AUC.
  5. Root AUC per weak class (hollow, pe-nun, final-he, geminate, quadriliteral, ...).
- **Sites:** all (cheap; features for existing readouts are already on disk, only `after` needs a new extraction).
- **Reading the result:**
  - `e0.out.after` ≈ `e1.L0.next` → the character encoder builds the root-like summary, stage 1 dilutes it and
    the main network restores it.
  - `e0.out.after` ≪ `e1.L0.next` → the stage-1 attention layer builds it.
- **Implementation (2026-10-02):** items 1, 2, 4, 5 in `mi_experiments/roots_e0.py` (CPU, reads saved features;
  `scripts/mi_e0.sbatch`, 8-task array on `studentkillable`). Item 3 (`after` readout) in `sweep.py`. It needs
  a new extraction on an Ampere+ GPU: `bash scripts/mi_submit_v2.sh` from an account with `gpu-research` runs sweep v2
  and then E0 on it.
- **Output:** `runs/mi/e0/report.md` (original features), `runs/mi/sweep_v2/report.md` + `runs/mi/e0_v2/report.md`
  (with `after`). Bootstrap replicates are in `runs/mi/e0*/boot/` for further paired contrasts.
- **Smoke-test observations (2026-10-02, 50–200 resamples; confirmed by the full run, see §6):**
  - Trained `e1.L0.next` .94 / .88 and `m.L4.next` .92 / .83 under the 2-D control. Random at the same sites:
    .45–.49.
  - `m.L4 − m.in` = +.12 [.10, .14] strong, +.10 [.07, .13] weak. The main network adds root similarity beyond
    its input.
  - **The control is not clean.** Under its own 2-D bins, the letters-only baseline scores .60 / .62 and the
    n-gram baseline .57 / .52. `e0.emb.mean` drops to .53 / .40 (below 0.5 for weak roots). This does not match
    the 2026-10-01 scratchpad numbers (`e0.emb.mean` .78 / .74). The definitions differ: here both bin
    variables are cosines of *standardized* features, edges at same-root quantiles, MIN_N = 20. Only 39% (strong)
    / 55% (weak) of same-root pairs are covered. See §8.

### E1. Supervised root metric, strong → weak transfer (RQ1, H1/H2) ☐ (slimmed: LDA only; sites `e1.L0`, `m.in`, `m.L4`, `d1.dechunk`, `d1.resid`, + `e0.out.after` if sweep v2 exists; the supervised letters-only probe doubles as the letter control)

- **Goal:** test whether a root code learned on strong roots identifies weak roots it never saw. Cosine over the
  full vector can hide a small root subspace. A learned projection can't hide it, but it has to generalise.
- **Method:** at each site, fit a rank-k linear projection (k ∈ {16, 32, 64}, picked on dev) on `root_split=train`
  (strong + guttural roots) to pull same-root tokens together: LDA, or a contrastive loss on pairs. Evaluate on
  `root_split=test` (unseen strong roots) and `heldout` (weak roots): surface-binned same-root AUC and same-root
  retrieval mAP. The roots are unseen in both, so this measures a general "same root" metric, not memorised root
  classes.
- **Controls:** the same probe on n-gram features, on `e0.emb.mean`, and on the random model at each site.
- **Sites:** §4 list.
- **Predictions:**
  - H2 true: weak-root AUC at the peak site beats the probe-on-`e0.emb.mean` floor by a clear margin, including
    for hollow and pe-nun (letters missing from the surface).
  - Form-based code: transfer works for strong-test, collapses on hollow / pe-nun.
- **Pre-registered criterion (proposal):** H2 supported if the weak-root AUC at some site exceeds the
  `e0.emb.mean` probe by ≥ .05 with the 95% CI excluding 0, and does so in ≥ 4 of the weak classes with ≥ 30
  same-root pairs.

### E2. Root vs meaning (RQ1, H1 alternative) ✂

- **Goal:** separate "same root" from "related meaning", since same-root lexemes are often semantically related.
- **Option A (automatic):** add an external semantic similarity for each pair (static Hebrew word embeddings of the
  two lemmas, e.g. fastText) and regress site cosine on `same_root + surface bins + semantic similarity`. Root
  coding predicts a same-root effect that survives semantic similarity.
- **Option B (curated):** 50–100 pairs each of (a) same root, related meaning, (b) same root, unrelated meaning,
  (c) different root, similar meaning (synonyms). Root coding predicts b > c. Meaning coding predicts c > b. Needs
  native-speaker judgement.
- **Sites:** `e1.L0`, `m.L4`, `d1.dechunk`, `d1.resid`. Possible outcome: early sites form/root-like, `m.L4`
  meaning-like.

### E3. Root / pattern dissociation and composition (RQ2, H3): E3a ☑ via E0 contrasts; E3b, E3c ✂

- **E3a, dissociation with error bars:** root metric (E1) and pattern probes (binyan, full cell) at `m.in`, `m.L4`,
  `d1.dechunk`, `d1.resid`, `d1.in`, with bootstrap CIs. H3 predicts root(dechunk) > root(resid) and
  pattern(resid) > pattern(dechunk), both with CIs excluding 0, and not present in the random model.
- **E3b, additive structure (no training):** average vectors per (root, binyan) cell over tokens. For roots X, Y
  attested in binyanim P, Q, test parallelism: cos( v(X,P) − v(X,Q), v(Y,P) − v(Y,Q) ) vs shuffled-cell
  baselines, and analogy accuracy (v(X,P) − v(X,Q) + v(Y,Q) → nearest is v(Y,P)?). Compositional root × pattern
  coding predicts high parallelism at the merge (`d1.in`) and in the main network. Control the tense/person/gender/
  number cell (match it, or average within binyan over matched cells).
- **E3c, interchange intervention:** patch `d1.dechunk.next` from verb A into the forward pass of verb B (same
  binyan, different root) and measure the shift in log-prob toward A's actual continuation vs B's. Repeat patching
  `d1.resid`. H3 predicts dechunk patches carry lexeme-dependent continuation preferences (governed prepositions,
  collocates) and resid patches carry form-dependent ones.

### E4. Causal use of the root code (RQ3, H4) ☐ (slimmed: `m.L4.next` only, next-word loss, 10 random-subspace controls; no preposition breakdown, no E4b)

- **Goal:** test whether the model needs the root code to predict what follows the verb.
- **Method:** project the E1 root subspace out of the residual stream at the verb's `next` chunk (`m.L4`, and
  separately `e1.L0`), using LEACE or plain orthogonal projection. Run the rest of the model and measure the change
  in per-character loss on the following word.
- **Where the root should matter:** positions whose identity depends on the verb lexeme, e.g. the governed
  preposition (התייחס **ל**, השתתף **ב**, פחד **מ**), and the object or complement. Compare against positions
  where it shouldn't (punctuation, far-away words).
- **Controls:** random subspace of the same rank (≥ 10 draws), binyan subspace of the same rank, and full
  mean-ablation of the vector as an upper bound.
- **Pre-registered criterion (proposal):** H4 supported if root-subspace ablation raises next-word loss more than
  the 95th percentile of random-subspace ablations, with the effect larger at lexeme-governed positions than
  elsewhere (CI over sentences).
- **Optional E4b, within-word use:** ablate a character-level root subspace at the verb's own characters
  (`e0`/`d0`) and compare the loss on the remaining root letters vs pattern letters.

### Order and dependencies

```
E0 ──► E1 ──► E3a, E3b ──► E4 (needs E1's subspace)
        └──► E2 (A any time after E0; B needs curated pairs)      E3c after E3a
```

E0 first: it is cheap, and its `after` readout decides whether RQ1 is about the main network or the char encoder.

---

## 6. Results log

One entry per run: date, job id, command, output path, headline numbers, and which hypothesis it bears on.

| date | exp | job | output | result | bears on |
|---|---|---|---|---|---|
| 2026-09-30 | sweep | 956463 | `runs/mi/sweep/report.md` | §3.1 | H1 (suggestive) |
| 2026-10-01 | sweep checks (scratchpad) | – | not saved | §3.2 items 1–4 | measurement |
| 2026-10-02 | E0 on sweep features | 963603 (array 0–7, done, ~1 h) | `runs/mi/e0/report.md` | 2-D control, 1000 root resamples. Root AUC strong / weak: e1.L0 .94 / .88, m.in .81 / .73, m.L4 .92 / .83, m.out .82 / .73; random ≤ .50. m.L4 − m.in +.118 [.102, .136] / +.104 [.079, .130]. dechunk − resid (root) +.078 [.051, .108] / +.089 [.049, .129]; resid − dechunk (pattern) +.254 [.230, .276] / +.246 [.225, .268]. Hollow and pe-nun are hardest (e1.L0 .82 / .84, m.L4 .76 / .76) but far above e0.emb.mean (.48 / .40). Caveats: letters-only baseline .60 / .62; coverage .39 / .55 | H1 supported (relative), H3 dissociation supported, H2 suggestive |
| 2026-10-02 | sweep v2 + E0 v2 (`after` readout) | not yet submitted | `runs/mi/sweep_v2/`, `runs/mi/e0_v2/` | needs `bash scripts/mi_submit_v2.sh` from a gpu-research account | H1 (where the peak is built) |

---

## 7. Decision log

| date | decision | why |
|---|---|---|
| 2026-09-30 | Throw out unreliable weak-root labels rather than hand-check | weak roots are the key letters-vs-root test, so labels must be gold |
| 2026-09-30 | Root split: train on strong + guttural, hold out weak + quadriliteral classes | tests out-of-class generalisation |
| 2026-10-01 | Use `next` for all chunk sites, drop `cur` from conclusions | `cur` at `m.*` has not seen the word |
| 2026-10-01 | Letter floor = `e0.emb.mean` under 2-D binning, not chance or the n-gram baseline | random features pass 1-D binning |
| 2026-10-02 | Adopted RQ1–3 / H1–H4 and the E0–E4 plan in this document | |
| 2026-10-02 | Bootstrap resamples shared across all sites and models (fixed seed) | site-vs-site and trained-vs-random differences get paired CIs |
| 2026-10-02 | Added to sweep v2: isolated-word extraction (each host alone as "<host> .") and `positions.npy` | the only remaining GPU-only data a later analysis could need: a cheap context control (replaces part of the cut E2) and a way to filter the 7.4% of tokens whose stage-2 `next` chunk does not start right after the word. Identical-host pairs: see the next entry |
| 2026-10-02 | Slimmed sweep v2: trained model only, extraction only, contextual run saves only `after`; no GPU probes; E0 on selected sites only, appended to `runs/mi/e0`. Identical-host pairs kept (not dropped) | GPU budget ~2 h for v2 + E4 today; sharing the v1 pair sample and resamples keeps contrasts across runs paired. Lost: random-model `after`/iso controls and the context-vs-isolated probe table |
| 2026-10-02 | Trimmed scope to §0: keep sweep v2, slim E1 and E4; cut control fix, E2, E3b/c, E4b | time-limited; E1's supervised letters baseline serves as the letter control, E0 already gave E3a |

## 8. Open questions

- **The 2-D surface control leaves letter-only features at .57–.62, not .50** (E0 full run). Not fixed (scope cut); report as a limitation. Options:
  finer grid (with lower coverage), raw instead of standardized letter counts, or replace bin matching with a
  regression / matched-pair design on surface features. Until this is settled, compare sites with each other and
  with the random model (paired contrasts), not with 0.5.
- Does 2-D binning drop mostly weak-root pairs (§E0.2)? If so, the weak-root numbers need a different control,
  e.g. regression on surface features instead of bin matching.
- Are the pre-registered thresholds in E1 and E4 right? Fix them before running those experiments.
- Is a pre-decay checkpoint worth adding as a third model (training-dynamics angle)?
