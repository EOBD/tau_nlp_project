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
| 2 ☑ | Sweep v2, slimmed (trained only, extraction only, ~15 GPU min): `after` readouts, isolated-word features (`trained_iso`), per-token chunk positions; then E0 on 27 new site.readouts appended to `runs/mi/e0` (CPU, 4 tasks) | ☑ jobs 968021 (GPU, 22 min) + 968022 (E0, CPU) | RQ1: where the `e1.L0` peak is built; word vs context (root AUC and probes, in context vs alone); filter tokens whose stage-2 `next` saw extra text |
| 3 | E1, slimmed: LDA root metric, strong → weak, supervised letters-only baseline (`mi_experiments/root_metric.py`, `scripts/mi_e1.sbatch`) | ☑ jobs 966212 + 968153 (with `after` / iso sites) | RQ1/H2, RQ2 (second measure), subspace for E4 |
| 5 | E5: PCA + variance decomposition at the key sites, trained vs random (`mi_experiments/pca_sites.py`, `scripts/mi_pca.sbatch`) | ☑ job 968065 (CPU, 3 min) | RQ1 / RQ2 from a third angle: what dominates each site's leading directions |
| 4 | E4, slimmed: remove root subspace at `m.L4.next`, next-word loss vs 20 random subspaces (`mi_experiments/root_ablation.py`, `scripts/mi_e4.sbatch`) | ☑ job 968023 (GPU, 33 min, 2000 verbs) | RQ3/H4 |
| 6 | E6: unvocalized ambiguity: probes on contested forms, in context vs isolated, split by subject position (`mi_experiments/ambiguity.py`, `scripts/mi_ambiguity.sbatch`) | ☑ job 968851 (CPU, 4 tasks, 12 min) | RQ4/H5 |

Cut: standalone letter-control fix (E1's supervised letters baseline replaces it; E0 numbers are reported only as
paired contrasts), E2, E3b, E3c, E4b, E4 preposition breakdown. Reported as limitations.

### E6 results (2026-10-03): context resolves gender ambiguity, from the main network on

Without vowels one spelling can stand for several cells (רוצה m/f, נעשה 3sg/1pl, נמצא past/present). An isolated
word, or its letters, can at best give each form its majority reading (the *ceiling*). On **contested** forms
(seen in UD with >= 2 readings), accuracy above the ceiling must come from context. Full table:
`runs/mi/ambiguity/report.md`; CIs are a cluster bootstrap over host forms (1000 resamples).

**Gender (1,203 contested tokens, 88 forms): H5 supported.** Criterion met at `m.in`, `m.L4`, `d1.resid`, `d1.in`:

| site | contested acc. | − ceiling | − ngram_ctx | context gain (DiD) | gain, subj before | gain, subj after |
|---|---|---|---|---|---|---|
| e1.L0 | .673 | −.058 | +.002 | −.012 | −.034 | +.003 |
| m.in | .805 | +.073 [.046, .106] | +.133 | +.108 [.071, .156] | +.129 | −.017 |
| m.L4 | .767 | +.036 [.014, .063] | +.096 | +.091 [.050, .143] | +.114 | −.017 |
| d1.in | .813 | +.081 [.055, .112] | +.141 | +.111 [.078, .154] | +.140 | −.021 |

- **The causal signature holds:** context helps only when the subject precedes the verb (before − after at `m.in`
  +.146 [.076, .219]; same sign at every site from `m.in` on, largest at `d0.out.last` +.304).
- **Context enters between `e1.L0` and `m.in`** (the stage-1 Mamba layers). `e1.L0` gains nothing and equals the
  random model, consistent with sweep v2: `e1.L0` holds a context-free word summary, the layers after it add context.
- **Both kinds of contested form:** true homographs (397 tokens) `m.in` .846 vs isolated .751, ceiling .756;
  common-gender forms tagged with the subject's gender (היו, 253 tokens) .779 vs .684, ceiling .719.
- **Not shallow cues:** a probe on the clitics plus the two preceding words stays below the ceiling (−.060).

**Other tasks.** *Person* (18 forms): criterion met at `m.in`, `d1.resid`, `d0.out.last`, but by +.03 with lower
bounds .010–.022; consistent with gender, too thin to stand alone. *Tense* (54 forms): context lifts minority
readings (`m.L4` .541 vs isolated .226; DiD +.053 [.001, .119], `d1.dechunk` +.108) but never beats the ceiling.
*Binyan* (22 forms): context lowers accuracy everywhere, also on unambiguous forms (`m.L4` −.061, `d1.dechunk`
−.112), consistent with E5 (binyan fades in the main network). *Number* (6 forms): not interpretable.

**Caveats:** no multiple-comparison correction over 10 sites × 5 tasks (gender's lower bounds would survive one,
person's probably not); contested forms are dominated by a few frequent ones (hence the bootstrap over forms);
the criterion was fixed after a 3-site smoke test (`m.L4` only) and before the full run; decodable, not shown
to be used (no E4-style ablation).

### Final results (2026-10-03): sweep v2, E4, E1 rerun

**H4 (use) supported, with a size caveat.** Removing E1's root subspace (rank 64) from `m.L4` at the chunk after
the verb raises next-word loss by +.040 [.035, .046] nats/char; random subspaces of the same rank +.0046 (20 draws,
range .0032–.0066). Root − mean random = +.036 [.031, .041]; the root edit beats all 20 draws. Removing the whole
vector: +.269. Checks: position cosine median 1.000 (min .996); loss before the edit unchanged (max change 0).
- **By root split:** train roots (whose tokens fitted the subspace) +.072 [.060, .086]; unseen strong roots (test)
  +.015 [.005, .026]; weak roots (heldout) +.017 [.012, .022]. Report the held-out numbers. The effect generalises
  to roots the subspace never saw, but is 4–5× smaller than on fitted roots, so part of the subspace encodes
  specific known words.
- **Size caveat:** the root edit changes the vector more than a random one (‖edit‖ 14.0 vs 8.5). Even if damage
  scaled linearly with edit size (random ×1.65 ≈ .0076), the held-out root effect (.0215) is ~2.8× larger. A
  size-matched control (random directions from the data's high-variance subspace) was not run.

**Context (sweep v2 isolated words): the core findings survive without any sentence.** Paired contrasts on
`trained_iso` (verbs alone as "<verb> ."), strong / weak:
- main network adds root similarity: `m.L4` − `m.in` +.088 [.071, .106] / +.067 [.042, .090] (in context
  +.118 / +.104). About 70% survives without context; context adds +.040 [.022, .057] / +.062 [.036, .088] at
  `m.L4`.
- decoder split: root `dechunk` − `resid` +.137 [.110, .167] / +.115 [.074, .156]; pattern `resid` − `dechunk`
  +.125 [.108, .142] / +.130 [.107, .151].
- `e1.L0` is context-invariant: in context vs alone −.002 [−.008, .005] / +.008 [−.002, .018].

**Where the `e1.L0` peak is built (`after` readout).** The character encoder's state at the space after the word:
in context .839 / .813, far below `e1.L0` (−.103 [−.120, −.088] / −.065 [−.091, −.039]). Alone .925 / .868, about
equal to `e1.L0` (+.019 [.010, .030] / +.001 [−.015, .017]). So the character encoder builds a root-like summary of
the word, but in running text its state is diluted by the preceding context (in context − alone −.086 [−.103,
−.070] / −.056 [−.075, −.038]). The stage-1 attention layer (`e1.L0`) restores a context-independent word summary.
E1: supervised availability at `e0.out.after` .936 (heldout, 2-D), slightly below `e1.L0` .948.

**Updated headline:** the character encoder summarises each word at its boundary; the stage-1 attention layer makes
that summary independent of the preceding context; the main network compresses it towards the root (partly helped
by context), while the pattern travels on the skip path; and the model uses the root direction to predict the next
word, including for roots it was never fitted on.

### Findings so far (2026-10-03, after E0 and E1)

**Headline claim (reframed):** a character-level language model with learned chunking, never told what a root is,
learns to organise Hebrew verbs by root in its main network. The root is already spelled out in the letters, so the
model does not create root knowledge. Its main network compresses each word towards its root while the pattern and
other form detail fade, and pattern information travels separately on the skip path to the decoder.

| finding | evidence (AUC, strong / weak or test / heldout, 95% CI over roots) | strength |
|---|---|---|
| Training makes the model group words by root | E0: trained `m.L4.next` .92 / .83 vs random .48 / .49; difference +.45 [.41, .48] / +.34 [.29, .39] | strong |
| The main network increases how much the root dominates similarity | E0: `m.L4` − `m.in` +.118 [.102, .136] / +.104 [.079, .130] | strong |
| ...without adding root information, while form and pattern detail fade: **compression towards the root** | E1: supervised availability `m.L4` − `m.in` −.032 [−.064, −.004] (heldout, 2-D); E0 pattern (cell) AUC falls `m.in` .84 → `m.L4` .75 → `m.out` .60 | new, from E1 |
| Root and pattern take different paths into the decoder | E0: root `dechunk` − `resid` +.078 [.051, .108]; pattern `resid` − `dechunk` +.254 [.230, .276]. E1: `resid` keeps more recoverable root information (−.135 [−.179, −.097]), i.e. `resid` is form-rich, `dechunk` root-salient | strong |
| No root information beyond the letters (H2 not supported) | E1: best site `e1.L0` beats the n-gram LDA (+.076 [.036, .118] heldout, 2-D) but not `e0.emb.mean` (+.020 [−.016, .053]); no weak class passes the criterion | conclusive for this test |

**Availability vs salience.** E1 measures *availability*: can a trained probe recover the root? Any representation
computed from the letters has the root available, because the root is (almost) a function of the letters. A
representation cannot hold more information about it than its input (data-processing inequality), so H2 as
formulated could hardly have succeeded. That is a design flaw of H2, not a lost finding. E0 measures *salience*:
without training, does the representation's own geometry group same-root words? Letters' geometry barely does (.60
under the 2-D control); the trained main network does strongly (.92); a random network does not (.48). The claim is
about salience: how the model organises information it necessarily has.

**Wherever salience rises, availability falls or stays flat** (main network: E0 +.12, E1 −.03; decoder paths: E0
dechunk > resid, E1 resid > dechunk). Early layers and the skip path carry the full word form, so the root is
recoverable there but is not what makes vectors similar. The main network and `dechunk` discard form detail and
keep the root as the main axis of similarity.

**E5 (PCA, 2026-10-03) agrees, from a third angle.** Excess share of variance inside the top-10 principal
components (root and binyan measured across lemmas):

| site | root | binyan | tense | number | gender |
|---|---|---|---|---|---|
| m.in.next | .047 | .210 | .206 | .105 | .069 |
| m.L4.next | **.198** | .142 | .194 | .015 | .009 |
| m.out.next | .195 | **.040** | .062 | .006 | .003 |
| d1.dechunk.next | **.221** | .044 | .063 | .006 | .004 |
| d1.resid.next | .067 | **.247** | .143 | .150 | .113 |

- In the main network the leading directions switch from pattern / inflection to root: root ×4 from `m.in` to
  `m.L4`, binyan falls ×5 by `m.out`, number and gender almost vanish.
- `dechunk` is root-dominated, `resid` pattern- and agreement-dominated; in `resid`, PC2 = number (excess eta² .75)
  and PC3 = gender (.71) are explicit axes.
- The PC1 × PC2 figure (`runs/mi/pca/figs/pc12_trained.png`) shows clear binyan clusters in `e0.out.mean` and
  `d1.resid`, fully mixed in `m.out` and `dechunk`.
- Random model: root share in the main network about .06–.07 (trained .20–.24).
- **New nuisance finding:** PC1 of `m.L4`, `m.out` and `dechunk` tracks the verb's relative position in the
  sentence (r = +.74, +.70, −.78). It is unrelated to roots (pairs come from different sentences), so it adds noise
  to E0's cosines rather than bias; removing it could sharpen E0 (not done).
- Caveat: these shares have no letter control. Letter-like representations (`e0.emb.mean`, trained or random) give
  the highest root share over all dimensions (.43). Read E5 only as comparisons within the trained model and
  against random, like E0.

**What could still weaken the claim:**
- *Context:* same-root verbs may be grouped because they occur in similar sentences. Sweep v2 (`trained_iso`) tests this.
- *Absolute numbers:* the 2-D control leaks (letters-only .57–.62), so claims rest on paired contrasts (trained vs
  random, site vs site), not on distance from 0.5. Coverage is .28–.40 (strong / test) and .55 (weak / heldout).
- *Use:* everything so far is representational. E4 tests whether the model relies on the root direction for prediction.
- *Generality:* one model, one language, one training run, verbs only.

### Why each remaining experiment is needed

Each step rules out one alternative explanation that the earlier ones leave open.

| after | what we can claim | still open to |
|---|---|---|
| sweep | root / pattern patterns appear across layers | noise, letters, context, unused by-product |
| E0 ☑ | trained ≫ random; the main network adds root similarity (+.12); root and pattern take different decoder paths (with CIs) | letters (the control leaks), context, by-product |
| sweep v2 | the root signal is in the word, not its sentence; built in component X (char encoder or stage-1 attention) | letters, by-product |
| E1 | how much root information is *linearly available* beyond letters, under a supervised letter control; strong → weak transfer | by-product |
| E4 | the model uses root-discriminative directions to predict upcoming text (or does not) | one model, one site |

- **Sweep v2** answers the context objection (verbs run alone, `trained_iso`) and localises the `e1.L0` peak
  (`after` readout).
- **E1** replaces E0's imperfect surface control with a stricter one (a letters-only probe trained the same way)
  and tests generalisation to weak roots never fitted. It also produces the root subspace for E4.
- **E4** is the only causal test: everything else measures what is in the vectors, not what the model relies on.
  Random subspaces of the same rank control for "any edit hurts".

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

**RQ4: Does the model resolve from context what the unvocalized spelling leaves open?** (added 2026-10-03)
Many verb forms are ambiguous without vowels (תכתוב 2ms/3fs, נמצא past/present). Where does the representation
reflect the reading the sentence requires rather than the form's usual reading, and does that depend on whether
the disambiguating cue (the subject) has been seen yet?

Hypotheses as falsifiable claims:

| | Claim | Fails if |
|---|---|---|
| H1 (RQ1) | Some site encodes root identity beyond surface letters | Root similarity falls to the letter floor (§3.2) once letters are controlled |
| H2 (RQ1) | The root code is abstract: it transfers from strong to weak roots | A root metric fit on strong roots does not beat the letter floor on held-out weak roots |
| H3 (RQ2) | Root and pattern are routed separately (`d1.dechunk` vs `d1.resid`) and are linearly separable | No dissociation once CIs are added, or no additive root × pattern structure |
| H4 (RQ3) | The root code is causally used for prediction after the verb | Ablating it hurts next-word loss no more than a random subspace of the same rank |
| H5 (RQ4) | Some site resolves contested forms from context, beyond the form's majority reading and beyond shallow context cues | No site beats both the form-majority ceiling and the `ngram_ctx` probe on contested tokens (E6 criterion) |

Status (2026-10-03): **H1** supported as *salience* (root dominates the geometry beyond letters and random,
paired contrasts) but not as extra information. **H2** not supported (E1, see Findings), and could hardly have been
under this formulation. **H3** supported for routing (E0 + E1 + E5, also without context); linear separability (E3b) cut. **H4** supported
(E4: root − random +.036 [.031, .041]; held-out roots +.015–.017), with the edit-size caveat. **H5** supported
for gender (E6: `m.in` +.073 [.046, .106] over the ceiling, gain only when the subject precedes the verb), weakly
for person, not for tense or binyan.

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

### E0. Measurement hardening (prerequisite for RQ1–3) ☑ (v2 sites pending)

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

### E1. Supervised root metric, strong → weak transfer (RQ1, H1/H2) ☑ (slimmed: LDA only; sites `e1.L0`, `m.in`, `m.L4`, `d1.dechunk`, `d1.resid`, + `e0.out.after` if sweep v2 exists; the supervised letters-only probe doubles as the letter control)

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

- **Smoke test (2026-10-03, 20 resamples, 4 sites; full run in job 966212).** Raw AUC test / heldout: n-gram LDA
  .993 / .979, `e0.emb.mean` .998 / .995, `m.in` .975 / .970, `m.L4` .970 / .938. Within 2-D surface bins: n-gram
  .913 / .872, `e0.emb.mean` .932 / .928, `m.in` .951 / .922, `m.L4` .949 / .890.
  **A trained probe identifies roots from letters alone about as well as from the model's vectors, weak roots
  included.** The H2 criterion will very likely not be met.
  - **Interpretation, availability vs salience:** root identity is linearly *available* in the input letters, so a
    supervised probe cannot separate "abstract root" from "letters". What the model adds is *salience*: without any
    training, its geometry groups same-root words (E0: `m.L4` .92 under the 2-D control), while the letters' own
    geometry does not (letters-only features .60). The claim becomes "the model organises its representation space
    around the root", not "the model holds root information its input lacks".
  - **Full run (2026-10-03) confirms it and adds a pattern:** wherever E0's *unsupervised* root similarity goes up,
    E1's *supervised* root availability goes down or stays flat. Main network: E0 +.12, E1 −.03. Decoder paths:
    E0 dechunk > resid (+.08), E1 resid > dechunk (−.13). Reading: the skip path and early layers carry the full
    word form, so the root is recoverable there by a trained probe but is not the dominant axis of similarity. The
    main network and `dechunk` discard form detail (and pattern, E0 cell AUC falls) but keep the root as the main axis
    of similarity. **The main network does not add root information; it compresses the word towards its root.**
  - The raw (uncontrolled) AUC is at ceiling for letters: most different-root pairs share almost no letters. Only the
    2-D column and the per-class numbers are informative.

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

### E4. Causal use of the root code (RQ3, H4) ☐ (slimmed: `m.L4.next` only, next-word loss, 20 random-subspace controls; no preposition breakdown, no E4b)

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
- **Implementation (2026-10-03):** `mi_experiments/root_ablation.py` edits the output of main-network block L (hook
  on `(hidden, residual)`) at the stage-2 chunk right after the verb. Conditions: root subspace (E1, rank k), 20
  random subspaces of the same rank drawn in the standardized space, full replacement by the train mean (upper bound).
  Records the next-word, rest-of-sentence and before-the-edit NLL changes (the last is a causality check) and the edit
  norms. `--budget-min` caps GPU time. Position check: the intercepted vector is compared with the sweep's saved
  `m.L4.next` (cosine ≈ 1 expected; flagged in the report otherwise). Per verb it also keeps the next word, its
  clean per-char NLL and the per-char change of the root / full / mean-random edits, so the cut governed-preposition
  breakdown can be done later on CPU. Tested on CPU: the projection zeroes the root readout, and the report runs on
  synthetic results. Not yet run on the model.
- **Caveat (from E1):** the root subspace also carries word-form identity (letters make roots linearly
  identifiable). A positive result means the model uses root-discriminative directions, not necessarily an abstract
  root. Compare edit norms: a random subspace may remove less of the vector.
- **Optional E4b, within-word use:** ablate a character-level root subspace at the verb's own characters
  (`e0`/`d0`) and compare the loss on the remaining root letters vs pattern letters.

### E6. Unvocalized ambiguity (RQ4, H5) ☑

- **Goal:** test whether the representation reflects the reading the context requires, for forms whose spelling
  allows several cells, and whether that depends on the cue having been seen (the model is causal).
- **Labels** (derived in the script and aligned to `runs/mi/sweep/tokens.jsonl`; `ud_verbs.jsonl` is not rebuilt):
  per token and task (binyan, tense, person, gender, number), the lexicon status of the spelling: `amb` (lexicon
  gives >= 2 values), `unamb` (1), `common` (none: the form is unmarked, e.g. past 3pl gender), `unknown` (not in
  the lexicon). **Contested** = the host form occurs in the scored tokens with >= 2 gold values, each >= 2 times.
  Subject position from UD (`nsubj*`/`csubj*` dependent of the verb): before / after / none.
- **Method:** logistic-regression probes (`sweep.fit_logreg`), 5-fold cross-validation over morph groups (every
  token gets a held-out prediction; no lemma group in train and test), L2 chosen on fold 0's dev fold. All 25
  feature sets on the same 27,696 tokens.
- **Sites:** §4 sites plus `m.out`, `d1.in`, `d0.out.last`, each in context (`trained`) and isolated
  (`trained_iso`); random model at `e1.L0`, `m.L4`, `d0.out.last`.
- **Baselines:** `ngram` (letters of the host: same ceiling as an isolated word); `ngram_ctx` (host n-grams + the
  clitics + the two preceding words and their last two letters: shallow cues a probe can read off directly).
- **Measures:** accuracy per slice; on contested tokens the form-majority ceiling (oracle), accuracy on
  minority-reading tokens, and the split by subject position and by lexicon status. Context gain = (in context −
  isolated) on contested − the same on unambiguous tokens (DiD). Cluster bootstrap over host forms, shared by all
  sites (paired differences).
- **Pre-registered criterion:** H5 supported at a site if contested accuracy beats the ceiling **and** `ngram_ctx`,
  both 95% CIs excluding 0. Prediction: `e0.*` and isolated features stay at or below the ceiling; gains appear
  where context is integrated; gains vanish when the subject follows the verb.
- **Result:** see "E6 results" at the top. Met for gender (4 sites), person (3, thin), number (3, 6 forms: not
  interpretable); not met for tense or binyan.
- **Not done:** an E4-style ablation of the gender direction (does the model *use* it to predict the agreeing
  word?); a correction for multiple comparisons.

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
| 2026-10-03 | E1 | 966212 | `runs/mi/e1/report.md` | **H2 criterion not met at any site.** Raw AUC at ceiling (n-gram LDA .993 / .979). Within 2-D bins: e1.L0 .989 / .948 beats the n-gram LDA (+.075 [.019, .111] / +.076 [.036, .118]) but not `e0.emb.mean` (+.057 [−.010, .096] / +.020 [−.016, .053]). Supervised availability *falls* through the main network (m.L4 − m.in, 2-D heldout −.032 [−.064, −.004]; m.out .82, dechunk .81 heldout 2-D) and resid > dechunk (−.135 [−.179, −.097]), the opposite of E0's unsupervised results. Trained ≫ random (m.L4 +.30). 2-D coverage .28 test / .55 heldout | H2 not supported beyond letters; supports availability vs salience (see E1 notes) |
| 2026-10-03 | E5 PCA | 968065 | `runs/mi/pca/report.md`, `figs/` | top-10-PC root share m.in .047 → m.L4 .198; binyan .210 → m.out .040; dechunk root .221 / binyan .044, resid root .067 / binyan .247; positional PC1 in m.L4 / m.out / dechunk | H1 (salience), H3 |
| 2026-10-03 | sweep v2 (extraction) + E0 on it | 968021, 968022 | `runs/mi/sweep_v2/feats/`, `runs/mi/e0/report.md` | e0.out.after .839 / .813 in context, .925 / .868 alone; e1.L0 context-invariant; iso m.L4 − m.in +.088 / +.067; decoder split holds alone | H1, H3, context objection |
| 2026-10-03 | E4 | 968023 | `runs/mi/e4/report.md`, `tokens.jsonl` | root +.040, random +.0046, full +.269; root − random +.036 [.031, .041]; held-out roots +.015–.017; ‖edit‖ 14.0 vs 8.5 | H4 supported (size caveat) |
| 2026-10-03 | E1 rerun (`after`, iso sites) | 968153 | `runs/mi/e1/report.md` | e0.out.after .936, iso e1.L0 .947, iso m.L4 .913 (heldout 2-D); criterion still not met anywhere | H2 |
| 2026-10-03 | E6 ambiguity | 968851 (array 0–3, 12 min) | `runs/mi/ambiguity/report.md`, `results.json`, `preds/`, `labels.jsonl` | Gender: criterion met at m.in (+.073 [.046, .106] over ceiling, +.133 over ngram_ctx), m.L4, d1.resid, d1.in; context gain only with the subject before the verb (m.in before − after +.146 [.076, .219]); e1.L0 no gain. Person met at 3 sites by ~+.03. Tense: DiD positive (m.L4 +.053, d1.dechunk +.108) but never above ceiling. Binyan: context lowers accuracy (m.L4 unamb −.061) | H5 supported (gender), weak (person), not (tense, binyan) |

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
| 2026-10-03 | E1 primary measure = AUC within the 2-D surface bins (raw AUC kept but at ceiling); claim reframed as salience (E0) vs availability (E1) | supervised letter features identify roots, weak ones included, about as well as the model's vectors (E1 smoke test) |
| 2026-10-03 | E4: 20 random draws (not 10), full-replacement upper bound, edit norms recorded, GPU budget cap | 95th-percentile criterion needs ≥ 20 draws; norms guard against a "random removes less" artefact |
| 2026-10-02 | Trimmed scope to §0: keep sweep v2, slim E1 and E4; cut control fix, E2, E3b/c, E4b | time-limited; E1's supervised letters baseline serves as the letter control, E0 already gave E3a |
| 2026-10-03 | Added RQ4 / H5 / E6 (unvocalized ambiguity) on the saved sweep and sweep v2 features | nearly free (CPU, no extraction); unlike H2 it asks for information the spelling does not contain; uses the iso features as the no-context control |
| 2026-10-03 | E6 main tasks = tense, person, gender; binyan and number reported as indicative | only 36 / 12 forms are binyan / number-contested in UD vs 106–231 for the others (counts before masking) |
| 2026-10-03 | E6 labels derived in the script, aligned to `tokens.jsonl` with a check; `ud_verbs.jsonl` not rebuilt | a rebuild could reorder rows and break the alignment with the saved features |
| 2026-10-03 | E6: 5-fold CV over morph groups instead of the fixed test split; L2 picked once (fold 0) | contested forms are too few for a 20% test split; one L2 choice keeps the CPU job ~1 min per site |
| 2026-10-03 | E6: added the `common` lexicon status and the contested split by lexicon status (after the smoke test) | the top gender-contested forms (היו, הגיעו) are common-gender past 3pl tagged with the subject's gender: agreement tracking, not homograph resolution; they were mislabelled `unamb` |

## 8. Open questions

- **Next steps that would strengthen the results** (not run, 2026-10-03): (1) size-matched E4 control (random
  directions from the high-variance subspace) and a binyan-subspace control, GPU ~20–30 min; (2) governed-preposition
  breakdown of E4 from the saved per-char data, CPU; (3) E0 with the positional PC removed (E5), CPU.
- **The 2-D surface control leaves letter-only features at .57–.62, not .50** (E0 full run). Not fixed (scope cut); report as a limitation. Options:
  finer grid (with lower coverage), raw instead of standardized letter counts, or replace bin matching with a
  regression / matched-pair design on surface features. Until this is settled, compare sites with each other and
  with the random model (paired contrasts), not with 0.5.
- Does 2-D binning drop mostly weak-root pairs (§E0.2)? If so, the weak-root numbers need a different control,
  e.g. regression on surface features instead of bin matching.
- Are the pre-registered thresholds in E1 and E4 right? Fix them before running those experiments.
- Is a pre-decay checkpoint worth adding as a third model (training-dynamics angle)?
