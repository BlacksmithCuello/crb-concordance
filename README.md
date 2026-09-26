# CRB-Concordance

Discovery code for the study "Knowledge-Graph-Augmented Multi-Agent Systems for
Discovering Metabolic Vulnerabilities in Radioresistant Colorectal Cancer".

Four evidence-producing agents — knowledge-graph path retrieval, CRISPR/drug
functional dependency, genome-scale flux feasibility and transcriptomic-clinical
association — each turn their raw evidence into a Dempster-Shafer mass function on
the frame `Theta_g = {V, not V}`. A critic/verifier discounts each mass by a
calibrated reliability rate, combines them with a conflict-redistributing
sequential rule that routes pairwise conflict into the uncommitted mass instead of
renormalising it away, and reports a belief interval, a pignistic point estimate
and an auditable conflict trace for every candidate. Ranking uses the pignistic
point estimate; the interval width carries the cross-modal disagreement that a
single score cannot express.

## Project context

| Field | Value |
| --- | --- |
| Paper | Knowledge-Graph-Augmented Multi-Agent Systems for Discovering Metabolic Vulnerabilities in Radioresistant Colorectal Cancer |
| Project name | `Knowledge_Graph_Augmented_Multi_Agent_Systems_for_Discovering_Metabolic_Vulnerabilities_in_Radioresistant_Colorectal_Cancer` |
| Importable package | `crb_concordance` |
| Domain | computational oncology — in silico target discovery on a genome-scale metabolic model with belief-function evidence fusion |
| Framework | Python 3.11, NumPy 2.4, SciPy 1.17 (HiGHS linear programming), PyTorch 2.11 for the calibration-stage fit only |
| Venue | npj Digital Medicine |
| Primary resources | 18 public resources plus two private clinical arms (see `dataset_urls.txt`, `NOTICE`) |
| Reported compute | the manuscript reports no hardware; the executed study is the in silico simulation, and every entry point runs on CPU |
| Hyperparameter reference | Supplementary Table S1 |
| Deviations | recorded in `claim_to_code.json` under `deviations` |

## What the release implements

| Statement | Location | Code |
| --- | --- | --- |
| Mass function on `Theta_g`, Eq. (1) | Sec. 4.1 | `belief/mass.py` |
| Shafer discounting, Eq. (2) | Sec. 4.1 | `belief/discount.py` |
| Conflict-redistributing combination, Eq. (3), Algorithm 2 | Sec. 4.1-4.2 | `belief/combine.py` |
| Belief, plausibility, pignistic, Eq. (4) | Sec. 4.1 | `belief/mass.py` |
| Conflict trace and flagging threshold | Sec. 4.1, Table S1 | `belief/trace.py` |
| Proposition 1, conflict monotonicity | Sec. 4.1 | `belief/guarantees.py` |
| Proposition 2, pignistic convergence to the fixed-discount floor | Sec. 4.1 | `belief/guarantees.py` |
| Substitution tier rules | Sec. 4.5 | `belief/fusion.py` |
| Knowledge-graph agent, bounded simple paths, pathway-level support | Sec. 4.2, 4.4 | `agents/kg_path.py`, `graph/paths.py` |
| Dependency agent | Sec. 4.2 | `agents/dependency.py` |
| Flux-feasibility agent, alternative-pathway feasibility | Sec. 4.2 | `agents/flux_feasibility.py`, `metabolism/knockout.py` |
| Transcriptomic-clinical agent | Sec. 4.2 | `agents/transcriptomic_clinical.py` |
| Critic/verifier, structural checks | Sec. 4.2 | `agents/verifier.py` |
| Fixed evidence-to-mass map | Algorithm 3 | `agents/mapping.py`, `calibration/mass_calibration.py` |
| Four-fold calibration panel | Sec. 4.1 | `calibration/folds.py`, `calibration/discount_fit.py` |
| Provenance ledger | Sec. 4.2 | `discovery/ledger.py` |
| Discovery pass, Algorithm 1 | Sec. 4.2 | `discovery/pipeline.py` |
| Knowledge-cutoff, circularity and edge-holding controls | Sec. 4.4 | `graph/leakage.py` |
| Metabolic reaction edges from Human-GEM v2.0.1 | Sec. 4.3 | `graph/metabolic_edges.py` |
| Context-specific model, flux balance, variability, sampling | Sec. 4.2-4.3 | `metabolism/model.py`, `metabolism/fba.py`, `metabolism/sampling.py` |
| Recall@k, precision@k, MRR, per-control hit rate | Sec. 4.4 | `metrics/ranking.py` |
| Expected calibration error, Brier score | Sec. 4.4, 4.6 | `metrics/calibration.py` |
| Provenance-grounding and hallucinated-edge rates | Sec. 4.4 | `metrics/provenance.py` |
| Bootstrap intervals, Cochran's Q, reader cluster bootstrap | Sec. 4.6 | `metrics/comparison.py` |
| DeLong inference for correlated curves | Sec. 4.6 | `stats/delong.py` |
| Holm-Bonferroni and Benjamini-Hochberg | Sec. 4.6 | `stats/multiplicity.py` |
| Stratified Cox with competing risks | Sec. 4.6 | `stats/survival.py` |
| Reader-study marginal logistic model | Sec. 4.6 | `stats/gee.py` |
| Hanley-McNeil prospective sizing | Sec. 4.6 | `stats/power.py` |
| Simulation grid and seeds | Table S1 | `simulation/design.py` |
| Falsification inequality, H1, H2 | Sec. 4.4-4.5 | `simulation/contrasts.py` |
| Four extraction levels plus substitution tier | Sec. 4.5 | `evaluation/ablations.py` |
| Benchmark table and operator bindings | Table 1 | `benchmark/registry.py`, `benchmark/baselines/operators.py` |
| Clinical comparator table, all cells pending | Table 2 | `evaluation/reporting.py` |
| Cohort design marginals and exclusion criteria | Sec. 4.6 | `cohorts/schema.py`, `cohorts/generator.py` |
| Control panel and its declared roles | Sec. 4.1 | `cohorts/control_panel.py` |

## Installation

pip:

```
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

conda:

```
conda env create -f environment.yml
conda activate crb-concordance
python -m pip install --no-deps -e .
```

container:

```
docker build -t crb-concordance .
docker run --rm crb-concordance crb-size-prospective
```

## Data

No third-party data is bundled. `dataset_urls.txt` lists every resource that was
fetched from a machine and read back, with its licence and release; `NOTICE` records
the same resources with their access route and role.

`crb_concordance.cohorts.public_cohorts` declares the expected on-disk path of each
resource and provides a schema reader for each: the GDC clinical export, a GEO
series matrix, a DepMap CRISPR gene-effect matrix, a drug-response table and the
substrate edge table. Point the readers at a downloaded copy under `data/` using
`scripts/prepare_data.sh`; a resource that is absent raises `SourceUnavailable`,
which the verification layer reports as `NOT_RUN` rather than turning into a value.

Two resources are held under their own agreements and are never redistributed: the
three-site retrospective arm and the three-site prospective arm. The in vitro
corroboration is laboratory work. Every cohort-level quantity in this release is
therefore reported as `NOT_RUN`, and the executable stand-ins are:

- `cohorts/generator.py` produces records carrying exactly the declared design
  marginals: three sites across three regions, 3,300 retrospective records at about
  1,000-1,200 per site, 825 prospective records, a pCR rate of 25%, and the declared
  exclusion criteria of insufficient RNA yield and missing outcome ascertainment.
- `graph/generator.py` builds a substrate carrying the declared node kinds, relation
  types, release-dated edges, the named pathways and the metabolic reaction layer.

## Running the studies

Calibration and discovery:

```
python -m crb_concordance.cli.calibrate --experiment main --output-root runs
python -m crb_concordance.cli.discover  --experiment main --output-root runs
```

The in silico simulation study of Supplementary Table S1 (20 seeds, 2,000 candidates
per seed, 15% prevalence, the 3x3 conflict and absent-evidence grid, 5,000
seed-level resamples):

```
python -m crb_concordance.cli.simulate --experiment main --output-root runs
```

The benchmark harness and the ablation battery:

```
python -m crb_concordance.cli.benchmark --experiment supplementary_edge_holdout --output-root runs
python -m crb_concordance.cli.ablate    --experiment ablation_substitution_dempster --output-root runs
```

The prospective-arm sizing calculation:

```
python -m crb_concordance.cli.size_prospective --experiment supplementary_prospective_sizing
```

Every report is written as `.json` plus a plain-text `.txt` rendering under
`runs/<experiment>/`.

## Results this release produces

The manuscript states that the in silico simulation is the study carried out and
that the cohort arms and the laboratory corroboration are pre-specified designs.
What is executed here is therefore the simulation, its contrasts, the ablation
battery and the statistical plan. Tolerances are stated as the spread over the
declared seeds or as an exactness bound.

- **Calibration.** Four leave-one-out folds over the four calibration controls with
  SLC16A1 withheld. The pooled rates are printed by `crb-calibrate`; the closed-form
  minimiser of the panel Brier loss and the derivative-free minimiser agree to better
  than 1e-3 on every modality.
- **Propositions.** Proposition 1 holds on a 26-point scan: the interval width rises
  from about 0.33 to about 0.44 as the accumulated conflict rises, with no decrease
  anywhere. Proposition 2 holds to 1e-5: in the zero-conflict limit the pignistic
  bias equals half the product of the per-modality knowledge gaps exactly, and it
  vanishes when any rate reaches one.
- **Flux feasibility.** The reduced reconstruction attains a positive growth optimum
  with a mass-balance residual below 1e-15, and the closed-form optimum of a
  two-reaction test network (0.5) is matched by the solver exactly.
- **Prospective sizing.** Analytic minima of 632, 542 and 454 records at assumed
  correlations of 0.3, 0.4 and 0.5, each below the manuscript's pre-specified targets
  of 690, 600 and 500, with power between 0.83 and 0.84 at those targets and between
  0.86 and 0.95 at the low end of the declared 750-900 accrual range.
- **Simulation grid.** Nine cells over twenty seeds. Mean conflict mass rises with the
  declared conflict probability, absent evidence widens the intervals, and the
  interval-width separation of externally labelled concordant from discordant
  profiles is above 0.6 at low conflict and absent evidence.
- **Falsification inequality.** Reported as FAIL at the declared cutoff, because with
  2,000 candidates at 15% prevalence the top-ten recall is bounded by 10/300 and the
  declared 15-point margin cannot hold at any ranking; the same margin is also
  reported at the first cutoff where it becomes arithmetically reachable.
- **Ablations.** Removing the knowledge-graph agent or any single modality lowers
  recall; replacing the combination rule with Dempster renormalisation collapses the
  mean interval width from about 0.33 to about 0.006 while the total conflict mass
  rises above 0.96, which is the behaviour Proposition 1 and the Zadeh argument
  predict.
- **Benchmark.** All 39 declared rows are bound to scoring operators and executed on
  the substrate pool. Every cell of Table 1 and Table 2 stays `[pending]`, because
  those values require the cohort arms.

## Compute budget

The executed studies are CPU-only and need no accelerator. The full simulation grid
of `main` is 180 runs of 2,000 candidates each, about 360,000 candidate evaluations
through four agents, four discounts and three combination steps; the benchmark
harness scores 39 operators over the substrate pool. Memory stays under 2 GB for the
default configuration. The 5,000-resample seed-level bootstrap is the dominant cost
after the grid itself. Setting `--seeds`, `--candidates` or `--bootstrap` reduces the
grid for a quick pass; `configs/experiment/_smoke.yaml` is marked for unit-test use
only and must not be used for reporting.

## Verification

```
python -m crb_concordance.cli.verify
```

Two passes are recorded in `verification_report.json`. The first maps every stated
artefact of the article to the module that carries it and confirms the symbol is
present on the tree (`claim_to_code.json`). The second executes the pipeline and its
independent checks: the mass algebra is compared against explicit focal-set
enumeration, the flux model against a hand-computed optimum, the sizing calculation
against closed-form arithmetic, the survival and marginal models against direct
write-outs, the ranking and calibration metrics against hand-computed values, and the
training path against a forward pass, a loss, a backward pass, a parameter update and
a stored-and-restored payload. The verification logic never calls the routine it is
checking to produce its own expected value.

Every check reports exactly `PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`.
`verification_report.txt` lists the outstanding entries, and
`integrity_manifest.json` carries a SHA-256 digest of every other file in the tree.

## Scope

The four evidence agents are deterministic producers over local resources rather
than hosted-model clients, and the only fitted component is the calibrated
evidence-to-mass map. The substrate and the metabolic reconstruction are
schema-compatible stand-ins for resources that are not redistributable, carrying the
declared structure so that retrieval, leakage control, flux feasibility and the
fusion machinery are all executable. Every place this release departs from, or
supplies a value the manuscript does not state, is listed in `claim_to_code.json`
under `deviations`, and everything that could not be produced is listed under
`not_reproduced`.

## Licence and notices

Apache-2.0; see `LICENSE`. Third-party resources keep their own terms and are
recorded in `NOTICE` and `dataset_urls.txt`.
