"""Verification driver: the claim mapping, the execution pass and the artefacts.

Ref: Sec. 4.1 to Sec. 4.6 and Supplementary Table S1 (the two design principles, the
combine rule, the four evidence agents, the temporal-holdout benchmark, the
simulation study and the statistical analysis plan).

Two passes are recorded. The first maps every stated artefact of the article to the
module that carries it and confirms the symbol is present on this tree. The second
executes the pipeline and its independent checks. Both write their verdicts to the
release root together with a content manifest of the final tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from crb_concordance.evaluation.audit import environment_block, run_all, status_counts
from crb_concordance.evaluation.reporting import render_text
from crb_concordance.utils.atomic import (
    atomic_write_json,
    atomic_write_text,
    iter_tree_files,
    sha256_bytes,
    sha256_file,
)
from crb_concordance.utils.logging import configure_logging, get_logger
from crb_concordance.utils.types import ClaimRecord, DeviationRecord, Verdict

LOGGER = get_logger("cli.verify")

MANIFEST_NAME = "integrity_manifest.json"
REPORT_NAME = "verification_report.json"
SUMMARY_NAME = "verification_report.txt"
CLAIM_NAME = "claim_to_code.json"
PAPER_TITLE = (
    "Knowledge-Graph-Augmented Multi-Agent Systems for Discovering Metabolic "
    "Vulnerabilities in Radioresistant Colorectal Cancer"
)


@dataclass(frozen=True, slots=True)
class ClaimSpec:
    """One stated artefact with the code that carries it."""

    claim_id: str
    paper_location: str
    statement: str
    code_paths: tuple[str, ...]
    symbols: tuple[str, ...] = ()

    def as_record(self, root: Path) -> ClaimRecord:
        missing: list[str] = []
        combined = ""
        for relative in self.code_paths:
            target = root / relative
            if not target.exists():
                missing.append(f"{relative}:absent")
                continue
            combined += target.read_text(encoding="utf-8", errors="replace")
        for symbol in self.symbols:
            if symbol not in combined:
                missing.append(f"{symbol}")
        verdict = Verdict.PASS if not missing else Verdict.FAIL
        statement = self.statement
        if missing:
            statement = f"{statement} [unresolved: {', '.join(missing)}]"
        return {
            "claim_id": self.claim_id,
            "paper_location": self.paper_location,
            "statement": statement,
            "code_paths": list(self.code_paths),
            "verification": verdict.value,
        }


CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        "C1",
        "Sec. 4.1, Eq. (1)",
        "Every modality reports a mass function on Theta_g = {V, not V} with "
        "m(V) + m(not V) + m(Theta_g) = 1",
        ("src/crb_concordance/belief/mass.py",),
        ("class MassTriple", "def __post_init__"),
    ),
    ClaimSpec(
        "C2",
        "Sec. 4.1, Eq. (2)",
        "Shafer discounting scales the committed mass by alpha_k and sends the shortfall to Theta_g",
        ("src/crb_concordance/belief/discount.py",),
        ("def discount", "def undiscount", "class DiscountRates"),
    ),
    ClaimSpec(
        "C3",
        "Sec. 4.2, Algorithm 2; Eq. (3)",
        "The sequential combination redistributes the step conflict K_step into m(Theta_g) and never "
        "renormalises it away",
        ("src/crb_concordance/belief/combine.py",),
        ("def combine_pair", "def combine_sequential", "def dempster_pair"),
    ),
    ClaimSpec(
        "C4",
        "Sec. 4.1, Eq. (4)",
        "Bel(V) = m(V), Pl(V) = m(V) + m(Theta_g) and BetP(V) = m(V) + m(Theta_g)/2",
        ("src/crb_concordance/belief/mass.py",),
        ("def belief", "def plausibility", "def pignistic", "def interval_width"),
    ),
    ClaimSpec(
        "C5",
        "Sec. 4.2, Algorithm 1",
        "The discovery pass collects the four evidences, discounts, combines, appends the ledger "
        "rows and returns the pool sorted by BetP(V) descending",
        ("src/crb_concordance/discovery/pipeline.py",),
        ("def run_discovery", "def collect_evidence", "def build_agents"),
    ),
    ClaimSpec(
        "C6",
        "Sec. 4.2, Algorithm 3",
        "Discovery-time scoring uses the fixed calibrated mapping and rates and updates no parameter",
        ("src/crb_concordance/agents/verifier.py", "src/crb_concordance/agents/mapping.py"),
        ("def combine_candidate", "class MassMapper"),
    ),
    ClaimSpec(
        "C7",
        "Sec. 4.1, Proposition 1",
        "The belief-plausibility width increases with the accumulated conflict while the "
        "pre-aggregation ignorance masses stay fixed",
        ("src/crb_concordance/belief/guarantees.py",),
        ("def conflict_monotonicity_scan", "def conflict_monotonicity_check"),
    ),
    ClaimSpec(
        "C8",
        "Sec. 4.1, Proposition 2",
        "Under conditional independence the pignistic estimate converges up to the fixed-discount "
        "floor half of the product of the knowledge gaps",
        ("src/crb_concordance/belief/guarantees.py",),
        ("def pignistic_bias_floor", "def zero_conflict_limit", "def pignistic_convergence_scan"),
    ),
    ClaimSpec(
        "C9",
        "Sec. 4.1, fitting of {alpha_k}",
        "The discount rates are fitted on the retrospective control panel built from GLUT1/SLC2A1, "
        "MCT1/SLC16A1, SLC16A9, CPT1A and CES1 and are never recalibrated during discovery",
        (
            "src/crb_concordance/calibration/discount_fit.py",
            "src/crb_concordance/cohorts/control_panel.py",
        ),
        ("def calibrate", "def fit_rate_closed_form", "def build_control_panel"),
    ),
    ClaimSpec(
        "C10",
        "Sec. 4.1, four-fold calibration panel",
        "Each fold uses the calibration controls and withholds one of them, so the withheld "
        "control's recall is computed from rates calibrated without it",
        ("src/crb_concordance/calibration/folds.py",),
        ("def build_folds", "def withheld_positives"),
    ),
    ClaimSpec(
        "C11",
        "Sec. 4.2, provenance ledger",
        "Every agent call is recorded with the agent, candidate, evidence nature, score, mass "
        "triple, discount rate, calibration fold, source, version, query and timestamp",
        ("src/crb_concordance/discovery/ledger.py", "src/crb_concordance/metrics/provenance.py"),
        ("class ProvenanceLedger", "REQUIRED_LEDGER_FIELDS", "def provenance_grounding_rate"),
    ),
    ClaimSpec(
        "C12",
        "Sec. 4.2 item (1); Sec. 4.4",
        "The knowledge-graph agent retrieves multi-hop routes to the phenotype and converts "
        "pathway-level support into m_g,KG under a bounded path length",
        ("src/crb_concordance/agents/kg_path.py", "src/crb_concordance/graph/paths.py"),
        ("class KnowledgeGraphAgent", "def enumerate_paths", "def path_support"),
    ),
    ClaimSpec(
        "C13",
        "Sec. 4.4, leakage controls",
        "The knowledge-cutoff control, the substrate-circularity control and the edge-holding "
        "scenario are applied to the retrieval agent",
        ("src/crb_concordance/graph/leakage.py",),
        ("class KnowledgeCutoffControl", "def edge_holdout", "def substrate_circularity_control"),
    ),
    ClaimSpec(
        "C14",
        "Sec. 4.3, Human-GEM v2.0.1",
        "Metabolic reaction edges are merged into the substrate so that they are structurally new "
        "relative to any pretraining graph",
        ("src/crb_concordance/graph/metabolic_edges.py",),
        ("def attach_reaction_edges", "HUMAN_GEM_SOURCE"),
    ),
    ClaimSpec(
        "C15",
        "Sec. 4.2 item (3)",
        "The flux-feasibility agent derives a context-specific model and measures the feasibility "
        "of alternative pathways under suppression of the candidate",
        (
            "src/crb_concordance/agents/flux_feasibility.py",
            "src/crb_concordance/metabolism/knockout.py",
        ),
        ("class FluxFeasibilityAgent", "def assess_gene_suppression", "def pathway_capacity"),
    ),
    ClaimSpec(
        "C16",
        "Sec. 4.2 item (4)",
        "The transcriptomic-clinical association agent regresses expression against radiotherapy "
        "response across the public cohorts",
        ("src/crb_concordance/agents/transcriptomic_clinical.py",),
        ("class ClinicalAssociationAgent", "def point_biserial"),
    ),
    ClaimSpec(
        "C17",
        "Sec. 4.2 item (2)",
        "The dependency agent reads functional-genomic loss of fitness together with dose-response "
        "sensitivity",
        ("src/crb_concordance/agents/dependency.py",),
        ("class DependencyAgent", "def dependency_score", "def drug_score"),
    ),
    ClaimSpec(
        "C18",
        "Sec. 4.2 item (5)",
        "The critic/verifier performs the discounting, the combination and the structural checks",
        ("src/crb_concordance/agents/verifier.py",),
        ("class Verifier", "def audit", "def structural_report"),
    ),
    ClaimSpec(
        "C19",
        "Table 1",
        "Thirty-seven agent, knowledge-graph and mechanistic-modelling baselines across six "
        "families plus two fusion-mechanism controls, with the proposed row",
        ("src/crb_concordance/benchmark/registry.py",),
        ("DECLARED_BASELINES = 37", "DECLARED_FUSION_CONTROLS = 2", "def validate_registry"),
    ),
    ClaimSpec(
        "C20",
        "Table 2",
        "The clinical-signature and standard-of-care comparison stays pending until the clinical "
        "arms are analysed",
        ("src/crb_concordance/evaluation/reporting.py",),
        ("TABLE_TWO_METHODS", "def table_two_snapshot", "PENDING"),
    ),
    ClaimSpec(
        "C21",
        "Sec. 4.4, five metrics",
        "Recall@k, precision@k, mean reciprocal rank and the fold-level hit rate recorded per "
        "named control",
        ("src/crb_concordance/metrics/ranking.py",),
        ("def recall_at_k", "def precision_at_k", "def mean_reciprocal_rank", "def hit_rate"),
    ),
    ClaimSpec(
        "C22",
        "Sec. 4.4; Sec. 4.6",
        "Expected calibration error and the Brier score are defined identically at discovery time "
        "and at clinical time",
        ("src/crb_concordance/metrics/calibration.py",),
        ("def expected_calibration_error", "def brier_score", "def calibration_summary"),
    ),
    ClaimSpec(
        "C23",
        "Sec. 4.4, provenance and hallucination",
        "The provenance-grounding rate and the complementary hallucinated-edge rate are audited "
        "against the ledger rather than self-reported",
        ("src/crb_concordance/metrics/provenance.py",),
        ("def audit_citations", "def hallucinated_edge_rate", "def summarise"),
    ),
    ClaimSpec(
        "C24",
        "Sec. 4.6, bootstrap resampling",
        "Confidence intervals use 2,000 clinical resamples and 5,000 seed-level resamples",
        ("src/crb_concordance/metrics/comparison.py",),
        ("CLINICAL_RESAMPLES = 2000", "SEED_RESAMPLES = 5000"),
    ),
    ClaimSpec(
        "C25",
        "Sec. 4.6, DeLong",
        "The difference between two correlated ROC curves on the same cohort is assessed by "
        "DeLong's method",
        ("src/crb_concordance/stats/delong.py",),
        ("def roc_estimate", "def paired_comparison"),
    ),
    ClaimSpec(
        "C26",
        "Sec. 4.6, cross-site consistency",
        "Cochran's Q with the I-squared statistic flags heterogeneity above fifty percent and the "
        "inter-site gap is flagged above ten percentage points",
        ("src/crb_concordance/metrics/comparison.py", "src/crb_concordance/stats/survival.py"),
        ("def cochran_q", "def site_gap_summary"),
    ),
    ClaimSpec(
        "C27",
        "Sec. 4.6, multiplicity",
        "Holm-Bonferroni controls the primary family and Benjamini-Hochberg the exploratory battery",
        ("src/crb_concordance/stats/multiplicity.py",),
        ("def holm_bonferroni", "def benjamini_hochberg", "def family_plan"),
    ),
    ClaimSpec(
        "C28",
        "Sec. 4.6, time-to-event",
        "Cox models are stratified by site with competing-risk handling for local recurrence",
        ("src/crb_concordance/stats/survival.py",),
        ("def fit_cox", "def fit_competing_risks", "def cumulative_incidence"),
    ),
    ClaimSpec(
        "C29",
        "Sec. 4.6, reader study",
        "The reader-study arm gain uses a marginal logistic model with reader as a random effect "
        "and a reader-level cluster bootstrap as a cross-check",
        ("src/crb_concordance/stats/gee.py", "src/crb_concordance/metrics/comparison.py"),
        ("def fit_gee_logistic", "def arm_gain_report", "def cluster_bootstrap_difference"),
    ),
    ClaimSpec(
        "C30",
        "Sec. 4.6, prospective sizing",
        "The prospective arm is sized by a Hanley-McNeil calculation for two correlated AUROC "
        "curves with the sensitivity to the assumed correlation reported explicitly",
        ("src/crb_concordance/stats/power.py",),
        ("def hanley_mcneil_variance", "def analytic_minimum_records", "DECLARED_TARGETS"),
    ),
    ClaimSpec(
        "C31",
        "Supplementary Table S1",
        "Twenty simulation seeds, two thousand candidates per seed, fifteen percent prevalence, "
        "the conflict and absent-evidence grids and 5,000 seed-level resamples",
        ("src/crb_concordance/simulation/design.py",),
        ("SIMULATION_SEEDS", "CONFLICT_GRID", "ABSENT_GRID", "SEED_BOOTSTRAP_RESAMPLES"),
    ),
    ClaimSpec(
        "C32",
        "Sec. 4.5, H1 and H2",
        "H1 removes the combination rule and H2 suppresses the conflict trace, whose interval-width "
        "partial R-squared must collapse",
        ("src/crb_concordance/simulation/contrasts.py",),
        ("def fusion_rule_h1", "def conflict_trace_h2", "def partial_r2"),
    ),
    ClaimSpec(
        "C33",
        "Sec. 4.4, falsification inequality",
        "The pre-specified margin of fifteen percentage points over the best single-modality "
        "baseline is evaluated on seed-level bootstrap intervals",
        (
            "src/crb_concordance/simulation/contrasts.py",
            "src/crb_concordance/simulation/design.py",
        ),
        ("def falsification_inequality", "def falsification_inequality_saturated", "PARITY_MARGIN"),
    ),
    ClaimSpec(
        "C34",
        "Sec. 4.5, ablation plan",
        "Four levels of extraction plus a substitution tier replacing the combination rule with "
        "Dempster renormalisation, naive summation or averaging and Bayesian log-odds",
        ("src/crb_concordance/evaluation/ablations.py",),
        ("class ExtractionLevel", "class SubstitutionRule", "def run_battery"),
    ),
    ClaimSpec(
        "C35",
        "Sec. 4.6; Data availability",
        "Three sites across three regions with the retrospective and prospective accrual targets "
        "and the declared exclusion criteria",
        ("src/crb_concordance/cohorts/schema.py", "src/crb_concordance/cohorts/generator.py"),
        ("class ClinicalRecord", "class CohortDesign", "def generate_cohort"),
    ),
    ClaimSpec(
        "C36",
        "Sec. 4.1, control panel",
        "GLUT1/SLC2A1 and MCT1/SLC16A1 are the positive controls and SLC16A9, CPT1A and CES1 the "
        "negative controls, with MCT1 expected to score high without deserving a high rank",
        ("src/crb_concordance/cohorts/control_panel.py",),
        ("CONTROL_GENES", "RankExpectation.DISCREPANT", "INFORMATIVE_NEGATIVE"),
    ),
    ClaimSpec(
        "C37",
        "Sec. 4.1, frame",
        "The frame is the binary set Theta_g = {V, not V} for every candidate",
        ("src/crb_concordance/belief/mass.py", "src/crb_concordance/utils/types.py"),
        ("class MassTriple", "class Modality"),
    ),
    ClaimSpec(
        "C38",
        "Sec. 4.6, design ranges",
        "The pCR rate sits in the reported twenty to thirty percent range and the standard-of-care "
        "baseline in the reported 0.65 to 0.74 range",
        ("src/crb_concordance/cohorts/generator.py",),
        ("pcr_rate: float = 0.25", "soc_auroc: float = 0.70"),
    ),
    ClaimSpec(
        "C39",
        "Supplementary Table S1, reader panel",
        "The reader study uses at least eight readers on at least one hundred and fifty-five "
        "shared cases",
        ("src/crb_concordance/cohorts/generator.py",),
        ("reader_count: int = 8", "shared_cases: int = 155"),
    ),
    ClaimSpec(
        "C40",
        "Sec. 4.2, calibrated mass map",
        "The evidence-to-mass map is a fitted object that is fixed for discovery time",
        ("src/crb_concordance/calibration/mass_calibration.py",),
        ("class CalibratedMassMap", "def fit_mass_map", "def brier_mass_loss"),
    ),
)

DEVIATIONS: tuple[DeviationRecord, ...] = (
    {
        "paper_location": "Sec. 4.2, system architecture",
        "departure": (
            "The four evidence agents are shipped as deterministic evidence producers over local "
            "resources; no hosted model client is included and the only fitted component is the "
            "calibrated mass map."
        ),
        "justification": (
            "The manuscript's backbones are privately hosted services. Shipping a client would put "
            "deployment code and credentials into a peer-review release, and the discovery-time "
            "behavior the paper specifies is a fixed mapping from evidence to mass with no "
            "parameter update."
        ),
    },
    {
        "paper_location": "Sec. 4.3, auxiliary public layer",
        "departure": (
            "The substrate ships as a schema-compatible construction carrying the declared node "
            "kinds, relation types, release-dated edges and the named pathways; the Human-GEM "
            "v2.0.1 layer is a reduced reconstruction with a reader for the released JSON export."
        ),
        "justification": (
            "The OptimusKG data deposit is CC BY-NC-SA 4.0 and is not redistributed by this "
            "release. The reduced reconstruction keeps every pathway the manuscript names so the "
            "retrieval and leakage controls remain executable."
        ),
    },
    {
        "paper_location": "Sec. 4.3, reconstruction techniques",
        "departure": (
            "The reduced reconstruction's exchange caps, cytosolic and mitochondrial redox split, "
            "shuttle capacity and biomass precursor set are declared engineering defaults."
        ),
        "justification": (
            "The manuscript reports no medium composition or biomass equation. The values are "
            "exposed in configs/data and every model output is labelled as a property of the "
            "reduced model rather than as a paper result."
        ),
    },
    {
        "paper_location": "Sec. 4.1, the four-fold calibration panel",
        "departure": (
            "The folds are read as four leave-one-out folds over the four calibration controls, "
            "with the discrepant positive control SLC16A1 withheld from calibration entirely."
        ),
        "justification": (
            "The manuscript states that every fold uses three of the controls with the whole "
            "negative set, that one positive is never processed, and that the panel is a four-fold "
            "panel. Reading it as four leave-one-out folds over four calibration controls is the "
            "only reading that satisfies all three statements."
        ),
    },
    {
        "paper_location": "Supplementary Table S1",
        "departure": (
            "The conflict-trace flagging threshold is 0.05 and the candidate-gating path length is "
            "three hops, both taken from the table's planning values."
        ),
        "justification": (
            "The table marks both as values to be fixed at implementation and gives these planning "
            "values, so they are exposed as configuration defaults rather than invented constants."
        ),
    },
    {
        "paper_location": "Sec. 4.1, control panel coding",
        "departure": (
            "Control observations are the declared threshold crossings coded as 0.82 and 0.12 with "
            "a declared dispersion of 0.16."
        ),
        "justification": (
            "The screen reports whether a control crossed each threshold rather than the underlying "
            "score. Without a declared dispersion the four controls separate perfectly and every "
            "fitted rate saturates at one, which the release reports as a panel property."
        ),
    },
    {
        "paper_location": "Sec. 4.4, pre-specified falsification inequality",
        "departure": (
            "The fifteen percentage point margin is reported at the declared recall@10 cutoff and "
            "again at the first cutoff where the margin is arithmetically reachable."
        ),
        "justification": (
            "With 2,000 candidates at fifteen percent prevalence the top-ten recall is bounded by "
            "10/300, so the declared inequality cannot hold at the declared cutoff whatever the "
            "ranking does. The second measurement makes the claim testable instead of vacuous."
        ),
    },
    {
        "paper_location": "Table 1",
        "departure": (
            "Each table row is bound to a declared scoring operator over the same four modalities."
        ),
        "justification": (
            "The manuscript specifies the rows, families and counts but not their implementations, "
            "and states that each row is reimplemented on this paper's own pool. The operators are "
            "distinct pipelines, not renamings, and they are declared in the benchmark report."
        ),
    },
    {
        "paper_location": "Abstract",
        "departure": (
            "The separability figure of 4.76 and the conflict mass of 0.339 are not reproduced; the "
            "simulation reports its own statistics instead."
        ),
        "justification": (
            "The manuscript states neither estimator, neither unit, nor a confidence-interval "
            "procedure for these two quantities, and the section that would define them is absent "
            "from the supplied document, so any number claimed for them would be invented."
        ),
    },
    {
        "paper_location": "Sec. 4.6 and Data availability",
        "departure": (
            "All cohort-level quantities, the in vitro corroboration and the clinical tables remain "
            "NOT_RUN."
        ),
        "justification": (
            "The retrospective and prospective arms are held at their institutions and are not "
            "redistributed. The analysis plan for them is implemented and exercised on synthetic "
            "records, which is reported as such."
        ),
    },
)

FINDINGS: tuple[dict[str, str], ...] = (
    {
        "check": "declared_margin_reachability",
        "outcome": (
            "With the declared pool of 2,000 candidates at 15% prevalence the top-ten recall is "
            "bounded by 10/300 = 0.033, so the pre-specified 15-percentage-point margin cannot be "
            "met at the declared cutoff by any ranking."
        ),
        "implication": (
            "The inequality is reported at the declared cutoff and again at the first cutoff where "
            "it is arithmetically reachable, and the reachability bound is verified separately."
        ),
    },
    {
        "check": "falsification_inequality_at_declared_cutoff",
        "outcome": (
            "The pre-specified margin is not met at the declared cutoff: measured over the reduced "
            "grid the mean parity gap is about 0.014 with an interval whose lower bound is about "
            "0.002. At the first cutoff where the margin becomes reachable the gap stays close to "
            "zero, so the concordance ranking matches the best single-modality baseline rather than "
            "beating it."
        ),
        "implication": (
            "The manuscript expects parity with the best unconstrained baseline rather than a "
            "discovery-utility advantage, so this measured outcome agrees with its own expectation. "
            "It is why this release claims the interval, provenance and conflict axes, and why the "
            "inequality is reported as measured instead of being reconciled."
        ),
    },
    {
        "check": "ablation_dempster_collapses_interval",
        "outcome": (
            "Replacing the combination rule with Dempster renormalisation collapses the mean "
            "interval width by roughly two orders of magnitude while the total conflict mass rises."
        ),
        "implication": (
            "This is the behaviour the Zadeh argument in the introduction predicts, and it is the "
            "measured basis for keeping the conflict-redistributing rule as the default."
        ),
    },
    {
        "check": "discount_rates_in_range",
        "outcome": (
            "On the coded control panel some fitted discount rates saturate at one, because the "
            "four calibration controls are nearly separable once the crossings are coded."
        ),
        "implication": (
            "Proposition 2 explicitly allows the ignorance floor to vanish when a rate reaches one, "
            "so the floor is verified both on the deployed rates and on a grid of rates below one."
        ),
    },
)

NOT_REPRODUCED: tuple[dict[str, str], ...] = (
    {
        "item": "Table 1 and Table 2 outcome cells",
        "reason": "the values require the cohort arms, which are not redistributed",
    },
    {
        "item": "retrospective and prospective cohort analyses",
        "reason": "private multi-site cohorts held under institutional agreements",
    },
    {
        "item": "in vitro cell-line corroboration",
        "reason": "laboratory work outside the scope of a code release",
    },
    {
        "item": "the five pretreatment-biopsy expression analyses",
        "reason": "the series are public but are not downloaded into this release",
    },
    {
        "item": "abstract separability and conflict-mass figures",
        "reason": "no estimator is stated in the manuscript for either quantity",
    },
)


@dataclass(slots=True)
class ArtefactContext:
    """The paths a verification run writes to."""

    root: Path
    workspace: Path
    claims: tuple[ClaimSpec, ...] = CLAIMS
    extra: dict[str, object] = field(default_factory=dict)


def resolve_root(config_root: str | Path) -> Path:
    """The release root that holds the experiment files.

    Accepts the configuration directory, an experiment file, or any path inside the
    release, and returns the directory that contains ``configs``.
    """

    candidate = Path(config_root)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if candidate.is_dir():
        return candidate.parent if candidate.name == "configs" else candidate
    if candidate.suffix in (".yaml", ".yml"):
        return candidate.parents[2]
    return candidate.parent


def claim_records(root: Path, claims: tuple[ClaimSpec, ...]) -> list[ClaimRecord]:
    return [claim.as_record(root) for claim in claims]


def build_claim_payload(
    root: Path, claims: tuple[ClaimSpec, ...], hygiene: dict[str, object]
) -> dict[str, object]:
    records = claim_records(root, claims)
    failed = [
        record["claim_id"] for record in records if record["verification"] != Verdict.PASS.value
    ]
    overall = Verdict.PASS.value if not failed else "PARTIALLY_VERIFIED"
    return {
        "overall": overall,
        "paper": PAPER_TITLE,
        "scope_note": (
            "Every entry maps a stated artefact of the article to the code that carries it and "
            "confirms the symbol is present on this tree. The second verification pass executes the "
            "pipeline itself; its checks are listed in verification_report.json."
        ),
        "claims": records,
        "unresolved_claims": failed,
        "deviations": [dict(record) for record in DEVIATIONS],
        "findings": [dict(record) for record in FINDINGS],
        "not_reproduced": [dict(record) for record in NOT_REPRODUCED],
        "hygiene": hygiene,
    }


def write_integrity_manifest(root: Path, *, exclude: str = MANIFEST_NAME) -> dict[str, object]:
    """Manifest every file of the final tree except the manifest itself."""

    files: list[dict[str, object]] = []
    for path in iter_tree_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == exclude:
            continue
        file_digest = sha256_file(path)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_digest})
    payload = {"algorithm": "SHA-256", "files": files}
    blob = "\n".join(f"{entry['path']}:{entry['sha256']}" for entry in files).encode("utf-8")
    payload["manifest_digest"] = sha256_bytes(blob)
    payload["file_count"] = len(files)
    atomic_write_json(root / exclude, payload)
    return payload


def verify(
    *,
    config_root: str | Path,
    root: str | Path | None = None,
    workspace: str | Path | None = None,
    claims: tuple[ClaimSpec, ...] = CLAIMS,
) -> dict[str, object]:
    """Run both verification passes and write every root artefact."""

    resolved = Path(root).resolve() if root is not None else resolve_root(config_root)
    room = Path(workspace).resolve() if workspace is not None else resolved / ".verify"
    room.mkdir(parents=True, exist_ok=True)
    results, hygiene = run_all(resolved, room)
    counts = status_counts(results)
    claim_payload = build_claim_payload(resolved, claims, hygiene)
    claim_verdicts = {
        record["claim_id"]: record["verification"] for record in claim_payload["claims"]  # type: ignore[union-attr]
    }
    outstanding = [
        result.name
        for result in results
        if result.status in (Verdict.FAIL, Verdict.NOT_RUN, Verdict.BLOCKED)
    ]
    if counts[Verdict.FAIL.value] > 0 or claim_payload["unresolved_claims"]:
        overall = "PARTIALLY_VERIFIED"
    else:
        overall = "PASS"
    report: dict[str, object] = {
        "overall": overall,
        "scope": (
            "Executed checks cover the belief-function core, the metabolic feasibility model, the "
            "graph retrieval and leakage controls, the calibration fold and fitting path, the "
            "cohort design marginals, the discovery pass and its ledger, the simulation study, the "
            "ablation battery, the transcribed benchmark table and the statistics plan."
        ),
        "environment": environment_block(),
        "status_counts": counts,
        "checks": [result.as_dict() for result in results],
        "outstanding": outstanding,
        "claims": {"total": len(claims), "verdicts": claim_verdicts},
        "not_run": [dict(record) for record in NOT_REPRODUCED],
        "hygiene": hygiene,
        "artefacts": {
            "claim_to_code": CLAIM_NAME,
            "report": REPORT_NAME,
            "summary": SUMMARY_NAME,
            "manifest": MANIFEST_NAME,
        },
    }
    atomic_write_json(resolved / CLAIM_NAME, claim_payload)
    atomic_write_json(resolved / REPORT_NAME, report)
    summary = render_text(
        "Verification summary",
        {
            "overall": overall,
            "status_counts": counts,
            "unresolved_claims": claim_payload["unresolved_claims"],
            "outstanding_checks": outstanding,
            "hygiene": hygiene,
        },
    )
    atomic_write_text(resolved / SUMMARY_NAME, summary)
    write_integrity_manifest(resolved)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run the two verification passes and write the artefacts")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--root", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    report = verify(config_root=args.config_root, root=args.root, workspace=args.workspace)
    LOGGER.info(
        "verification overall %s with counts %s", report["overall"], report["status_counts"]
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
