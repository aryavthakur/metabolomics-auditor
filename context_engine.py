# context_engine.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional


StudyGoal = Literal["exploratory", "confirmatory"]
MetabolomicsType = Literal["untargeted", "targeted"]
DesignType = Literal["independent", "paired", "longitudinal"]
GroupCount = Literal["two_groups", "multi_group"]


@dataclass(frozen=True)
class Context:
    metabolomics_type: MetabolomicsType = "untargeted"
    study_goal: StudyGoal = "exploratory"
    design_type: DesignType = "independent"
    group_count: GroupCount = "two_groups"
    has_batches: bool = False
    small_n: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_context() -> Context:
    return Context()


def context_expectations(ctx: Context) -> Dict[str, Any]:
    """
    Returns what *should* be present / emphasized given context.
    This does NOT look at the dataframe; it's purely "what should the reviewer expect".
    """
    expect_fdr = ctx.metabolomics_type == "untargeted" and ctx.study_goal in ("exploratory", "confirmatory")
    # reviewers expect correction even more when confirmatory, but untargeted alone is enough
    expect_effect_size = True  # generally yes for any inferential claims
    expect_p_values = ctx.study_goal == "confirmatory" or ctx.metabolomics_type == "untargeted"

    # paired / longitudinal designs: reviewers expect explicit modeling choices
    expect_paired_awareness = ctx.design_type in ("paired", "longitudinal")

    # batch effects: expect batch info or correction note
    expect_batch_handling = ctx.has_batches

    # small n: prefer nonparametric / effect sizes / confidence intervals; warn on overclaiming
    small_n_risk = ctx.small_n

    return {
        "expect_p_values": expect_p_values,
        "expect_fdr": expect_fdr,
        "expect_effect_size": expect_effect_size,
        "expect_paired_awareness": expect_paired_awareness,
        "expect_batch_handling": expect_batch_handling,
        "small_n_risk": small_n_risk,
    }


def context_narrative(ctx: Context) -> List[str]:
    """
    Human-readable bullets for the report.
    """
    bullets: List[str] = []
    bullets.append(f"Metabolomics type: **{ctx.metabolomics_type}**")
    bullets.append(f"Study goal: **{ctx.study_goal}**")
    bullets.append(f"Design: **{ctx.design_type}**")
    bullets.append(f"Group structure: **{ctx.group_count}**")
    bullets.append(f"Batch effects likely: **{ctx.has_batches}**")
    bullets.append(f"Small sample size: **{ctx.small_n}**")
    return bullets
