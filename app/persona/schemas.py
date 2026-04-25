"""Validated JSON bundles for persona extraction (Step B)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ContrastPromptPair(BaseModel):
    """One positive / negative system-prompt pair (paper §2.1 contrastive prompts)."""

    positive: str = Field(
        ...,
        min_length=40,
        max_length=8000,
        description="System prompt eliciting the target persona.",
    )
    negative: str = Field(
        ...,
        min_length=40,
        max_length=8000,
        description="Contrast system prompt (e.g. propriety baseline).",
    )


class JudgeCriterion(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    description: str = Field(..., min_length=10, max_length=2000)
    scale_min: int = Field(default=1, ge=1, le=10)
    scale_max: int = Field(default=5, ge=2, le=10)


class JudgeRubric(BaseModel):
    """Instructions for a later LLM judge scoring Gemma rollouts."""

    task_summary: str = Field(..., min_length=20, max_length=4000)
    criteria: list[JudgeCriterion] = Field(..., min_length=3, max_length=12)
    pass_threshold_notes: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="How to turn criterion scores into accept/reject for persona-aligned responses.",
    )


class JudgeJsonScore(BaseModel):
    """Vertex judge response for one transcript (plan §2.2)."""

    score: int = Field(..., ge=0, le=100)
    short_reason: str = Field(..., min_length=1, max_length=800)


class PersonaTraitArtifact(BaseModel):
    """Single-trait bundle: contrast prompts, Q lists, and judge rubric."""

    schema_version: str = Field(default="1", pattern=r"^1$")
    trait_label: str = Field(..., min_length=3, max_length=500)
    trait_description: str = Field(..., min_length=10, max_length=4000)

    pos_system_prompt: str = Field(
        ...,
        min_length=40,
        max_length=8000,
        description="Primary positive system prompt (= first contrast pair).",
    )
    neg_system_prompt: str = Field(
        ...,
        min_length=40,
        max_length=8000,
        description="Primary negative system prompt (= first contrast pair).",
    )

    contrastive_system_prompts: list[ContrastPromptPair] | None = Field(
        default=None,
        description="Paper §2.1: 1 (pilot) or 5 (full) {positive,negative} pairs. "
        "If omitted, derived from pos_system_prompt / neg_system_prompt.",
    )

    contrast_scenarios: list[str] = Field(
        ...,
        min_length=4,
        max_length=20,
        description="Short user messages where pos vs neg behavior should differ clearly.",
    )
    extraction_questions: list[str] = Field(
        ...,
        min_length=8,
        max_length=40,
        description="Extraction split; rolled out under each pos/neg system (per pair). Paper: 20.",
    )
    eval_questions: list[str] = Field(
        ...,
        min_length=4,
        max_length=40,
        description="Eval / held-out questions (pos vs neg via eval-answers, first pair). Paper: 20.",
    )
    judge_rubric: JudgeRubric

    @model_validator(mode="after")
    def _sync_contrast_pairs(self) -> PersonaTraitArtifact:
        if self.contrastive_system_prompts is None or len(self.contrastive_system_prompts) == 0:
            self.contrastive_system_prompts = [
                ContrastPromptPair(
                    positive=self.pos_system_prompt,
                    negative=self.neg_system_prompt,
                )
            ]
        else:
            self.pos_system_prompt = self.contrastive_system_prompts[0].positive
            self.neg_system_prompt = self.contrastive_system_prompts[0].negative
        return self

    def contrast_pair_count(self) -> int:
        return len(self.contrastive_system_prompts or ())
