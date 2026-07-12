"""
Pydantic schemas for every structured output in the pipeline.
"""
from pydantic import BaseModel, Field


class VideoBrief(BaseModel):
    """Structured scene understanding produced by the vision model."""

    setting: str = Field(description="Where and when the video takes place.")
    subjects: str = Field(description="People, animals, or main entities visible.")
    actions: str = Field(description="What the subjects are doing.")
    objects: str = Field(description="Notable objects, props, or environmental details.")
    mood: str = Field(description="Atmosphere or emotional tone of the clip.")
    sounds: str = Field(description="Notable non-speech sounds if any, otherwise 'none'.")
    dialogue_summary: str = Field(
        description="Summary of spoken dialogue or narration, if any."
    )
    notable_details: str = Field(
        description="Any other distinctive details worth mentioning."
    )
    overall_summary: str = Field(
        description="2-3 sentence summary of the video's content."
    )

    def to_text(self) -> str:
        """Flatten the brief into a readable block for text-only caption models."""
        return (
            f"Setting: {self.setting}\n"
            f"Subjects: {self.subjects}\n"
            f"Actions: {self.actions}\n"
            f"Objects: {self.objects}\n"
            f"Mood: {self.mood}\n"
            f"Sounds: {self.sounds}\n"
            f"Dialogue summary: {self.dialogue_summary}\n"
            f"Notable details: {self.notable_details}\n"
            f"Overall summary: {self.overall_summary}"
        )


class StyledCaptions(BaseModel):
    """Final captions in each required style."""

    formal: str = Field(description="Professional, objective, factual tone.")
    sarcastic: str = Field(description="Dry, ironic, lightly mocking tone.")
    humorous_tech: str = Field(
        description="Funny caption with technology or programming references."
    )
    humorous_non_tech: str = Field(
        description="Funny, everyday humour with no technical jargon."
    )


class CandidateScore(BaseModel):
    """Judge score for a single caption candidate."""

    accuracy: float = Field(
        ge=0.0, le=1.0, description="How faithfully the caption reflects the video."
    )
    style_match: float = Field(
        ge=0.0, le=1.0, description="How well the caption matches the requested tone."
    )
    feedback: str = Field(
        description="Brief, specific reason for the scores and how to improve."
    )


class StyleCandidateScores(BaseModel):
    """Judge output for the candidate set of one style."""

    candidate_0: CandidateScore
    candidate_1: CandidateScore


class CaptionResult(BaseModel):
    """One entry in results.json."""

    task_id: str
    captions: StyledCaptions
