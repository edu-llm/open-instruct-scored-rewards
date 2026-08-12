"""Online multi-head reward for scenario-conditioned tutor RL."""

from __future__ import annotations

from open_instruct.scored_rewards import Sample, ScoreResult, register
from open_instruct.scored_rewards.types import TRANSCRIPT_KEY, parse_transcript
from projects.pedagogy_rm.plugin import PedagogyHead
from projects.tutor_metrics.metrics import CONTRIBUTIONS, PENALTY_WEIGHTS, probe_reward

REWARD_KEYS = (
    "assistance_level",
    *(metric.key for metric in CONTRIBUTIONS),
    "locates_student_object",
    "verdict_with_referent",
    "demand_is_specific",
    *PENALTY_WEIGHTS,
)


class TutorMetricsHead(PedagogyHead):
    """One frozen encoder, with a separate tiny Ridge head per metric."""

    name = "tutor_metrics"

    def _default_dimensions(self) -> list[str]:
        return [
            key
            for key in REWARD_KEYS
            if key in self.meta["dimensions"] and self.meta["dimensions"][key].get("deployable", True)
        ]

    def context(self, sample: Sample) -> tuple[list[dict], str]:
        from projects.tutor_metrics.generate import tutor_messages, tutor_messages_neutral  # noqa: PLC0415

        item = sample.item
        transcript = parse_transcript(sample.env_info.get(TRANSCRIPT_KEY, "")) if sample.env_info else []
        tutor_turns = [turn.get("text", "") for turn in transcript if turn.get("who") in ("policy", "tutor", "assistant")]
        turn = tutor_turns[-1] if tutor_turns else sample.policy_text
        student_turns = [turn.get("text", "") for turn in transcript if turn.get("who") in ("partner", "student", "user")]
        student = student_turns[-1] if student_turns else item.get("student_before", "")
        question = item.get("question") or item.get("prompt") or sample.prompt
        problem = {"question": question, "choices": item.get("choices")}
        scheme = self.meta.get("prompt_scheme")
        if scheme == "tutor_metrics":
            style = item.get("style")
            if not style:
                raise ValueError("a tutor_metrics head requires style in every dataset row")
            return tutor_messages(problem, student, style), turn
        if scheme == "tutor_metrics_neutral":
            return tutor_messages_neutral(problem, student), turn
        raise ValueError(f"unsupported tutor-metrics prompt scheme {scheme!r}")

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        import numpy as np  # noqa: PLC0415

        contexts = [self.context(sample) for sample in group]
        pooled = self.states(contexts)
        predicted: dict[str, list[float]] = {}
        for dimension in self.dims:
            spec = self.meta["dimensions"][dimension]
            weights = self.weights[dimension]
            states = np.stack(pooled[(spec["pooling"], spec["layer"])]).astype(np.float64)
            scaled = (states - weights["mean"]) / weights["scale"]
            raw = scaled @ weights["coef"] + weights["intercept"]
            predicted[dimension] = [float(np.clip(value, spec["lo"], spec["hi"])) for value in raw]

        results = []
        for index, (sample, (_, turn)) in enumerate(zip(group, contexts, strict=True)):
            labels = {dimension: predicted[dimension][index] for dimension in self.dims}
            target = sample.item.get("target_level")
            if not target:
                raise ValueError("every tutor-metrics RL row requires target_level in ground_truth")
            length = self.length_fit(len(turn.split())) * self.length_weight if self.length_weight else 0.0
            score, terms = probe_reward(labels, str(target), length_term=length)
            unweighted = score
            score *= self.reward_weight
            results.append(
                ScoreResult(
                    score=score,
                    dimensions=terms,
                    info={
                        "raw": labels,
                        "target_level": target,
                        "words": len(turn.split()),
                        "unweighted_score": unweighted,
                    },
                )
            )
        return results


register("tutor_metrics", TutorMetricsHead)
