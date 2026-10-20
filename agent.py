"""Agentic loop: evaluate -> critique -> refine a prompt until it scores well."""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    clarity: int = Field(description="1-10: is the instruction unambiguous?")
    specificity: int = Field(description="1-10: are details, scope and goals concrete?")
    context: int = Field(description="1-10: does it give enough background / role / audience?")
    constraints: int = Field(description="1-10: are limits, tone, length, rules stated?")
    output_format: int = Field(description="1-10: is the desired output structure defined?")
    strengths: List[str] = Field(description="What the prompt already does well")
    weaknesses: List[str] = Field(description="Concrete problems in the prompt")
    suggestions: List[str] = Field(description="Specific, actionable fixes")

    @property
    def overall(self) -> float:
        scores = [self.clarity, self.specificity, self.context,
                  self.constraints, self.output_format]
        return round(sum(scores) / len(scores), 2)


@dataclass
class Iteration:
    number: int
    prompt: str
    evaluation: Evaluation

    @property
    def score(self) -> float:
        return self.evaluation.overall


@dataclass
class OptimizationResult:
    iterations: List[Iteration] = field(default_factory=list)
    final_prompt: str = ""
    stopped_reason: str = ""


_eval_parser = PydanticOutputParser(pydantic_object=Evaluation)

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict prompt-engineering critic. Score the user's prompt "
     "honestly on each dimension from 1 to 10 (most rough prompts deserve 3-6). "
     "Be specific in weaknesses and suggestions.\n\n{format_instructions}"),
    ("human",
     "Original intent of the user:\n{intent}\n\n"
     "Prompt to evaluate:\n\"\"\"\n{prompt}\n\"\"\""),
]).partial(format_instructions=_eval_parser.get_format_instructions())

REFINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert prompt engineer. Rewrite the prompt so it fixes every "
     "weakness in the critique while preserving the user's original intent. "
     "Add a role, context, explicit constraints and an output format where "
     "useful. Do NOT answer the prompt. Do NOT invent facts the user did not "
     "provide; use [placeholders] if details are missing. "
     "Return ONLY the improved prompt text, with no preamble or quotes."),
    ("human",
     "Original intent:\n{intent}\n\n"
     "Current prompt:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
     "Critique:\nWeaknesses:\n{weaknesses}\n\nSuggestions:\n{suggestions}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([("human", "{prompt}")])


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {i}" for i in items) or "- (none)"


class PromptOptimizer:
    def __init__(self, llm, target_score: float = 8.5, max_iterations: int = 4):
        self.llm = llm
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.evaluator = EVALUATOR_PROMPT | llm | _eval_parser
        self.refiner = REFINER_PROMPT | llm | StrOutputParser()
        self.answerer = ANSWER_PROMPT | llm | StrOutputParser()

    def _evaluate(self, intent: str, prompt: str) -> Evaluation:
        last_err: Optional[Exception] = None
        for _ in range(3):  # LLMs occasionally emit malformed JSON; retry
            try:
                return self.evaluator.invoke({"intent": intent, "prompt": prompt})
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(f"Evaluator failed to return valid JSON: {last_err}")

    def run(
        self,
        raw_prompt: str,
        on_step: Optional[Callable[[str, Optional[Iteration]], None]] = None,
    ) -> OptimizationResult:
        notify = on_step or (lambda *_: None)
        result = OptimizationResult()
        current = raw_prompt.strip()

        for n in range(1, self.max_iterations + 1):
            notify(f"Evaluating prompt (iteration {n})...", None)
            evaluation = self._evaluate(raw_prompt, current)
            it = Iteration(n, current, evaluation)
            result.iterations.append(it)
            notify(f"Iteration {n} scored {it.score}/10", it)

            if it.score >= self.target_score:
                result.stopped_reason = f"Target score {self.target_score} reached."
                break
            if n == self.max_iterations:
                result.stopped_reason = "Max iterations reached."
                break

            notify(f"Refining prompt (iteration {n})...", None)
            current = self.refiner.invoke({
                "intent": raw_prompt,
                "prompt": current,
                "weaknesses": _bullets(evaluation.weaknesses),
                "suggestions": _bullets(evaluation.suggestions),
            }).strip()

        # Return the best-scoring version, not necessarily the last one.
        best = max(result.iterations, key=lambda i: i.score)
        result.final_prompt = best.prompt
        return result

    def answer(self, prompt: str) -> str:
        return self.answerer.invoke({"prompt": prompt})
