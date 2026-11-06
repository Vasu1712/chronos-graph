"""LongMemEval's released answer-checking prompts, vendored verbatim.

Source: https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py
(function get_anscheck_prompt, commit 9e0b455f, fetched 2026-11-06). Each template takes
(question, correct answer | rubric | explanation, model response) in that order.

"default" is used for single-session-user, single-session-assistant and multi-session, and, in this
repository, for every answerable LoCoMo question. "abstention" is the _abs prompt; ChronosGraph's
evaluation scores abstention questions from the abstained flag instead (eval/judge.py) and keeps the
template only for completeness.
"""

import re

TEMPLATES = {
    "default": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}

_SHAPE = re.compile(
    r"^(?P<head>.*?)\n\nQuestion: \{\}\n\n(?P<label>[A-Za-z ]+): \{\}\n\nModel Response: \{\}\n\n(?P<tail>.*)$",
    re.S,
)


def prompt_kind(dataset: str, category: str) -> str:
    """Which released template grades a question of this dataset and category."""
    if dataset == "longmemeval":
        if category in ("temporal-reasoning", "knowledge-update"):
            return category
        if category == "single-session-preference":
            return "preference"
    return "default"


def batch_rubric(kind: str) -> str:
    """System prompt that applies a released template to a batch of items.

    The template's wording is kept as is; only the three per-item slots are described, because a batch
    carries several items in one request. "yes" maps to label CORRECT and "no" to WRONG.
    """
    m = _SHAPE.match(TEMPLATES[kind])
    assert m, f"template {kind!r} changed shape"
    head, label, tail = m["head"].strip(), m["label"], m["tail"].strip()
    return (
        f"{head}\n\n"
        "You are given several items as JSON. Each has a question, a reference_answer (the "
        f"'{label}' of the instruction above) and a response (the 'Model Response'). "
        f"For every item, consider: {tail}\n"
        "Judge each item independently. Reply with label CORRECT when your answer would be 'yes' "
        "and WRONG when it would be 'no'."
    )
