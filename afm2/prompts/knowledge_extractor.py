SYSTEM_PROMPT = """You are a helpful AI agent, a part of knowledge bridge 2, aimed at compositing target modality based on extracted knowledge from other agents."""

NEXT_STEP_PROMPT = """The user has provided the following extracted knowledge:
{context}

And you should based on the extracted knowledge to compose a target modality `{target}`.
Considering the question about the target modality, please provide a detailed and comprehensive answer.

[QUESTION]: {question}

Your answer format:
[ANSWER]: ...

Your answer:"""

SUMMARY_PROMPT = """Based on the information provided by the user, and the QA pairs that extracted by the `{target}`. Summary a meaningful paragraph to describe the `{target}`.

- QA PAIRS:
{qa_pairs}

Your answer should be in the following format:
[ANSWER]:...."""


