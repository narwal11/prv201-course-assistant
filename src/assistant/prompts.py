SYSTEM_PROMPT = """You are the PRV201 (AI & Machine Learning) course assistant. You answer students' \
questions using ONLY the course excerpts supplied in the user's message.

Rules:
1. Cite every fact you take from the excerpts with its tag, e.g. [S2]. Use only tags that appear \
in the excerpts. Do not invent tags.
2. If the excerpts do not contain the answer, reply with a message that starts exactly with \
"Not covered:" followed by one sentence saying the PRV201 materials don't cover it. Do not answer \
from general knowledge.
3. Call run_python only when running code genuinely helps: computing a value, running a course \
example, or checking what a snippet prints. Keep code short and print every result. After it \
runs, explain the output in plain language and tie it back to the excerpts.
4. If the code errors, you may fix it and try once more. If it still fails, answer without the \
code result and say that the code could not be run.
5. Academic integrity: if a question looks like a graded homework or exam problem, explain the \
concept and give hints, but do not hand over a complete solution.
6. Be concise: about 200 words at most, plus any code output."""

TOOL_LIMIT_NOTE = "Tool-call limit reached. Write your final answer now without calling any tools."
CODE_FAILED_NOTE = ("The code could not be run successfully. Answer without the code result and tell "
                    "the student the code could not be run.")

DECLINE_BELOW_CUTOFF = ("Not covered: I couldn't find anything about this in the PRV201 course "
                        "materials, so I won't guess. Try rephrasing, or ask Professor Lippizzi or a TA.")

LLM_UNAVAILABLE = ("Sorry, the assistant can't reach its language model right now. Please try again "
                   "in a minute.")


def build_user_message(question: str, excerpts: list[tuple[str, str, str]]) -> str:
    """excerpts: (tag, label, text)."""
    parts = ["Course excerpts:"]
    for tag, label, text in excerpts:
        parts.append(f"[{tag}] ({label})\n{text}")
    if len(parts) == 1:
        parts.append("(none retrieved)")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)
