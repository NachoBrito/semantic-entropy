"""LLM Wrapper.

Provides utilities for initializing a local GGUF language model using
llama-cpp-python and querying it with prompt validation via ask_llm.
"""
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="lmstudio-community/Qwen3.5-0.8B-GGUF",
    filename="*Q8_0.gguf",
    verbose=False
)

def ask_llm(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 128,
) -> str:
    if not isinstance(question, str):
        raise TypeError(f"Question must be a string, got {type(question).__name__}.")

    if not isinstance(temperature, (int, float)):
        raise TypeError(f"Temperature must be a float or int, got {type(temperature).__name__}.")

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty or whitespace only.")

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who answers questions in a single brief but complete sentence.",
            },
            {"role": "user", "content": cleaned_question},
        ],
        temperature=float(temperature),
        max_tokens=max_tokens,
    )

    if not isinstance(response, dict):
        raise ValueError(f"Expected dict response from LLM, got {type(response).__name__}.")

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response did not contain a valid 'choices' list.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("First choice in LLM response is not a valid dictionary.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response choice is missing a valid 'message' object.")

    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(
            f"Expected string content in LLM message, got {type(content).__name__ if content is not None else 'None'}."
        )

    return content.strip()