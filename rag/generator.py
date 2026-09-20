import ollama


def generate_answer(question, context):

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful AI assistant.

Answer the user's question using ONLY
the provided context.

If the answer cannot be found in the context,
say that the information is not available
in the provided document.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:

{context}

Question:

{question}
"""
            }
        ]
    )

    return response["message"]["content"]