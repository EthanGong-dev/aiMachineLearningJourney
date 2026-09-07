from langchain_ollama import ChatOllama


# ==========================================
# 1. Create LLM
# ==========================================

llm = ChatOllama(
    model="qwen3:1.7b"
)


# ==========================================
# 2. Ask question
# ==========================================

response = llm.invoke(
    "Explain what machine learning is in simple terms."
)


# ==========================================
# 3. Display answer
# ==========================================

print(response.content)