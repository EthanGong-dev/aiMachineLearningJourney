from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. Create LLM
# ==========================================

llm = ChatOllama(
    model="qwen3:1.7b"
)


# ==========================================
# 2. Create Prompt Template
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI tutor.

Explain the following topic in simple terms.

Topic: {topic}

The explanation must include:
1. Definition
2. Simple example
3. Why it is useful
"""
)


# ==========================================
# 3. Create Chain
# ==========================================

chain = prompt | llm


# ==========================================
# 4. Get user input
# ==========================================

topic = input("Enter a topic: ")


# ==========================================
# 5. Run chain
# ==========================================

response = chain.invoke({
    "topic": topic
})


# ==========================================
# 6. Display result
# ==========================================

print("\nAI:")
print(response.content)