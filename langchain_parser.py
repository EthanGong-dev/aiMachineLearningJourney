from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. Create LLM
# ==========================================

llm = ChatOllama(
    model="qwen3:1.7b"
)


# ==========================================
# 2. Create Prompt
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI tutor.

Explain this topic in simple terms:

{topic}
"""
)


# ==========================================
# 3. Create Output Parser
# ==========================================

parser = StrOutputParser()


# ==========================================
# 4. Create Chain
# ==========================================

chain = prompt | llm | parser


# ==========================================
# 5. Get user input
# ==========================================

topic = input("Enter a topic: ")


# ==========================================
# 6. Run chain
# ==========================================

result = chain.invoke({
    "topic": topic
})


# ==========================================
# 7. Display result
# ==========================================

print("\nAI:")
print(result)

print("\nData type:")
print(type(result))