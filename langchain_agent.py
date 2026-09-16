from langchain_ollama import ChatOllama

from langchain_core.tools import tool

from langchain.agents import create_agent


# ==========================================
# 1. Create Calculator Tool
# ==========================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Example: 25 * 48
    """

    try:

        result = eval(
            expression,
            {"__builtins__": {}}
        )

        return str(result)

    except Exception:

        return "Invalid calculation"


# ==========================================
# 2. Create LLM
# ==========================================

llm = ChatOllama(
    model="qwen3:1.7b"
)


# ==========================================
# 3. Give tools to Agent
# ==========================================

tools = [
    calculator
]


# ==========================================
# 4. Create Agent
# ==========================================

agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt="""
You are a helpful AI assistant.

Use the calculator tool whenever
the user asks you to perform calculations.
"""
)


# ==========================================
# 5. User input
# ==========================================

question = input(
    "You: "
)


# ==========================================
# 6. Run Agent
# ==========================================

result = agent.invoke({

    "messages": [

        {
            "role": "user",

            "content": question
        }

    ]

})


# ==========================================
# 7. Get final response
# ==========================================

final_message = result["messages"][-1]


print("\nAI:")

print(final_message.content)