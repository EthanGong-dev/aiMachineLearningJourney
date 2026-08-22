from ollama import chat


# =========================
# 1. Calculator Tool
# =========================

def calculator(a: int, b: int, operation: str):

    operation = operation.lower().strip()

    if operation in ["add", "addition", "plus"]:
        return a + b

    elif operation in ["subtract", "subtraction", "minus"]:
        return a - b

    elif operation in ["multiply", "multiplication", "times"]:
        return a * b

    elif operation in ["divide", "division"]:
        return a / b

    return "Invalid operation"


# =========================
# 2. Available Tools
# =========================

tools = [
    calculator
]


# =========================
# 3. Conversation History
# =========================

messages = [
    {
        "role": "system",
        "content": """
        You are a helpful AI assistant.

        You have access to a calculator tool.

        "Use the calculator whenever the user asks for mathematical calculations.

        The calculator operation MUST be exactly one of:
        - add
        - subtract
        - multiply
        - divide

        For multiplication, always use 'multiply'.
        For addition, always use 'add'.
        For subtraction, always use 'subtract'.
        For division, always use 'divide'."

        After receiving the calculator result,
        give the user a clear final answer.
        """
    }
]


# =========================
# 4. Chat Loop
# =========================

while True:

    user_input = input("\nYou: ")

    # Exit command
    if user_input.lower() == "exit":
        print("Goodbye!")
        break


    # Add user message to history
    messages.append({
        "role": "user",
        "content": user_input
    })


    # =========================
    # 5. First AI Call
    # =========================

    response = chat(
        model="qwen3:1.7b",
        messages=messages,
        tools=tools
    )


    # Save AI response
    messages.append(response.message)


    # =========================
    # 6. Check Tool Call
    # =========================

    if response.message.tool_calls:

        for tool_call in response.message.tool_calls:

            if tool_call.function.name == "calculator":

                args = tool_call.function.arguments


                # Execute calculator
                result = calculator(
                    args["a"],
                    args["b"],
                    args["operation"]
                )


                print("Tool used: calculator")
                print("Calculator result:", result)


                # Send result back to AI
                messages.append({
                    "role": "tool",
                    "content": str(result)
                })


        # =========================
        # 7. Second AI Call
        # =========================

        final_response = chat(
            model="qwen3:1.7b",
            messages=messages,
            tools=tools
        )


        print("\nAI:", final_response.message.content)


        # Save final AI response
        messages.append(final_response.message)


    else:

        # AI didn't need a tool
        print("\nAI:", response.message.content)