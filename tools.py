def calculator(a, b, operation):

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        return a / b

    else:
        return "Invalid operation"