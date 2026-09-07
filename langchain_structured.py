from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ==========================================
# 1. Define the output structure
# ==========================================

class Person(BaseModel):

    name: str = Field(
        description="The person's name"
    )

    age: int = Field(
        description="The person's age"
    )

    course: str = Field(
        description="The person's university course"
    )


# ==========================================
# 2. Create LLM
# ==========================================

llm = ChatOllama(
    model="qwen3:1.7b"
)


# ==========================================
# 3. Tell LLM to follow our structure
# ==========================================

structured_llm = llm.with_structured_output(Person)


# ==========================================
# 4. Create prompt
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
Extract the person's information.

Return:
- name
- age
- course

Person description:

{description}
"""
)


# ==========================================
# 5. Create chain
# ==========================================

chain = prompt | structured_llm


# ==========================================
# 6. Get user input
# ==========================================

description = input(
    "Enter a person description: "
)


# ==========================================
# 7. Run chain
# ==========================================

result = chain.invoke({
    "description": description
})


# ==========================================
# 8. Display result
# ==========================================

print("\nResult:")

print(result)

print("\nName:", result.name)
print("Age:", result.age)
print("Course:", result.course)