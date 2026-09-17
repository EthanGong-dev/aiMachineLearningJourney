from fastapi import FastAPI
from pydantic import BaseModel, Field
import ollama

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question that the user wants to ask the AI"
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls how creative the AI response is"
    )


class AIResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def home():
    return {
        "message": "AI Backend is running!"
    }


@app.post("/ask", response_model=AIResponse)
def ask_question(data: QuestionRequest):

    response = ollama.chat(
        model="qwen3:1.7b",
        messages=[
            {
                "role": "user",
                "content": data.question
            }
        ],
        options={
            "temperature": data.temperature
        }
    )

    return {
        "question": data.question,
        "answer": response["message"]["content"]
    }