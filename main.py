from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Backend is running!"
    }


@app.post("/ask")
def ask_question(data: Question):
    return {
        "question": data.question,
        "answer": "This is a test AI response."
    }