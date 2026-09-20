from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.retriever import retrieve_documents
from rag.generator import generate_answer


app = FastAPI()


class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=1
    )


@app.get("/")
def home():

    return {
        "message": "RAG API is running!"
    }


@app.post("/ask")
def ask_question(data: QuestionRequest):

    documents = retrieve_documents(
        data.question
    )

    context = "\n\n".join(documents)

    answer = generate_answer(
        data.question,
        context
    )

    return {
        "question": data.question,
        "answer": answer,
        "sources": documents
    }