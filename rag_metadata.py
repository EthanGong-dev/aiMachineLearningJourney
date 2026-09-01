import chromadb
from ollama import chat


# ==========================================
# 1. Create ChromaDB client
# ==========================================

client = chromadb.Client()


# ==========================================
# 2. Create collection
# ==========================================

collection = client.create_collection(
    name="study_notes"
)


# ==========================================
# 3. Our knowledge
# ==========================================

documents = [

    "Python is a high-level programming language.",

    "Machine learning allows computers to learn patterns from data.",

    "RAG allows an AI system to retrieve relevant information before generating an answer.",

    "Vector databases store embeddings and allow semantic search."

]


# ==========================================
# 4. Metadata
# ==========================================

metadata = [

    {
        "source": "AI_Notes.pdf",
        "page": 3,
        "topic": "Python"
    },

    {
        "source": "AI_Notes.pdf",
        "page": 12,
        "topic": "Machine Learning"
    },

    {
        "source": "AI_Notes.pdf",
        "page": 18,
        "topic": "RAG"
    },

    {
        "source": "AI_Notes.pdf",
        "page": 21,
        "topic": "Vector Database"
    }

]


# ==========================================
# 5. Add documents + metadata
# ==========================================

collection.add(

    documents=documents,

    metadatas=metadata,

    ids=[
        "chunk1",
        "chunk2",
        "chunk3",
        "chunk4"
    ]

)


# ==========================================
# 6. Ask question
# ==========================================

question = input("\nYou: ")


# ==========================================
# 7. Search database
# ==========================================

results = collection.query(

    query_texts=[question],

    n_results=2

)


# ==========================================
# 8. Get results
# ==========================================

retrieved_documents = results["documents"][0]

retrieved_metadata = results["metadatas"][0]


# ==========================================
# 9. Display retrieved information
# ==========================================

print("\nRetrieved information:")

for document, meta in zip(
    retrieved_documents,
    retrieved_metadata
):

    print("\nContent:")
    print(document)

    print("\nSource:")
    print(meta["source"])

    print("Page:")
    print(meta["page"])

    print("Topic:")
    print(meta["topic"])


# ==========================================
# 10. Create context
# ==========================================

context_parts = []


for document, meta in zip(
    retrieved_documents,
    retrieved_metadata
):

    context_parts.append(

        f"""
Content:
{document}

Source: {meta["source"]}
Page: {meta["page"]}
Topic: {meta["topic"]}
"""

    )


context = "\n".join(context_parts)


# ==========================================
# 11. Ask Qwen
# ==========================================

response = chat(

    model="qwen3:1.7b",

    messages=[

        {
            "role": "system",

            "content": """
You are an AI study assistant.

Answer the question using ONLY the provided context.

Do not invent information.

Always provide the source and page
when answering.
"""
        },

        {
            "role": "user",

            "content": f"""
Context:

{context}


Question:

{question}
"""
        }

    ]

)


# ==========================================
# 12. Display answer
# ==========================================

print("\nAI:")

print(response.message.content)