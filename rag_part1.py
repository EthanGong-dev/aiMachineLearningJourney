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
    name="rag_knowledge"
)


# ==========================================
# 3. Read knowledge file
# ==========================================

with open("knowledge.txt", "r", encoding="utf-8") as file:

    knowledge = file.read()


# ==========================================
# 4. Split knowledge into chunks
# ==========================================

chunks = knowledge.split("\n\n")


# ==========================================
# 5. Store chunks in ChromaDB
# ==========================================

collection.add(
    documents=chunks,

    ids=[
        f"chunk{i}"
        for i in range(len(chunks))
    ]
)


# ==========================================
# 6. Ask user
# ==========================================

question = input("\nYou: ")


# ==========================================
# 7. Search relevant information
# ==========================================

results = collection.query(
    query_texts=[question],
    n_results=2
)


# ==========================================
# 8. Get retrieved documents
# ==========================================

retrieved_documents = results["documents"][0]


print("\nRetrieved information:")

for document in retrieved_documents:

    print("-", document)


# ==========================================
# 9. Combine retrieved information
# ==========================================

context = "\n\n".join(retrieved_documents)


# ==========================================
# 10. Send context to Qwen
# ==========================================

response = chat(

    model="qwen3:1.7b",

    messages=[

        {
            "role": "system",

            "content": """
You are an AI assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the context,
say that you do not have enough information.
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
# 11. Display answer
# ==========================================

print("\nAI:")

print(response.message.content)