import chromadb
import ollama
from pypdf import PdfReader


# ==========================================
# 1. Read PDF
# ==========================================

reader = PdfReader("notes.pdf")

documents = []


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text()

    if text:

        documents.append({
            "text": text,
            "page": page_number
        })


# ==========================================
# 2. Chunking
# ==========================================

def create_chunks(text, chunk_size=100, overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


# ==========================================
# 3. Create chunks
# ==========================================

all_chunks = []


for document in documents:

    chunks = create_chunks(
        document["text"],
        chunk_size=100,
        overlap=20
    )

    for chunk in chunks:

        all_chunks.append({

            "text": chunk,

            "page": document["page"],

            "source": "notes.pdf"

        })


print("Total chunks:", len(all_chunks))


# ==========================================
# 4. Create ChromaDB
# ==========================================

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="pdf_rag"
)


# ==========================================
# 5. Add embeddings
# ==========================================

for i, chunk in enumerate(all_chunks):

    embedding_response = ollama.embed(

        model="nomic-embed-text",

        input=chunk["text"]

    )

    embedding = embedding_response["embeddings"][0]


    collection.add(

        ids=[f"chunk_{i}"],

        documents=[chunk["text"]],

        embeddings=[embedding],

        metadatas=[{

            "source": chunk["source"],

            "page": chunk["page"]

        }]

    )


print("PDF stored in vector database!")


# ==========================================
# 6. User question
# ==========================================

question = input("\nYou: ")


# ==========================================
# 7. Embed question
# ==========================================

question_response = ollama.embed(

    model="nomic-embed-text",

    input=question

)

question_embedding = question_response["embeddings"][0]


# ==========================================
# 8. Search
# ==========================================

results = collection.query(

    query_embeddings=[question_embedding],

    n_results=3

)


retrieved_documents = results["documents"][0]

retrieved_metadata = results["metadatas"][0]


# ==========================================
# 9. Build context
# ==========================================

context = ""


for document, metadata in zip(
    retrieved_documents,
    retrieved_metadata
):

    context += f"""

Source: {metadata["source"]}
Page: {metadata["page"]}

Content:
{document}

"""


# ==========================================
# 10. Send context to Qwen
# ==========================================

response = ollama.chat(

    model="qwen3:1.7b",

    messages=[

        {
            "role": "system",

            "content": """
You are a helpful AI study assistant.

Answer the user's question using ONLY
the provided context.

If the answer is not in the context,
say that the information is not available
in the document.

Do not invent information.
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


# ==========================================
# 12. Display sources
# ==========================================

print("\nSources:")

for metadata in retrieved_metadata:

    print(
        f"- {metadata['source']} "
        f"(Page {metadata['page']})"
    )