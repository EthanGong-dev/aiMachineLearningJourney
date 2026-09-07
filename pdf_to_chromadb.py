import chromadb
import ollama
from pypdf import PdfReader


# ==========================================
# 1. Open PDF
# ==========================================

reader = PdfReader("notes.pdf")


# ==========================================
# 2. Extract pages
# ==========================================

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
# 3. Chunking function
# ==========================================

def create_chunks(text, chunk_size=100, overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(
            " ".join(chunk)
        )

        start += chunk_size - overlap

    return chunks


# ==========================================
# 4. Create chunks + metadata
# ==========================================

all_chunks = []


for document in documents:

    page_text = document["text"]

    page_number = document["page"]

    chunks = create_chunks(
        page_text,
        chunk_size=100,
        overlap=20
    )


    for chunk in chunks:

        all_chunks.append({

            "text": chunk,

            "page": page_number,

            "source": "notes.pdf"

        })


print("Total chunks:", len(all_chunks))


# ==========================================
# 5. Create ChromaDB
# ==========================================

client = chromadb.Client()


collection = client.get_or_create_collection(
    name="pdf_notes"
)


# ==========================================
# 6. Create embeddings
# ==========================================

for i, chunk in enumerate(all_chunks):

    response = ollama.embed(

        model="nomic-embed-text",

        input=chunk["text"]

    )

    embedding = response["embeddings"][0]


    # ======================================
    # 7. Store in ChromaDB
    # ======================================

    collection.add(

        ids=[f"chunk_{i}"],

        documents=[chunk["text"]],

        embeddings=[embedding],

        metadatas=[{

            "source": chunk["source"],

            "page": chunk["page"]

        }]

    )


print("All chunks stored in ChromaDB!")


# ==========================================
# 8. Ask a question
# ==========================================

question = input("\nYou: ")


# ==========================================
# 9. Create embedding for question
# ==========================================

question_response = ollama.embed(

    model="nomic-embed-text",

    input=question

)

question_embedding = question_response["embeddings"][0]


# ==========================================
# 10. Search ChromaDB
# ==========================================

results = collection.query(

    query_embeddings=[question_embedding],

    n_results=3

)


# ==========================================
# 11. Display results
# ==========================================

print("\n===== SEARCH RESULTS =====")


for i in range(
    len(results["documents"][0])
):

    document = results["documents"][0][i]

    metadata = results["metadatas"][0][i]

    distance = results["distances"][0][i]


    print("\n--------------------------")

    print("Distance:", distance)

    print("Source:", metadata["source"])

    print("Page:", metadata["page"])

    print("Text:")

    print(document[:500])