import ollama
import chromadb

from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ==========================================
# 1. Create Ollama Embedding Class
# ==========================================

class OllamaEmbeddings(Embeddings):

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            response = ollama.embed(
                model="nomic-embed-text",
                input=text
            )

            embeddings.append(
                response["embeddings"][0]
            )

        return embeddings


    def embed_query(self, text):

        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )

        return response["embeddings"][0]


# ==========================================
# 2. Read PDF
# ==========================================

reader = PdfReader("notes.pdf")

documents = []


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text()

    if text:

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "notes.pdf",
                    "page": page_number
                }
            )
        )


print(
    "PDF pages loaded:",
    len(documents)
)


# ==========================================
# 3. Create embeddings
# ==========================================

embedding_model = OllamaEmbeddings()


# ==========================================
# 4. Create Chroma vector store
# ==========================================

vector_store = Chroma(
    collection_name="langchain_pdf",
    embedding_function=embedding_model
)


# ==========================================
# 5. Add documents
# ==========================================

vector_store.add_documents(
    documents
)


print("Documents stored in ChromaDB!")


# ==========================================
# 6. Create Retriever
# ==========================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ==========================================
# 7. Ask question
# ==========================================

question = input(
    "\nEnter your question: "
)


# ==========================================
# 8. Retrieve documents
# ==========================================

results = retriever.invoke(
    question
)


# ==========================================
# 9. Display results
# ==========================================

print(
    "\n===== RESULTS ====="
)


for i, document in enumerate(
    results,
    start=1
):

    print(
        f"\n--- Document {i} ---"
    )

    print(
        "Source:",
        document.metadata["source"]
    )

    print(
        "Page:",
        document.metadata["page"]
    )

    print(
        "\nContent:"
    )

    print(
        document.page_content[:500]
    )