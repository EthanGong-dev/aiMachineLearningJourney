import chromadb
import ollama

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from typing import List


# ==========================================
# 1. Connect to ChromaDB
# ==========================================

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="pdf_rag"
)


# ==========================================
# 2. Create custom Retriever
# ==========================================

class ChromaRetriever(BaseRetriever):

    def _get_relevant_documents(self, query: str) -> List[Document]:

        # Convert question into embedding
        response = ollama.embed(
            model="nomic-embed-text",
            input=query
        )

        query_embedding = response["embeddings"][0]


        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )


        # Store retrieved documents
        documents = []


        for i in range(
            len(results["documents"][0])
        ):

            text = results["documents"][0][i]

            metadata = results["metadatas"][0][i]


            document = Document(
                page_content=text,
                metadata=metadata
            )


            documents.append(document)


        return documents


# ==========================================
# 3. Create retriever
# ==========================================

retriever = ChromaRetriever()


# ==========================================
# 4. Ask question
# ==========================================

question = input(
    "Enter your question: "
)


# ==========================================
# 5. Retrieve documents
# ==========================================

documents = retriever.invoke(
    question
)


# ==========================================
# 6. Display results
# ==========================================

print("\n===== RETRIEVED DOCUMENTS =====")


for i, document in enumerate(
    documents,
    start=1
):

    print("\n----------------------------")

    print("Document:", i)

    print("Page:", document.metadata["page"])

    print("Source:", document.metadata["source"])

    print("\nContent:")

    print(document.page_content[:500])