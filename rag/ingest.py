import chromadb
import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/notes.pdf"
CHROMA_PATH = "./chroma_db"


def ingest_pdf():

    # 1. Read PDF
    reader = PdfReader(PDF_PATH)

    # 2. Prepare text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # 3. Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="pdf_knowledge"
    )

    chunk_number = 0

    # 4. Process every page
    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if not text:
            continue

        # 5. Split page into smaller chunks
        chunks = splitter.split_text(text)

        # 6. Process every chunk
        for chunk in chunks:

            # Create embedding
            response = ollama.embed(
                model="nomic-embed-text",
                input=chunk
            )

            embedding = response["embeddings"][0]

            # Save chunk into ChromaDB
            collection.upsert(
                ids=[f"chunk_{chunk_number}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "page": page_number + 1,
                        "chunk": chunk_number
                    }
                ]
            )

            chunk_number += 1

    print(f"PDF successfully ingested.")
    print(f"Total chunks: {chunk_number}")


if __name__ == "__main__":
    ingest_pdf()