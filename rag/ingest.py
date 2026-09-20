import chromadb
import ollama
from pypdf import PdfReader


PDF_PATH = "data/notes.pdf"
CHROMA_PATH = "./chroma_db"


def ingest_pdf():

    reader = PdfReader(PDF_PATH)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="pdf_knowledge"
    )

    for index, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            response = ollama.embed(
                model="nomic-embed-text",
                input=text
            )

            embedding = response["embeddings"][0]

            collection.upsert(
                ids=[f"page_{index}"],
                documents=[text],
                embeddings=[embedding]
            )

    print("PDF successfully added to ChromaDB.")


if __name__ == "__main__":
    ingest_pdf()