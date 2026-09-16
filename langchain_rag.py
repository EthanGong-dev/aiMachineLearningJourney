import ollama

from pypdf import PdfReader

from langchain_chroma import Chroma

from langchain_ollama import ChatOllama

from langchain_core.documents import Document

from langchain_core.embeddings import Embeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 1. Ollama Embeddings
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

pages = []


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text()

    if text:

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": "notes.pdf",
                    "page": page_number
                }
            )
        )


print("Pages loaded:", len(pages))


# ==========================================
# 3. Chunk PDF
# ==========================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100
)


chunks = splitter.split_documents(
    pages
)


print("Chunks created:", len(chunks))


# ==========================================
# 4. Create embeddings
# ==========================================

embedding_model = OllamaEmbeddings()


# ==========================================
# 5. Create ChromaDB
# ==========================================

vector_store = Chroma(

    collection_name="complete_pdf_rag",

    embedding_function=embedding_model
)


# ==========================================
# 6. Store chunks
# ==========================================

vector_store.add_documents(
    chunks
)


print("Chunks stored in ChromaDB!")


# ==========================================
# 7. Create Retriever
# ==========================================

retriever = vector_store.as_retriever(

    search_kwargs={
        "k": 3
    }

)


# ==========================================
# 8. Create Qwen
# ==========================================

llm = ChatOllama(

    model="qwen3:1.7b"
)


# ==========================================
# 9. Create Prompt
# ==========================================

prompt = ChatPromptTemplate.from_template(

    """
You are a helpful AI study assistant.

Answer the question using ONLY the
information provided in the context.

If the answer cannot be found in the
context, say:

"I cannot find the answer in the document."

Context:

{context}

Question:

{question}

Answer:
"""
)


# ==========================================
# 10. Output Parser
# ==========================================

parser = StrOutputParser()


# ==========================================
# 11. Ask question
# ==========================================

question = input(
    "\nYou: "
)


# ==========================================
# 12. Retrieve relevant chunks
# ==========================================

retrieved_documents = retriever.invoke(
    question
)


# ==========================================
# 13. Build context
# ==========================================

context = "\n\n".join(

    document.page_content

    for document in retrieved_documents

)


# ==========================================
# 14. Create RAG chain
# ==========================================

chain = prompt | llm | parser


# ==========================================
# 15. Generate answer
# ==========================================

answer = chain.invoke({

    "context": context,

    "question": question

})


# ==========================================
# 16. Display answer
# ==========================================

print("\n==============================")

print("AI ANSWER")

print("==============================")

print(answer)


# ==========================================
# 17. Display sources
# ==========================================

print("\n==============================")

print("SOURCES")

print("==============================")


for document in retrieved_documents:

    print(
        f"- {document.metadata['source']} "
        f"(Page {document.metadata['page']})"
    )