from pypdf import PdfReader


# ==========================================
# 1. Open PDF
# ==========================================

reader = PdfReader("notes.pdf")


# ==========================================
# 2. Store pages
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
# 3. Create chunks
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
# 4. Process every page
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


# ==========================================
# 5. Display results
# ==========================================

print("Total chunks:", len(all_chunks))


for i, chunk in enumerate(all_chunks):

    print("\n================================")

    print("Chunk:", i)

    print("Source:", chunk["source"])

    print("Page:", chunk["page"])

    print("Text:")

    print(chunk["text"][:300])