# ==========================================
# RAG Advanced - Word Based Chunking
# ==========================================


text = """
Python is a high-level programming language.
Python is widely used in software development,
data analysis, artificial intelligence,
machine learning, and automation.

Functions are reusable blocks of code.
A function can receive input through parameters
and can return a result.

Machine learning is a branch of artificial intelligence.
Machine learning allows computers to learn patterns
from data without being explicitly programmed
for every possible situation.

Retrieval-Augmented Generation, or RAG,
allows an AI system to retrieve relevant information
from an external knowledge source before generating
an answer.
"""


# ==========================================
# 1. Convert text into words
# ==========================================

words = text.split()

print("Total words:", len(words))


# ==========================================
# 2. Create chunks
# ==========================================

def create_chunks(words, chunk_size, overlap):

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
# 3. Generate chunks
# ==========================================

chunks = create_chunks(
    words,
    chunk_size=30,
    overlap=5
)


# ==========================================
# 4. Display chunks
# ==========================================

print("\nNumber of chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i} ---")

    print(chunk)