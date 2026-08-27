# ==========================================
# RAG Advanced - Chunking
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
# 1. Simple chunking
# ==========================================

sentences = text.strip().split("\n")

print("Number of sentences:", len(sentences))


# ==========================================
# 2. Display sentences
# ==========================================

print("\nSentences:")

for i, sentence in enumerate(sentences):

    print(f"{i}: {sentence}")


def create_chunks(sentences, chunk_size, overlap):

    chunks = []

    start = 0

    while start < len(sentences):

        end = start + chunk_size

        chunk = sentences[start:end]

        chunks.append(
            " ".join(chunk)
        )

        start += chunk_size - overlap

    return chunks


chunks = create_chunks(
    sentences,
    chunk_size=3,
    overlap=1
)


print("\nChunks:")

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i}:")
    print(chunk)