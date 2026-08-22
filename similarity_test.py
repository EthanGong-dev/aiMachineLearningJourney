from ollama import embed
import math


# =================================
# 1. Function untuk buat embedding
# =================================

def get_embedding(text):

    response = embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


# =================================
# 2. Function Cosine Similarity
# =================================

def cosine_similarity(vector_a, vector_b):

    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0

    for i in range(len(vector_a)):

        dot_product += vector_a[i] * vector_b[i]

        magnitude_a += vector_a[i] ** 2

        magnitude_b += vector_b[i] ** 2


    magnitude_a = math.sqrt(magnitude_a)

    magnitude_b = math.sqrt(magnitude_b)


    if magnitude_a == 0 or magnitude_b == 0:
        return 0


    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity


# =================================
# 3. Sentences
# =================================

sentence_a = "I love programming"

sentence_b = "I enjoy coding"

sentence_c = "The weather is very hot today"


# =================================
# 4. Create embeddings
# =================================

embedding_a = get_embedding(sentence_a)

embedding_b = get_embedding(sentence_b)

embedding_c = get_embedding(sentence_c)


# =================================
# 5. Calculate similarity
# =================================

similarity_ab = cosine_similarity(
    embedding_a,
    embedding_b
)

similarity_ac = cosine_similarity(
    embedding_a,
    embedding_c
)

similarity_bc = cosine_similarity(
    embedding_b,
    embedding_c
)


# =================================
# 6. Print results
# =================================

print("\nSimilarity Results")

print("-------------------------")

print("A ↔ B:", similarity_ab)

print("A ↔ C:", similarity_ac)

print("B ↔ C:", similarity_bc)