import chromadb


# =========================
# 1. Create ChromaDB client
# =========================

client = chromadb.Client()


# =========================
# 2. Create collection
# =========================

collection = client.create_collection(
    name="my_first_database"
)


# =========================
# 3. Add documents
# =========================

collection.add(
    documents=[
        "Python is a programming language.",
        "Java is an object-oriented programming language.",
        "The CPU executes instructions.",
        "Machine learning allows computers to learn patterns from data.",
        "Recursion happens when a function calls itself."
    ],

    ids=[
        "doc1",
        "doc2",
        "doc3",
        "doc4",
        "doc5"
    ]
)


# =========================
# 4. User question
# =========================

question = input("\nAsk a question: ")


# =========================
# 5. Search database
# =========================

results = collection.query(
    query_texts=[question],
    n_results=2
)


# =========================
# 6. Display results
# =========================

print("\nMost relevant documents:")

for document in results["documents"][0]:
    print("-", document)


print("\nDistances:")

for distance in results["distances"][0]:
    print(distance)