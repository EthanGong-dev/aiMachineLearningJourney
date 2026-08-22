import chromadb


# =========================
# 1. Create ChromaDB client
# =========================

client = chromadb.Client()


# =========================
# 2. Create a collection
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
# 4. Check database
# =========================

print("Database created successfully!")

print("Number of documents:",
      collection.count())