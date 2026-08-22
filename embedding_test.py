from ollama import embed

text = "Python is a programming language"

response = embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Number of dimensions:", len(embedding))
print("First 10 numbers:")
print(embedding[:10])