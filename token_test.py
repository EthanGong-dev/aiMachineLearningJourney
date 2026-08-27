from transformers import AutoTokenizer


# ==========================================
# 1. Load tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B"
)


# ==========================================
# 2. Text
# ==========================================

text = """
Python is a programming language.
Machine learning allows computers to learn
patterns from data.
"""


# ==========================================
# 3. Tokenize
# ==========================================

tokens = tokenizer.tokenize(text)


# ==========================================
# 4. Display tokens
# ==========================================

print("Text:")
print(text)

print("\nTokens:")

for i, token in enumerate(tokens):

    print(i, token)


# ==========================================
# 5. Count tokens
# ==========================================

words = text.split()

print("\nNumber of words:", len(words))
print("Number of tokens:", len(tokens))