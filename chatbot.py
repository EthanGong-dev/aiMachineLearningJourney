from ollama import chat

#Set AI Personality
messages = [
    {
        'role': 'system',
        'content': 'You are a helpful AI tutor.'
    }
]

#Chat Berulang Kali
while True:
    #Ambil Soalan User
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    #Simpan Soalan Dalam History
    messages.append({
        'role': 'user',
        'content': question
    })

    #Hantar Ke Ollama
    response = chat(
        model='qwen3:1.7b',
        messages=messages
    )

    answer = response['message']['content']

    #Cetak Jawapan
    print("\nAI:", answer)

    #Simpan Jawapan AI
    messages.append({
        'role': 'assistant',
        'content': answer
    })