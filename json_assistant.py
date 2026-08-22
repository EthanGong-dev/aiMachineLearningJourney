from ollama import chat
import json

question = input("Enter a person description: ")

response = chat(
    model='qwen3:1.7b',
    messages=[
        {
            'role': 'system',
            'content': '''
Extract the information and return ONLY valid JSON.

Format:
{
    "name": "",
    "age": 0,
    "course": ""
}
'''
        },
        {
            'role': 'user',
            'content': question
        }
    ]
)

print(response['message']['content'])