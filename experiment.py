import openai

openai.base_url = "http://localhost:11434/v1/"
openai.api_key = "dummy-key"

response = openai.chat.completions.create(
    model="gemma3",
    messages=[
        {"role": "user", "content": "Hello, how are you?"},
    ],
)

print(response.choices[0].message.content)
