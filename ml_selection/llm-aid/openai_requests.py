from openai import OpenAI

print("API KEY:")
client = OpenAI(api_key=str(input()))

models = client.models.list()

while True:
    resp = client.chat.completions.create(
        model="chatgpt-4o-latest", messages=[{"role": "user", "content": str(input())}]
    )

    print(resp.choices[0].message.content)
