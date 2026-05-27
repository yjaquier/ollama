from ollama import chat, ChatResponse

response: ChatResponse = chat(
    model='llama3.2:latest',
    messages=[
        {'role': 'user', 'content': 'Explain Llama model in one sentence'}
    ]
)

print("=== Model response ===")
print(response.message.content)

print("\n=== Metadata ===")
print(f"Model used to generate response : {response.model}")
print(f"Number of tokens evaluated in the prompt : {response.prompt_eval_count}")
print(f"Number of tokens evaluated in inference : {response.eval_count}")
print(f"Duration of evaluating inference in seconds : {response.eval_duration / 1e9:.2f}s") # Duration of evaluating inference in nanoseconds / 1e9
print(f"Total duration in seconds : {response.total_duration / 1e9:.2f}s") # Total duration in nanoseconds / 1e9