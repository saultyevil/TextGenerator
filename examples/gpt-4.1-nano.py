import asyncio
import os

from text_generator import TextGenerationInput, TextGenerator

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    exc_msg = "OPENAI_API_KEY env var containing API key is not set"
    raise ValueError(exc_msg)

client = TextGenerator("gpt-4.1-nano", api_key)
request = TextGenerationInput(text="Hello. What can you do?")
response = asyncio.run(client.generate_response_with_context([request]))

print(f"Input:\n{request}\n")
print(f"Response:\n{response}\n")
print(f"Internal state:\n{client.history}")
