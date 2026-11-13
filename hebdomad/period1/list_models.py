import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

for m in client.models.list().data:
    print(m.id)

    '''allam-2-7b
whisper-large-v3
openai/gpt-oss-20b
canopylabs/orpheus-arabic-saudi
meta-llama/llama-prompt-guard-2-22m
meta-llama/llama-prompt-guard-2-86m
openai/gpt-oss-safeguard-20b
qwen/qwen3.8-27b
openai/gpt-oss-120b
canopylabs/orpheus-v1-english
whisper-large-v3-turbo

it will update by the groq by api key'''