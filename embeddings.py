import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    return client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding


def product_to_text(name: str, code: str, description: str, attributes: dict) -> str:
    attrs = (attributes or {})
    attr_str = " ".join(f"{k} {v}" for k, v in attrs.items())
    attr_list = ", ".join(f"{k}: {v}" for k, v in attrs.items())
    return f"{name} {code}. {description or ''}. {attr_str}. Specifications: {attr_list}"
