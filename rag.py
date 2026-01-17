from openai import OpenAI
import os

def generate_answer(query, retrieved_images):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Did you load env vars?")

    client = OpenAI(api_key=api_key)

    context = "\n".join(retrieved_images)

    prompt = f"""
You are a multimodal assistant.

User Question:
{query}

Relevant retrieved image file paths:
{context}

Based on this context, generate a helpful, grounded answer.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
