from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(context_chunks, question):
    context = "\n\n".join([c["content"] for c in context_chunks])

    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the context below.
    Provide a clear and slightly detailed answer (2-3 sentences).
    If the answer is not in the context, say:
    "I could not find the answer in the documents."

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content