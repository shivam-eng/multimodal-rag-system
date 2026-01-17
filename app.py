from search import search_images
from rag import generate_answer

from dotenv import load_dotenv
load_dotenv()


query = input("Enter your query: ")

images = search_images(query)
print("\n🔍 Retrieved Images:")
for img in images:
    print(img)

answer = generate_answer(query, images)
print("\n🤖 RAG Answer:\n", answer)
