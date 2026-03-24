from google import genai
from google.genai import types
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
# 1. SETUP
client = genai.Client(api_key=api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')


db_client = chromadb.PersistentClient(path='./vancouver_bylaws_db')
collection = db_client.get_or_create_collection(name='vancouver_law')

# 2. EXTRACTION

reader = PdfReader("./vancouver_law.pdf")
full_content = ""

for page in reader.pages:

    full_content += page.extract_text() + " "

# 3. CHUNKING
chunks = [full_content[i:i+1000] for i in range(0, len(full_content), 800)]
ids = [f"id_{i}" for i in range(len(chunks))]

# 4. LOADING (With a safety check)
if collection.count() == 0:
    print("Database is empty. Loading PDF...")
    collection.add(
        embeddings=model.encode(chunks).tolist(),
        ids=ids,
        documents=chunks
    )
    print(f"Success! Loaded {len(chunks)} chunks.")
else:
    print(f"Database already contains {collection.count()} chunks. Skipping load.")

print(f"Success! Loaded {len(chunks)} chunks into your AI memory.")


def ai_ask(question):
    q_vec = model.encode(chunks).tolist()

    result = collection.query(query_embeddings=q_vec, n_results=2)
    context = "\n".join(result["documents"][0])

    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=f"context:{context}\n\nquestion:{question}",
        config=types.GenerateContentConfig(
            system_instruction='you are a  traffic law assistant'
        )
    )
    return response.text
print(ai_ask("what is the parking spot price for van ? "))