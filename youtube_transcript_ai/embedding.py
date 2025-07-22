from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()

print("Loaded API Key:", os.getenv("GOOGLE_API_KEY"))

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

def data_embedding(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    # print("Vector store created with FAISS.")
    # print(f"Number of documents in vector store: {len(vector_store)}")
    # print(vector_store.index_to_docstore_id)
    return vector_store