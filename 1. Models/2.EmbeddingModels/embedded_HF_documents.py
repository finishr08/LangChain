from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    task="feature-extraction"
)

documents = [
    "Islamabad is the capital of Pakistan",
    "Lahore is the capital of Punjab",
]

vector = embedding.embed_documents(documents)

print(str(vector))