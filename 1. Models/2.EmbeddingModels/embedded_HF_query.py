from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    task="feature-extraction"
)

vector = embedding.embed_query("Islamabad is the capital of Pakistan")

print(str(vector))