from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

chat_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=chat_llm)

# result: str = model.invoke("What is the capital of Pakistan?")

result: str = model.invoke("Who is the best Cricket Captain  in pakistan Era? only names")
print(result.content)
