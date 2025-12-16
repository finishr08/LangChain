from langchain_community.document_loaders import TextLoader

file_path = "D:/Work experience/GitHub REPO/LangChain/5. RAG/1.Document Loaders/store/cricket.txt"

loader = TextLoader(file_path=file_path)

docs = loader.lazy_load()

for i in docs:
    print(i.page_content)