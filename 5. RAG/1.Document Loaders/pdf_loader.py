from langchain_community.document_loaders import PyPDFLoader

file_path = "D:/Work experience/GitHub REPO/LangChain/5. RAG/1.Document Loaders/store/curriculum.pdf"

loader = PyPDFLoader(file_path=file_path)

docs = loader.lazy_load()

for i in docs:
    print(i.page_content)