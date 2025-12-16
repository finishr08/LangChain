from langchain_community.document_loaders import CSVLoader

file_path = "D:/Work experience/GitHub REPO/LangChain/5. RAG/1.Document Loaders/store/data.csv"

loader = CSVLoader(file_path=file_path)

docs = loader.lazy_load()

for i in docs:
    print(i.page_content)