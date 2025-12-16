from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = "D:/Work experience/GitHub REPO/LangChain/5. RAG/2.Text Splitters/curriculum.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)