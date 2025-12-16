from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# create the hugging face model

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.amazon.com/Apple-iPhone-16-Pro-Max/dp/B0DHJ896RY/ref=sr_1_1?dib=eyJ2IjoiMSJ9.kWyuq8QIuEDqxOYOskeWdbIErhBu76iUxpZ7WNaCwx3C5LpGSu7KhmnjBcKzLvMeZygdNI9sfRu8UFa02jopkKZFoAujPdTWJ48q0hoSQ_ly_jTJ2cPFPJxdcoB2_PCNPNDikZfCdooLjTHrvMe_-8HzKXWo0isPOhQPmJD95YOXlMIk4LOQBMWV2snyu-5TOJjbYcqnLSghXK0IYEi7p8EvP2fPofF3Va2WVYVVKkU.M3hrxZ4FobcOUcnwxfPrcQeaAxPBnZ2HOj87c7Emg0k&dib_tag=se&keywords=iphone&qid=1765871081&sr=8-1&th=1'
loader = WebBaseLoader(url)

docs = loader.lazy_load()


chain = prompt | model | parser

for doc in docs:
    result = chain.invoke({
        'question': 'What is the product that we are talking about?',
        'text': doc.page_content
    })
    print(result)
