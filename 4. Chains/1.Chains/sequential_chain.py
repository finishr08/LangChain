from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# load env variables

load_dotenv()

# create the hugging face model

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm= llm)

# create the prompt for generating report

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# create the prompt for generating summary

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# create the parser for str output

parser = StrOutputParser()

# create the chain to generate report and summary

chain = prompt1 | model | parser | prompt2 | model | parser

# invoke the chain

result = chain.invoke({'topic': 'Software Engineer Jobs in Lahore,Pakistan'})

print(result)