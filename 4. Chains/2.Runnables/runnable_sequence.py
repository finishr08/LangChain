from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables

load_dotenv()

# Create the prompt template for wriing joke of the topic

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Create the huggingface endpoint for text generation

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# create the hugging face model

model = ChatHuggingFace(llm = llm)

# create the str ouput parser

parser = StrOutputParser()

# create the prompt template for explaining the joke

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# create the runnable sequence chain

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# invoke the chain

result = chain.invoke({'topic':'AI'})

print(result)
