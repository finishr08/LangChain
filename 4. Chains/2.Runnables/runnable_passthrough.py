from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables

load_dotenv()

# Define the prompt

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

# create the str output parser

parser = StrOutputParser()

# Define the prompt for explanation

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# create the sequence chain

joke_gen_chain = RunnableSequence(prompt1, model, parser)

# create the parallel chain

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

# create the final chain

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'cricket'})

print(result)