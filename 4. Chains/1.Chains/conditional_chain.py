from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal

# load env variables

load_dotenv()

# hugging face model setup

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# create string format output parser

parser = StrOutputParser()

# create the Feedback class

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# create the pydantic parser

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# create the prompt

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

# create the classifier chain for analyzing its neg or pos

classifier_chain = prompt1 | model | parser2

# create the prompt for positive feedback

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# create the prompt for negative feedback

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# create the branch chain and add conditions

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

# create the final chain

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a 3rd Class phone'}))