from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print('Reasearch Tool')

paper_input = "GPT-3: Language Models are Few-Shot Learners"
style_input = "Beginner-Friendly"
length_input = "Medium (3-5 paragraphs)"

template = PromptTemplate(
    template="""
Please analyze the cricket report or document titled "{paper_input}" with the following specifications:
Analysis Style: {style_input}  
Analysis Length: {length_input}  

1. Statistical Breakdown:  
   - Include relevant cricket statistics (e.g., runs, wickets, strike rate, economy, partnerships).  
   - If applicable, explain calculations like run rate, strike rate, required run rate, or probability models in simple terms.  

2. Cricketing Logic & Analogies:  
   - Use relatable cricket analogies (e.g., comparing strategies to powerplay decisions, field placements, captaincy calls by PCB-level standards).  
   - Simplify complex cricket strategies or performance analysis using intuitive examples.  

3. Contextual Insights:  
   - Mention team dynamics, PCB selection policies, player form, match conditions, or tactical choices — but only if they are present in the document.  

If certain information is not available in the document, respond with:  
"Insufficient information available"  
instead of guessing.

Ensure the analysis is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True
)

chain = template | model

result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

print(result.content)
