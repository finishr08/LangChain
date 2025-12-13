from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

chatModel = GoogleGenerativeAI(model="gemini-2.5-flash")

result: str = chatModel.invoke("Who is the best Cricket Captain  in pakistan Era? only names")

print(result)