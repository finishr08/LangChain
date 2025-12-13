from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""I recently upgraded to the iPhone 17 Pro, and I must say, it's an absolute powerhouse! The A19 Bionic chip makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The enhanced all-day battery easily lasts a full day even with heavy use, and the next-generation 60W fast charging is a lifesaver.

The Action Button customization is a great touch for quick access to apps and shortcuts, though I don't use it often. What really blew me away is the 48MP Main Camera with new Computational Photography features—the Deep Fusion Night mode is stunning, capturing crisp, vibrant images even in low light. The optical zoom up to 10x actually works well for distant objects, but anything beyond 50x digital zoom loses quality.

However, the ProMotion Display's variable refresh rate still feels a bit too aggressive in certain apps. Also, Apple’s iOS still has some limitations, like restricting third-party app stores—why can’t I truly customize my home screen widgets the way I want? The $1,200 starting price tag is also a hard pill to swallow.

✅ Pros:
Insanely powerful A19 Bionic processor (great for gaming and productivity)

Stunning 48MP camera with incredible computational capabilities

Long all-day battery life with next-generation fast charging

Action Button support is unique and useful

Review by Mustafa Ahmed""")

print(result)