from ApiKeys import openAPI
import os 
os.environ['OPENAI_API_KEY'] = openAPI

from langchain_openai import OpenAI  # Correct import


llm = OpenAI(temperature=0.6)
name=llm("How to be rich")
print(name)


