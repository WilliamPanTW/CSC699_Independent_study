import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)