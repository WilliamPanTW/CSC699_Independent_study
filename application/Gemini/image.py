import os 
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the API key
genai.configure(api_key=os.environ["API_KEY"])

# Correct file path (use raw string for Windows paths)
file_path = r'D:\CSC699_Independent_study\application\map\test.jpg'

# Upload file
myfile = genai.upload_file(file_path)
# print(f"{myfile=}")

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Generate content
result = model.generate_content(
    [myfile, "what is the scene,take and roll number?"]
)

# Print the result
print(result.text)