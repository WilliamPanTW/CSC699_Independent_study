import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the API key
genai.configure(api_key=os.environ["API_KEY"])

# Correct file path (use raw string for Windows paths)
file_path = r'D:\CSC699_Independent_study\application\ocr\3.jpg'

# Upload file
myfile = genai.upload_file(file_path)

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Ask the model to return the attributes in a structured format
result = model.generate_content(
    [myfile, "Extract the roll, scene, and take numbers from this image and return them as a JSON object with keys 'roll', 'scene', and 'take'."]
)

# Handle Markdown-style JSON response
raw_output = result.text
print("Raw Output:", raw_output)

# Remove Markdown backticks if present
if raw_output.startswith("```json") and raw_output.endswith("```"):
    json_text = raw_output[7:-3].strip()  # Strip "```json" and "```"
else:
    json_text = raw_output.strip()

# Parse the cleaned JSON text
try:
    attributes = json.loads(json_text)  # Parse the JSON
except json.JSONDecodeError as e:
    print(f"Error: The model response is not in valid JSON format. Details: {e}")
    attributes = {}

# Save the attributes to a JSON file
if attributes:
    output_file = "extracted_attributes.json"
    with open(output_file, "w") as f:
        json.dump(attributes, f, indent=4)
    print(f"Extracted attributes saved to {output_file}:")
    print(attributes)
else:
    print("No attributes were extracted or saved.")
