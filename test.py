import google.generativeai as genai

genai.configure(api_key="AIzaSyByUjYci4Xv7d5AE6hSO4FCaaWmAqABYWs")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)