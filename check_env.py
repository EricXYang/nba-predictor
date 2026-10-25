import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    print("No key found. Check the .env file.")
else:
    print("Key found. Starts with:", key[:7], "| Length:", len(key))