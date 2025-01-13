import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
