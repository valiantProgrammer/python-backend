from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect
from pymongo import MongoClient
from googletrans import Translator
import json
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load IPC categories
json_path = os.path.join(os.path.dirname(__file__), "ipc_categories.json")
with open(json_path, "r", encoding="utf-8") as file:
    ipc_categories = json.load(file)

# Load model from Hugging Face Hub
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",  # Directly from Hugging Face
    device=-1  # CPU; use `device=0` if GPU is available
)

# MongoDB (change to use Atlas if deploying to cloud)
client = MongoClient(
    "mongodb+srv://rupayandey134:Cd0pPzvDey0MemLG@cluster0.mwbajxb.mongodb.net")
db = client["BailSETU"]
collection = db["ipc_sections"]

# Translator
translator = Translator()


class InputData(BaseModel):
    message: str


def translate_doc_to_hindi(doc):
    fields = ["title", "description", "punishment",
              "bail_type", "bail_time_limit"]
    for field in fields:
        if field in doc and isinstance(doc[field], str):
            try:
                doc[field] = translator.translate(
                    doc[field], src="en", dest="hi").text
            except Exception:
                pass
    return doc


@app.post("/api")
async def classify(data: InputData):
    message = data.message.strip()
    if not message or len(message.split()) < 2:
        return ""

    try:
        lang = detect(message)
    except:
        return ""

    is_hindi = lang == "hi"
    translated = translator.translate(
        message, src="hi", dest="en").text if is_hindi else message

    try:
        result = classifier(translated, candidate_labels=ipc_categories)
    except Exception:
        return ""

    top_categories = [
        label for label, score in zip(result["labels"], result["scores"])
        if score >= 0.1
    ]

    if not top_categories:
        return ""

    mongo_results = []
    for category in top_categories:
        docs = list(collection.find({"category": category}, {"_id": 0}))
        mongo_results.extend(docs)

    if is_hindi:
        mongo_results = [translate_doc_to_hindi(
            doc.copy()) for doc in mongo_results]

    return {
        "original_language": lang,
        "matched_categories": top_categories,
        "sections": mongo_results
    }
