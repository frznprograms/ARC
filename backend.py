from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()
CSV_FILE = "reviews.csv"

class Review(BaseModel):
    name: str
    category: str
    description: str
    review: str
    rating: int

@app.post("/submit_review/")
def submit_review(review: Review):
    data = review.model_dump()
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    df.to_csv(CSV_FILE, index=False)
    return {"status": "success"}
