import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

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

    prompt = f"""
    Business Name: {data["name"]}
    Category: {data["category"]}
    Description: {data["description"]}
    Review: {data["review"]}
    Rating: {data["rating"]}
    """

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    df.to_csv(CSV_FILE, index=False)
    return {"status": prompt}
