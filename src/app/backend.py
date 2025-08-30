from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from src.pipelines.inference_pipeline import InferencePipeline
from io import StringIO

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Review schema
class ReviewRequest(BaseModel):
    review: dict


# Capture logs in memory
log_stream = StringIO()
logger.remove()
logger.add(log_stream, format="{level} | {message}", level="SUCCESS")
logger.add(log_stream, format="{level} | {message}", level="WARNING")

# Load pipeline once
pipeline = InferencePipeline(safety_model_path="models/safety-model-test.pkl")


@app.post("/analyze_review/")
async def analyze_review(request: ReviewRequest):
    # Reset logs
    log_stream.seek(0)
    log_stream.truncate(0)

    try:
        pipeline.run_inference(request.review)
        logger.success("Inference completed successfully.")
    except Exception as e:
        logger.warning(f"Pipeline error: {e}")

    # Collect logs
    log_stream.seek(0)
    logs = []
    for line in log_stream.readlines():
        if line.startswith("SUCCESS"):
            logs.append({"type": "success", "message": line.strip()})
        elif line.startswith("WARNING"):
            logs.append({"type": "warning", "message": line.strip()})

    return {"logs": logs[1:]}
