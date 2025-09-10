from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
from src.pipelines.inference_pipeline import InferencePipeline
from io import StringIO
import json
import re
import asyncio
import torch

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


@app.get("/stage_counters/")
async def get_stage_counters():
    try:
        safety_count = pipeline.redis.get("safety_stage")
        fasttext_count = pipeline.redis.get("fasttext_stage")
        encoder_count = pipeline.redis.get("encoder_stage")

        return {
            "safety_stage": int(safety_count.decode("utf-8")) if safety_count else 0,
            "fasttext_stage": (
                int(fasttext_count.decode("utf-8")) if fasttext_count else 0
            ),
            "encoder_stage": int(encoder_count.decode("utf-8")) if encoder_count else 0,
        }
    except Exception as e:
        return {"safety_stage": 0, "fasttext_stage": 0, "encoder_stage": 0}


async def safety_stage(review_data):
    # Stage 1: Safety check
    pipeline.redis.incr("safety_stage")
    review = review_data.get("review", None)
    review = re.sub(r"\b(they|they're|them|up)\b", "", review, flags=re.IGNORECASE)  # type: ignore
    # remove extra spaces
    review = re.sub(r"\s+", " ", review).strip()

    if review is None:
        yield {"stage": 1, "status": "error", "message": "Review is empty"}
        return

    if isinstance(review, str):
        review = [review]

    safe_value = pipeline.safety_model.predict(review)
    pred_strength = pipeline.safety_model.predict_proba(review)[:, 1]

    if safe_value > 0:
        pipeline.add_banned_ids(pipeline.user_id)
        yield {
            "stage": 1,
            "status": "rejected",
            "message": f"Review failed safety check (probability: {pred_strength[0]:.3f})",
        }
        return
    else:
        yield {"stage": 1, "status": "passed", "message": "Safety check passed"}

    await asyncio.sleep(0.2)


async def fasttext_stage(review_data, prompt):
    # Stage 2: Fasttext check
    pipeline.redis.incr("fasttext_stage")
    yield {
        "stage": 2,
        "status": "starting",
        "message": "Running fasttext classification...",
    }
    await asyncio.sleep(0.1)
    fasttext_results = pipeline.fasttext_model.predict_all_heads(prompt)
    thresholds = {"ad": 0.7, "irrelevant": 0.7, "rant": 0.7, "unsafe": 0.7}
    for cat, prob in fasttext_results.items():
        if cat == "ad" and prob < thresholds[cat]:
            url_pattern = r"(?:https://[^\s]+|www\.[^\s]+|[^\s]+\.com(?:/[^\s]*)?)"
            match = re.search(url_pattern, review_data["review"])
            if match:
                prob = max(1, prob + 0.4)
        # if any cat exceeds threshold, fail it
        if prob > thresholds[cat]:
            yield {
                "stage": 2,
                "status": "rejected",
                "message": f"Review rejected by fasttext heads: {cat}",
            }
            return

    max_positive_confidence = max(fasttext_results.values())
    early_accept_threshold = 0.3

    if max_positive_confidence <= early_accept_threshold:
        yield {
            "stage": 2,
            "status": "passed",
            "message": f"Early acceptance triggered: max confidence {max_positive_confidence:.3f} <= {early_accept_threshold}, skipping Stage 3. Review accepted!",
        }
        return
    yield {
        "stage": 2,
        "status": "uncertain",
        "message": "Fasttext confidence within uncertain range",
    }
    await asyncio.sleep(0.2)


async def encoder_stage(prompt):
    # stage 3 encoder check
    pipeline.redis.incr("encoder_stage")
    yield {"stage": 3, "status": "starting", "message": "Running encoder model..."}
    await asyncio.sleep(0.1)

    inputs = pipeline.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = pipeline.encoder(**inputs)
        probs = torch.sigmoid(outputs.logits)
        preds = (probs > 0.5).int()

    # Check if any prediction is positive (rejected)
    has_positive_pred = torch.any(preds > 0).item()

    # Get prediction scores for each bucket for console logging
    scores = probs.squeeze().tolist()
    bucket_names = ["ad", "irrelevant", "rant", "unsafe"]
    score_details = {bucket_names[i]: round(scores[i], 3) for i in range(len(scores))}

    if not has_positive_pred:
        yield {
            "stage": 3,
            "status": "passed",
            "message": "Review passed all checks and was accepted!",
            "scores": score_details,
        }
    else:
        # Find which labels triggered rejection
        failed_labels = [bucket_names[i] for i in range(len(preds)) if preds[0, i] > 0]
        max_prob_idx = probs.argmax().item()
        primary_label = bucket_names[max_prob_idx]

        if len(failed_labels) == 1:
            reject_reason = f"'{primary_label}' (probability: {probs.max().item():.3f})"
        else:
            reject_reason = f"'{primary_label}' and {len(failed_labels)-1} other(s) (max probability: {probs.max().item():.3f})"

        yield {
            "stage": 3,
            "status": "rejected",
            "message": f"Review rejected by encoder for {reject_reason}",
            "scores": score_details,
        }


@app.post("/analyze_review_stream/")
async def analyze_review_stream(request: ReviewRequest):

    review_data = request.review

    async def generate_stream():
        try:
            # check if the id is in the banned list
            value = pipeline.redis.get(pipeline.user_id)

            if value:
                value = value.decode("utf-8")  # convert bytes
                if int(value) == -1:  # compare as integer
                    logger.warning(
                        "This user has been flagged for reviews that did not pass our pipeline in the past"
                    )
                    yield f"data: {json.dumps({'stage': 0, 'user_id': value, 'status': 'banned', 'message': 'This user has been flagged for reviews that did not pass our pipeline in the past'})}\n\n"
                    return
            yield f"data: {json.dumps({'stage': 1, 'status': 'starting', 'message': 'Starting safety check...'})}\n\n"
            await asyncio.sleep(0.1)  # Small delay for UI


            # stage 1
            async for message in safety_stage(review_data):
                yield f"data: {json.dumps(message)}\n\n"
                if message["status"] in {"error", "rejected", "banned"}:
                    pipeline.add_banned_ids(value)
                    return

                if message['status'] in {"error","rejected","banned"}:
                    pipeline.add_banned_ids(pipeline.user_id)
                    return 
                
            prompt = f"""
                Business Name: {review_data["name"]}
                Category: {review_data["category"]}
                Description: {review_data["description"]}
                Review: {review_data["review"]}
                Rating: {review_data["rating"]}
            """.replace(
                "\n", ""
            ).strip()

            # stage 2
            async for message in fasttext_stage(review_data, prompt):
                yield f"data: {json.dumps(message)}\n\n"
                if message["status"] in {"rejected"}:
                    pipeline.add_banned_ids(value)
                    return
                if message['status'] in {"rejected"}:
                    pipeline.add_banned_ids(pipeline.user_id)
                    return 
                if message['status'] == "passed":
                    return

            # Stage 3: Encoder check
            async for message in encoder_stage(prompt):
                yield f"data: {json.dumps(message)}\n\n"
                if message['status'] in {"rejected"}:
                    print('hi')
                    print(value)
                    pipeline.add_banned_ids(pipeline.user_id)
                    return             

        except Exception as e:
            yield f"data: {json.dumps({'stage': -1, 'status': 'error', 'message': f'Pipeline error: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
