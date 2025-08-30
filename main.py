import argparse
import json

from src.pipelines.inference_pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run safety inference pipeline")

    parser.add_argument(
        "--safety-model",
        type=str,
        default="models/safety-model-test.pkl",
        help="Path to the safety model .pkl file (default: models/safety-model-test.pkl)",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="lora_sft_encoder.pth",
        help="Path to encoder model weights (.pth) (default: lora_sft_encoder.pth)",
    )
    parser.add_argument(
        "--review-file",
        type=str,
        default="data/for_model/review_1.json",
        help="Path to JSON review file (default: data/for_model/review_1.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for fasttext heads (default: 0.7)",
    )

    args = parser.parse_args()

    # Initialize pipeline with args or defaults
    pipeline = InferencePipeline(
        safety_model_path=args.safety_model,
        encoder_model_path=args.encoder_model,
    )

    # Load review JSON
    with open(args.review_file, "r") as f:
        review = json.load(f)

    # Run inference
    pipeline.run_inference(review, default_threshold=args.threshold)


if __name__ == "__main__":
    main()
