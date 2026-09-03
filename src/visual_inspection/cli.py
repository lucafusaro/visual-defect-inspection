import argparse
import json
from pathlib import Path

from visual_inspection.artifacts import load_model_artifact
from visual_inspection.features import create_feature_extractor
from visual_inspection.inference import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run visual defect prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image", required=True, type=Path)
    predict_parser.add_argument("--artifact", required=True, type=Path)

    args = parser.parse_args()

    if args.command == "predict":
        artifact = load_model_artifact(args.artifact)
        model, preprocess = create_feature_extractor()
        result = predict_image(args.image, model, preprocess, artifact)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
