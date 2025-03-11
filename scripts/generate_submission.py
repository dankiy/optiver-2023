"""
Generate submission table using local test data and simulation of the Kaggle iterative
environment.
"""

import argparse
import joblib as jb
import json
import os
import sys

sys.path.append(".")

from src.modeling.inference import generate_submission  # noqa:E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=int,
        help="Model ID as an integer; use the latest if not specified.",
        default=-1,
    )
    args = parser.parse_args()
    if args.model_id == -1:
        model_paths = [fname for fname in os.listdir("models") if ".jb" in fname]
        model_id = int(model_paths[-1].replace(".jb", ""))
    else:
        model_id = args.model_id
    models = jb.load(f"models/{model_id}.jb")

    with open("configs/inference.json") as f:
        inference_config = json.load(f)

    generate_submission(
        models,
        inference_config["test_root"],
        model_id,
        inference_config["clip_min"],
        inference_config["clip_max"],
        inference_config["tail_size"],
    )


if __name__ == "__main__":
    main()
