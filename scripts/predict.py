"""
scripts/predict.py
------------------

Run inference on new data using a trained anomaly model.

Typical usage:

    python scripts/predict.py \
        --checkpoint checkpoints/model.pth \
        --input path/to/video.mp4
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video or image"
    )

    return parser.parse_args()



def main():

    args = parse_args()

    # -------------------------------------------------
    # 1 Load model
    # -------------------------------------------------

    # TODO:
    # model = load_model(args.checkpoint)

    # -------------------------------------------------
    # 2 Load input data
    # -------------------------------------------------

    # TODO:
    # video / frames / image preprocessing

    # -------------------------------------------------
    # 3 Run inference
    # -------------------------------------------------

    # TODO:
    # outputs = model(input_tensor)

    # -------------------------------------------------
    # 4 Convert logits to prediction
    # -------------------------------------------------

    # TODO:
    # prediction = torch.argmax(outputs)

    # -------------------------------------------------
    # 5 Print / save results
    # -------------------------------------------------

    # TODO:
    # print("Prediction:", prediction)

    pass


if __name__ == "__main__":
    main()