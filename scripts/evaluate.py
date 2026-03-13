"""
scripts/evaluate.py
-------------------

Evaluation script for anomaly detection models.

Responsibilities
----------------

1 Load trained model
2 Load validation/test dataset
3 Run inference
4 Compute metrics
5 Print results
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

    return parser.parse_args()



def main():

    args = parse_args()

    # -------------------------------------------------
    # 1 Load trained model
    # -------------------------------------------------

    # TODO:
    # model = load_model(args.checkpoint)

    # -------------------------------------------------
    # 2 Load dataset
    # -------------------------------------------------

    # TODO:
    # dataset = AnomalyDataset(...)
    # dataloader = DataLoader(...)

    # -------------------------------------------------
    # 3 Create evaluator
    # -------------------------------------------------

    # TODO:
    # evaluator = Evaluator(model, dataloader)

    # -------------------------------------------------
    # 4 Run evaluation
    # -------------------------------------------------

    # TODO:
    # results = evaluator.evaluate()

    # -------------------------------------------------
    # 5 Print metrics
    # -------------------------------------------------

    # TODO:
    # print(results)

    pass


if __name__ == "__main__":
    main()