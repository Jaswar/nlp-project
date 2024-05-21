import argparse
import json
from sklearn.metrics import accuracy_score
import os
import random


def main(args):
    random.seed(args.seed)

    with open(args.claims_path) as f:
        claims = json.load(f)
    with open(args.gt_path) as f:
        ground_truth = f.read().split(',')
        ground_truth = [int(gt) for gt in ground_truth]

    mapping = {}
    for claim, gt in zip(claims, ground_truth):
        str_gt = claim['label']
        if str_gt not in mapping:
            mapping[gt] = str_gt

    predictions = {}
    for file in os.listdir(args.pred_path):
        joint_path = os.path.join(args.pred_path, file)
        model = file.replace('predictions', '').replace('.csv', '')
        model = ''.join(c for c in model if c.isalpha())
        with open(joint_path) as f:
            preds = f.read().split(',')
            preds = [int(p) for p in preds]
            predictions[model] = preds

    incorrect_claims = {}
    for model, preds in predictions.items():
        incorrect_claims[model] = set()
        for i, (pred, gt) in enumerate(zip(preds, ground_truth)):
            if pred != gt:
                incorrect_claims[model].add(i)

    intersection = set.intersection(*incorrect_claims.values())
    intersection = list(intersection)
    # this is to allow us to select which claims to print
    random.shuffle(intersection)
    intersection = intersection[:args.n]
    print(f'The claims to print are: {intersection}\n')
    for i in intersection:
        print(f"Claim: {claims[i]['claim']}")
        print(f"Ground truth: {mapping[ground_truth[i]]}")
        for model in predictions:
            model_prediction = predictions[model][i]
            print(f"{model}: {mapping[model_prediction]}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims_path', type=str, help='Path to the .json file containing the claims')
    parser.add_argument('--pred_path', type=str, help='Path to the folder containing the predictions')
    parser.add_argument('--gt_path', type=str, help='Path to the .csv file containing the ground truth')
    parser.add_argument('--n', type=int, default=5, help='The number of incorrect claims to print')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for claims reordering')
    args = parser.parse_args()
    main(args)
