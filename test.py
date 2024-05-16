import argparse
import json
from sklearn.metrics import accuracy_score


def main(args):
    with open(args.claims_path) as f:
        claims = json.load(f)
        categories = [claim['taxonomy_label'] for claim in claims]
    with open(args.pred_path) as f:
        predictions = f.read().split(',')
        predictions = [int(p) for p in predictions]
    with open(args.gt_path) as f:
        ground_truth = f.read().split(',')
        ground_truth = [int(gt) for gt in ground_truth]

    preds_per_category = {}
    for i, category in enumerate(categories):
        category = category.strip()
        if category not in preds_per_category:
            preds_per_category[category] = []
        preds_per_category[category].append((predictions[i], ground_truth[i]))
    print('Number of samples per category:')
    for category, preds in preds_per_category.items():
        print(f'{category}: {len(preds)}')
    print()

    print('Accuracy per category:')
    for category, preds in preds_per_category.items():
        y_pred, y_true = zip(*preds)
        acc = accuracy_score(y_true, y_pred)
        print(f'{category}: {round(acc * 100, 2)}%')
    print()

    # This should match the accuracy from the main training script
    print(f'Overall accuracy: {round(accuracy_score(ground_truth, predictions) * 100, 2)}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims_path', type=str, help='Path to the .json file containing the claims')
    parser.add_argument('--pred_path', type=str, help='Path to the .csv file containing the predictions')
    parser.add_argument('--gt_path', type=str, help='Path to the .csv file containing the ground truth')
    args = parser.parse_args()
    main(args)
