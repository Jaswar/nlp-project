import argparse
import json
from sklearn.metrics import accuracy_score, f1_score


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
    sorted_categories = sorted(preds_per_category.keys())
    for category in sorted_categories:
        preds = preds_per_category[category]
        y_pred, y_true = zip(*preds)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        if not args.latex:
            print(f'F1 score for {category}: {round(f1 * 100, 2)}%')
            print(f'Accuracy for {category}: {round(acc * 100, 2)}%')
        else:
            print(f'{round(f1 * 100, 2)} & {round(acc * 100, 2)} & ', end='')
    if not args.latex:
        print()

    # This should match the accuracy from the main training script
    overall_accuracy = accuracy_score(ground_truth, predictions)
    overall_f1 = f1_score(ground_truth, predictions, average='weighted')
    if not args.latex:
        print(f"Overall F1 score: {round(overall_f1 * 100, 2)}%")
        print(f'Overall accuracy: {round(overall_accuracy * 100, 2)}%')
    else:
        print(f'{round(overall_f1 * 100, 2)} & {round(overall_accuracy * 100, 2)} \\\\')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims_path', type=str, help='Path to the .json file containing the claims')
    parser.add_argument('--pred_path', type=str, help='Path to the .csv file containing the predictions')
    parser.add_argument('--gt_path', type=str, help='Path to the .csv file containing the ground truth')
    parser.add_argument('--latex', type=bool, default=False, help='Whether to output the results in a latex format')
    args = parser.parse_args()
    main(args)
