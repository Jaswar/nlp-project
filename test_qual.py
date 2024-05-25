import argparse
import json
import os
import random

nli_models = ['BARTlargeMNLI', 'deberta']
generative_models = ['bart', 'flant5', 'gpt']
math_models = ['ElasticBERTlarge', 'MathRoBERTa']  # numt5 and pasta removed due to poor performance
all_models = nli_models + generative_models + math_models


def print_subset(subset, claims, mapping, ground_truth, predictions):
    for i in random.sample(subset, min(args.n, len(subset))):
        print(f'Claim: {claims[i]["claim"]}')
        for e in claims[i]['top_n'][:10]:
            print(f'Evidence: {e}')
        print(f'Label: {mapping[ground_truth[i]]}')
        for model, preds in predictions.items():
            print(f'{model}: {mapping[preds[i]]}')
        print()


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
        model = ''.join(c for c in model if c.isalnum())
        with open(joint_path) as f:
            preds = f.read().split(',')
            preds = [int(p) for p in preds]
            predictions[model] = preds

    # only choose models defined in all_models
    # this is because some models (numt5, pasta) were removed due to poor performance
    # keeping them would cause some intersection sets to be empty
    filtered_predictions = {}
    for model in all_models:
        filtered_predictions[model] = predictions[model]
    predictions = filtered_predictions

    incorrect_claims = {}
    correct_claims = {}
    for model, preds in predictions.items():
        incorrect_claims[model] = set()
        correct_claims[model] = set()
        for i, (pred, gt) in enumerate(zip(preds, ground_truth)):
            if pred != gt:
                incorrect_claims[model].add(i)
            else:
                correct_claims[model].add(i)

    nli_generative_incorrect = set.intersection(
        *[set(incorrect_claims[model]) for model in nli_models + generative_models])
    math_incorrect = set.intersection(*[set(incorrect_claims[model]) for model in math_models])
    nli_generative_correct = set.intersection(
        *[set(correct_claims[model]) for model in nli_models + generative_models])
    math_correct = set.intersection(*[set(correct_claims[model]) for model in math_models])

    # get cases where nli/generative incorrect, math correct
    nli_gen_inc_math_cor = list(nli_generative_incorrect.intersection(math_correct))
    # get cases where nli/generative correct, math incorrect
    nli_gen_cor_math_inc = list(nli_generative_correct.intersection(math_incorrect))
    # get cases where all incorrect
    all_incorrect = list(nli_generative_incorrect.intersection(math_incorrect))

    print('Incorrect claims where NLI/generative models are incorrect and math models are correct:')
    print_subset(nli_gen_inc_math_cor, claims, mapping, ground_truth, predictions)

    print('Incorrect claims where NLI/generative models are correct and math models are incorrect:')
    print_subset(nli_gen_cor_math_inc, claims, mapping, ground_truth, predictions)

    print('Incorrect claims where all models are incorrect:')
    print_subset(all_incorrect, claims, mapping, ground_truth, predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims_path', type=str, help='Path to the .json file containing the claims')
    parser.add_argument('--pred_path', type=str, help='Path to the folder containing the predictions')
    parser.add_argument('--gt_path', type=str, help='Path to the .csv file containing the ground truth')
    parser.add_argument('--n', type=int, default=5, help='The number of incorrect claims to print')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for claims reordering')
    args = parser.parse_args()
    main(args)
