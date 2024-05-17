## Models

 - BART-large-MNLI: https://www.kaggle.com/code/alexandraneagu101/nlp-bart-large-mnli 
 - Roberta-Large-MNLI:
 - sileod/deberta-v3-base-tasksource-nli: `deberta.ipynb`
 - FlanT5: 
 - GPT2: 
 - BART:
 - MathRoberta: `nlp-math-roberta.ipynb`
 - NumT5:
 - LUNA:
 - PASTA:
 - ElasticRoBERTa: Couldn't set up ElasticRoBERTa, set up ElasticBERT(-large) instead: https://www.kaggle.com/code/alexandraneagu101/nlp-elasticbert-large

## Scripts

 - `test.py`: Script for the quantitative evaluation of the models. It computes the accuracy per each `taxonomy_label` and the overall accuracy.
The script can be run with the following command:
```bash
python test.py --claims_path=<claims_path> --pred_path=<pred_path> --gt_path=<ground_truth_path>
```
For example:
```bash
python test.py --claims_path=NumTemp-E9C0/output/bm25_top_100_test --pred_path=predictions.csv --gt_path=ground_truth.csv
```
