## Models

The models can be found in the `models` directory.

 - BART-large-MNLI: `bart-large-mnli.ipynb`
 - Roberta-Large-MNLI: `roberta-large-mnli.ipynb`
 - sileod/deberta-v3-base-tasksource-nli: `deberta.ipynb`
 - FlanT5: `flant5.ipynb`
 - GPT2: `gpt2.ipynb`
 - BART: `bart.ipynb`
 - Math(Ro)BERTa: `math-roberta.ipynb`
 - NumT5: `numt5.ipynb`
 - PASTA: `pasta.ipynb`
 - ElasticBERT: `elastic-bert.ipynb`

### Utility models:

- Claim type classifier: BART-large-MNLI: https://www.kaggle.com/code/alexandraneagu101/claim-type-bart-large-mnli
- Claim decomposition - Flan-T5-large prompting: https://www.kaggle.com/code/alexandraneagu101/claim-decomposition
- Claim decomposition - Flan-T5-large fine-tuned on StrategyQA: https://www.kaggle.com/code/alexandraneagu101/flan-t5-large-on-strategyqa

## Scripts

 - `test_quant.py`: Script for the quantitative evaluation of the models. It computes the accuracy per each `taxonomy_label` and the overall accuracy.
The script can be run with the following command:
```bash
python test_quant.py --claims_path=<claims_path> --pred_path=<pred_path> --gt_path=<ground_truth_path>
```
For example:
```bash
python test_quant.py --claims_path=NumTemp-E9C0/output/bm25_top_100_test --pred_path=./predictions/predictions_deberta.csv --gt_path=ground_truth.csv
```

 - `test_qual.py`: Script for the qualitative evaluation of the models. It prints the claims that were classified incorrectly by all models.
The script can be run with the following command:
```bash
python test_qual.py --claims_path=<claims_path> --pred_path=<pred_path> --gt_path=<ground_truth_path>
```
For example:
```bash
python test_qual.py --claims_path=NumTemp-E9C0/output/bm25_top_100_test --pred_path=./predictions --gt_path=ground_truth.csv
```

The script also optionally takes the number of claims to print (`--n`, default set to 5) and the seed used for the random selection of the claims (`--seed`).
