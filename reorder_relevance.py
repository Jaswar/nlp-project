import json
import numpy as np
from tqdm import tqdm
import csv
import logging
tqdm.pandas()


def cosine_similarity(a, b):
    dot = np.sum(a * b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-6)


def separate_tuples(tuples_list):
    list1, list2 = zip(*tuples_list)
    list1 = list(list1)
    list2 = list(list2)
    return list1, list2


def reorder(claim, evidence, model):
    claim_embedding = model.encode(claim)
    evidence_embeddings = [model.encode(ev)for ev in evidence]
    similarities = [cosine_similarity(claim_embedding, evidence_embedding) for evidence_embedding in evidence_embeddings]
    sorted_evidence = list(sorted(zip(evidence, similarities), key=lambda x: x[1], reverse=True))
    return separate_tuples(sorted_evidence)


def main():
    def run(data):
        from sentence_transformers import SentenceTransformer
        logging.basicConfig(level=logging.ERROR)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        with open(f"./output/bm25_top_100_{data}", 'r') as f:
            claims = json.load(f)

        reordered_claims = []
        all_scores = []
        for claim in tqdm(claims):
            cl = claim['claim']
            evidence = claim['top_n']
            evidences, scores = reorder(cl, evidence, model)
            claim['top_n'] = evidences
            reordered_claims.append(claim)
            all_scores.append(scores)

        with open(f'./output/bm25_top_100_{data}_reorderedd.json', 'w') as f:
            json.dump(reordered_claims, f, indent=4)

        with open(f'./output/bm25_top_100_{data}_reordered_scores.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(all_scores)

    run('train')
    run('val')
    run('test')


if __name__ == '__main__':
    main()
