import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


def cosine_similarity(a, b):
    dot = np.sum(a * b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-6)


def reorder(claim, evidence, model):
    claim_embedding = model.encode(claim)
    evidence_embeddings = [model.encode(ev)for ev in evidence]
    similarities = [cosine_similarity(claim_embedding, evidence_embedding) for evidence_embedding in evidence_embeddings]
    sorted_evidence = list(sorted(zip(evidence, similarities), key=lambda x: x[1], reverse=True))
    return [ev for ev, _ in sorted_evidence]


def main():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    with open('NumTemp-E9C0/output/bm25_top_100_test', 'r') as f:
        claims = json.load(f)

    reordered_claims = []
    for claim in tqdm(claims):
        cl = claim['claim']
        evidence = claim['top_n']
        evidence = reorder(cl, evidence, model)
        claim['top_n'] = evidence
        reordered_claims.append(claim)

    with open('NumTemp-E9C0/output/bm25_top_100_test_reordered.json', 'w') as f:
        json.dump(reordered_claims, f, indent=4)


if __name__ == '__main__':
    main()


