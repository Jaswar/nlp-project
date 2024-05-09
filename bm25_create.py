from rank_bm25 import BM25Plus
import json
from tqdm import tqdm
from multiprocessing import Process, Queue, Manager


def thread_job(thread_id, claims, corpus, tokenized_corpus, output):
    result = []
    bm25 = BM25Plus(tokenized_corpus)
    for i, claim in enumerate(claims):
        print(f'Processing claim {i + 1} in thread {thread_id}')
        tokenized_claim = claim['claim'].split(' ')
        top_n = bm25.get_top_n(tokenized_claim, corpus, n=100)
        claim['top_n'] = top_n
        result.append(claim)
    output[thread_id] = result


def main():
    # uses "only" 40GB of RAM LOL
    # feel free ot decrease the number of processes
    num_processes = 16
    with open('NumTemp-E9C0/data/corpus_evidence_unified.json') as f:
        corpus = json.load(f)
    with open('NumTemp-E9C0/data/raw_data/val_claims_quantemp.json') as f:
        claims = json.load(f)

    corpus = [corpus[key] for key in corpus]

    tokenized_corpus = [doc.split(' ') for doc in corpus]

    manager = Manager()
    output = manager.dict()
    cpp = len(claims) // num_processes + 1
    processes = [Process(target=thread_job,
                         args=(i,
                               claims[i*cpp:(i+1)*cpp],
                               corpus,
                               tokenized_corpus,
                               output)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    result = []
    for key in output:
        result += output[key]

    print(len(result))

    with open('NumTemp-E9C0/output/bm25_top_100_val', 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()

