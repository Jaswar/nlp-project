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
        if 'predicted_programs' in claim:
            program = claim['predicted_programs'][0]
            program = ' '.join(p for p in program)
            tokenized_program = program.split(' ')
            tokenized_claim += tokenized_program
        top_n = bm25.get_top_n(tokenized_claim, corpus, n=100)
        claim['top_n'] = top_n
        result.append(claim)
    output[thread_id] = result

def thread_job_sq(thread_id, claims, corpus, tokenized_corpus, output):
    result = []
    bm25 = BM25Plus(tokenized_corpus)
    for i, claim in enumerate(claims):
        print(f'Processing claim {i + 1} in thread {thread_id}')
        subqs = claim['subquestions']
        n = 100 // len(subqs)
        top_n = []
        for subq in subqs:
            tokenized_subq = subq.split(' ')
            top_n_subq = bm25.get_top_n(tokenized_subq, corpus, n=n)
            top_n.extend(top_n_subq)
            if "rotavirus" in subqs[0]:
                print(subq)
                print(top_n_subq)
                print("\n\n\n")

        claim['top_n'] = top_n
        result.append(claim)
    output[thread_id] = result


def main():
    # uses "only" 40GB of RAM LOL
    # feel free ot decrease the number of processes
    file_type = "val"
    claim_file_name = f"{file_type}_claim_with_subq.json"
    claim_dir = "NumTemp-E9C0/data/claim_with_subq_data_prompt/"
    claim_file = claim_dir + claim_file_name

    num_processes = 8
    with open('NumTemp-E9C0/data/corpus_evidence_unified.json') as f:
        corpus = json.load(f)
    with open(claim_file) as f:
        claims = json.load(f)
        print(len(claims))

    corpus = [corpus[key] for key in corpus]

    tokenized_corpus = [doc.split(' ') for doc in corpus]

    manager = Manager()
    output = manager.dict()
    cpp = len(claims) // num_processes + 1
    processes = [Process(target=thread_job_sq,
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

    output_dir = "NumTemp-E9C0/output/"
    output_file_name = f"bm25_top_100_{file_type}_subq_new.json"
    output_file = output_dir + output_file_name

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()

