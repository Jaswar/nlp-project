from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
def test_data():
    file = "val"
    subq_claim_dir = "NumTemp-E9C0/data/claim_with_subq_data_prompt/"
    subq_claim_file_ending = "_claim_with_subq.json"

    with open(f"{subq_claim_dir}{file}{subq_claim_file_ending}", 'r') as f:
        data = json.load(f)

    similarity_threshold = 0.2

    # Function to calculate cosine similarity
    def calculate_similarity(text1, text2):
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]

    diff_in_row = 0
    # Process each item in the JSON data
    for i, item in enumerate(data):
        concatenated_questions = " ".join(item["subquestions"])
        claim = item["claim"]

        similarity = calculate_similarity(concatenated_questions, claim)

        if similarity < similarity_threshold:
            print(f"Row {i}")
            print(f"Claim {claim}")
            print(f"Subquestions {concatenated_questions}")
            print(f"Similarity: {similarity}")
            print("\n\n\n")
        else:
            diff_in_row = 0


def merge_data():
    file = "val"

    json_dir = "NumTemp-E9C0/data/raw_data/"
    json_file_ending = "_claims_quantemp.json"

    subq_dir = "NumTemp-E9C0/data/decomp_data_ft/"
    subq_file_ending = "_claims_quantemp_decomp_ft.json"

    subq_claim_dir = "NumTemp-E9C0/data/claim_with_subq_data_ft/"
    subq_claim_file_ending = "_claim_with_subq_ft.json"

    with open(f"{json_dir}{file}{json_file_ending}", 'r') as f:
        claims = json.load(f)
    with open(f"{subq_dir}{file}{subq_file_ending}", 'r') as f:
        # read the lists of lists that are not json format
        subqs = f.read().split('\n')
        subqs = [json.loads(subq) for subq in subqs if subq]

    for claim, subq in zip(claims, subqs):
        claim['subquestions'] = subq

    with open(f"{subq_claim_dir}{file}{subq_claim_file_ending}", 'w') as f:
        json.dump(claims, f, indent=4)

merge_data()