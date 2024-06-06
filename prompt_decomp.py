from openai import OpenAI
import json
from datasets import Dataset

init_message = """
We need to determine the truth of political claims. 
The first step towards this is to break down the claims into many smaller, independently verifiable sub-questions (usually about 2-4).
The sub-questions should be yes/no questions. If the answer to all questions is yes, then the claim is true. 
Return the subsquestions for each claim in a valid json as the example below.
If it is a short claim that REALLY cannot be broken down to subquestions, it is fine to only return 1 question.
Tip: use words like "any" and "all" often.

Here is an example:
Claim 1: Do private insurance companies spend between 12 and 18 percent on administration costs? Is the cost of administering the Medicare program 2%? Can 500 billion dollars be saved in administration costs? Does lower administrative costs result in lower overall costs? 
Claim 2: Unemployment now pays $24/hour, even if your wages were lower. Why don’t ‘essential’ people forced to still work get $24, too?
Subquestions: {
    Claim 1: ["Do private insurance companies spend between 12 and 18 percent on administration costs?", "Is the cost of administering the Medicare program 2%?", "Can 500 billion dollars be saved in administration costs?", "Does lower administrative costs result in lower overall costs?"],
    Claim 2: ["Can unemployment recipients receive up to $24 per hour?", "Will all unemployment recipients receive $24 per hour in benefits?", "Are employed people receiving financial boosts as well?"]
}

The claims are as follows:

"""

file = "train"

data_dir = "NumTemp-E9C0/data/raw_data/"
data_file_ending = "_claims_quantemp.json"

write_dir = "NumTemp-E9C0/data/decomp_data/"
write_file_ending = "_claims_quantemp_decomp.json"

claim_with_subq_dir = "NumTemp-E9C0/data/claim_with_subq_data/"
claim_with_subq_file_ending = "_claim_with_subq.json"


def load_data(filepath):
    data = {'input_text': [], 'target_text': []}
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            input_text = f"{entry['claim']}" # \n\nPerson: {entry['person']}\n\nVenue: {entry['venue']}"
            target_text = ' '.join(entry['annotations'][0]['questions'])

            data['input_text'].append(input_text)
            data['target_text'].append(target_text)
    return Dataset.from_dict(data)

def load_raw_data(filepath):
    with open(filepath, 'r') as f:
        data = json.loads(f.read())
        return [data['claim'] for data in data]

def decompose_claims():

    data = load_raw_data(f"{data_dir}{file}{data_file_ending}")
    # randomly shuffle the data with a seed, dont break it down into train/test
    # data = data.shuffle(seed=42)
    client = OpenAI()

    n = 5
    total_entries = len(data)

    begin = 0

    # Loop through the data in chunks of size n
    for start_idx in range(begin, total_entries, n):
        print(start_idx)
        end_idx = min(start_idx + n, total_entries)  # Calculate the end index for slicing
        chunk = data[start_idx:end_idx]  # Get the current chunk of data

        # Create the message for the current chunk
        message = init_message
        for i, entry in enumerate(chunk):
            message = f"{message}Claim {i + 1}: {entry} \n"

        json_data = parse_json_from_gpt(client, message)
        list_data = list(json_data.values())

        # Write the JSON data to a file
        with open(f"{write_dir}{file}{write_file_ending}", 'a') as f:
            for sublist in list_data:
                print(sublist)
                f.write(json.dumps(sublist) + '\n')


def ask_gpt_api(client, message):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": message}]
    )
    return completion.choices[0].message.content

def parse_json_from_gpt(client, message):
    counter = 0
    while True:
        reply = ask_gpt_api(client, message)
        try:
            return json.loads(reply)
        except (json.JSONDecodeError, StopIteration):
            # If JSON is invalid or generator is exhausted, continue
            counter += 1
            print(f"Error in parsing JSON. Attempt {counter}")
            print(reply)

            if counter == 10:
                print(message)
                return """Subquestions: {
                            Claim 1: ["Error"],
                            Claim 2: ["Error"],
                            Claim 3: ["Error"],
                            Claim 4: ["Error"],
                            Claim 5: ["Error"]
                        }"""
            pass

def fix_broken():
    client = OpenAI()
    with open(f"{claim_with_subq_dir}{file}{claim_with_subq_file_ending}", 'r') as f:
        data = json.loads(f.read())

    for entry in data:
        subqs = entry['subquestions']
        if "FIX THIS" in subqs[0]:
            message = f"{init_message}Claim 1: {entry['claim']} \n"
            json_data = parse_json_from_gpt(client, message)
            new_subq = list(json_data.values())[0]
            entry['subquestions'] = new_subq
            print(f"Fixed {entry['claim']} with {new_subq}")

    with open(f"{claim_with_subq_dir}{file}{claim_with_subq_file_ending}", 'w') as f:
        json.dump(data, f, indent=4)

fix_broken()


# "And the number one killer of teens in the United States is driving while using cell phones."
# "54 senators said, let's do background checks, let's get rid of assault weapons, and with 54 senators it failed because of the filibuster."
# Wearing face mask for 20 minutes will contaminate it with bacteria.
# In 2013, only 76% of total votes were transmitted or counted. In 2019, 961 [vote counting] machines became defective and 1,000 SD cards had to be replaced.
