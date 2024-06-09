from openai import OpenAI
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import torch
import json
import re


def fix_line(output):
    tries = 10
    for i in range(tries):
        try:
            line = ask_gpt(output)
            return line
        except:
            continue

    return f"FIX THIS OUTPUT: {output}\n"


def ask_gpt(broken_json):
    init_message = """This JSON is formatted incorrectly. 
    You need to simply format it correctly and send it back. Usually the problem is with double quotes that has to replaced with single quotes.
    SEND NOTHING ELSE BACK THAN THE FIXED !VALID! JSON. 
    
    For example:
    ["Did Obama say, "If I don't have this done in 3 years, then there's going to be a one-term proposition?"]
    should become
    ["Did Obama say, 'If I don't have this done in 3 years, then there's going to be a one-term proposition?'"]
    
    Here is broken JSON:

    """
    print("Asking GPT-3.5 for help...")
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": init_message + broken_json}]
    )
    reply = completion.choices[0].message.content
    print("GPT-3.5 replied: " + reply)
    return json.loads(reply)


def fix_claims():
    with open("NumTemp-E9C0/data/decomp_data_ft/val_claims_quantemp_decomp_ft.json", 'r') as f:
        data = f.readlines()


    new_data = []
    for i, line in enumerate(data):
        if i % 100 == 0:
            print(f"Processing claim {i + 1}/{len(data)}", end='\r', flush=True)
        if "FIX THIS" in line:
            output = line.split("OUTPUT: ")[1]
            print(claim)
            line = fix_line(output)
        else:
            line = json.loads(line)

        new_data.append(line)

    with open("NumTemp-E9C0/data/decomp_data_ft/val_claims_quantemp_decomp_ft_fixed.json", 'w') as f:
        for line in new_data:
            f.write(json.dumps(line) + '\n')

# fix_claims()
# exit(0)

model_dir = "QDecomp_model_new"

model = BartForConditionalGeneration.from_pretrained("QDecomp_model_new")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

file = "val"

data_dir = "NumTemp-E9C0/data/raw_data/"
data_file_ending = "_claims_quantemp.json"
data_file = f"{data_dir}{file}{data_file_ending}"

write_dir = "NumTemp-E9C0/data/decomp_data_ft/"
write_file_ending = "_claims_quantemp_decomp_ft.json"
write_file = f"{write_dir}{file}{write_file_ending}"

with open(data_file, 'r') as f:
    data = json.loads(f.read())

new_data = []
for i, claim in enumerate(data):
    print(f"Processing claim {i + 1}/{len(data)}", end='\r', flush=True)
    input_text = f"{claim['claim']}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length").to(device)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens = 256)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    try:
        line = json.loads(output)
    except:
        line = fix_line(output)

    new_data.append(line)

with open(write_file, 'w') as f:
    for line in new_data:
        f.write(json.dumps(line) + '\n')



