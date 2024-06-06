import json
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch


def load_data(filepath):
    data = {'input_text': [], 'target_text': []}
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            input_text = f"Claim: {entry['claim']}\n\nPerson: {entry['person']}\n\nVenue: {entry['venue']}"
            target_text = ' '.join(entry['annotations'][0]['questions'])

            data['input_text'].append(input_text)
            data['target_text'].append(target_text)
    return Dataset.from_dict(data)

print("loading data")
dataset = load_data('ClaimDecomp/all.json')
dataset = dataset.train_test_split(test_size=0.1)

print("loading model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model = model.to(device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=1024, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=1024, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print("tokenizing data")
tokenized_datasets = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# Start training
print("training model")
trainer.train()

print("saving model")
model.save_pretrained('./my_bart_question_decomp')
tokenizer.save_pretrained('./my_bart_question_decomp')

print("evaluating model")
results = trainer.evaluate()
print(results)

input_text = "Claim: The president is not only criminal but also heavily overweight \n\nPerson: Kamala Harris\n\nVenue: Twitter"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
