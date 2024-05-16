import json
from sklearn.preprocessing import LabelEncoder
import csv


with open("NumTemp-E9C0/output/bm25_top_100_train") as f:
    train_data = json.load(f)
with open('NumTemp-E9C0/output/bm25_top_100_test') as f:
    test_data = json.load(f)

LE = LabelEncoder()
train_labels = [fact["label"] for fact in train_data]
test_labels = [fact["label"] for fact in test_data]
train_labels_final = LE.fit_transform(train_labels)
test_labels_final = LE.transform(test_labels)

test_labels_final = test_labels_final.tolist()

with open('ground_truth.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the list as a single row in the CSV
    writer.writerow(test_labels_final)
