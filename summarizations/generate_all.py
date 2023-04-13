from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import csv
import torch

device = torch.device('cuda:0')
print("Loading model")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("t5-base")

print("Loading training data")
train = pd.read_csv('../data/train.tsv', '\t')
 
print(pd.Categorical(train['Label']))

divisor = 2

for i in range(10, 50, 10):
    for label in ['severe', 'moderate', 'not depression']:
        count = 0
        print(f"Getting indexes for {label} for {i}")
        pids = train['PID'][train['Label'] == label]
        articles = train['Text_data'][train['Label'] == label]
        labels = train['Label'][train['Label'] == label]
        with open(f"divide_{divisor}_word_{label}.tsv", 'wt') as out_file:
            try:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(['PID', 'Text_data', 'Label'])

                for article in articles:
                    inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=512, truncation=True).to(device)

                    outputs = model.generate(
                        inputs["input_ids"], max_length=(len(inputs["input_ids"][0])//divisor) + 2, min_length=(len(inputs["input_ids"][0])//divisor) - 5, num_beams=4, early_stopping=True
                    ).to(device)

                    tsv_writer.writerow([pids.iloc[count], tokenizer.decode(outputs[0])[6:], labels.iloc[count]])
                    count += 1
                    
            except Exception as e:
                print("ERROR:",i, e)
