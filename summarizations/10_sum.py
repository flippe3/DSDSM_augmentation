from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import csv
import torch

device = torch.device('cuda:0')

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)

train = pd.read_csv('../data/train.tsv', '\t')
 
pids = train['PID'][train['Label'] == 'not depression']
articles = train['Text_data'][train['Label'] == 'not depression']
labels = train['Label'][train['Label'] == 'not depression']

tokenizer = AutoTokenizer.from_pretrained("t5-base")
count = 0

with open('10_word.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['PID', 'Text_data', 'Label'])

    for article in articles:
        inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=512, truncation=True).to(device)

        outputs = model.generate(
            inputs["input_ids"], max_length=12, min_length=8, length_penalty=2.0, num_beams=4, early_stopping=True
        ).to(device)

        tsv_writer.writerow([pids.iloc[count], tokenizer.decode(outputs[0])[6:], labels.iloc[count]])
        count += 1
        print("Count:", count, " of ", len(articles), pids.iloc[count])
