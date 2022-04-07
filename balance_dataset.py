import pandas as pd
import csv
import random

train = pd.read_csv('data/train.tsv', '\t')
not_10 = pd.read_csv('not_dep_summaries/10_word.tsv', '\t')
not_20 = pd.read_csv('not_dep_summaries/20_word.tsv', '\t')
not_35 = pd.read_csv('not_dep_summaries/35_word.tsv', '\t')
not_50 = pd.read_csv('not_dep_summaries/50_word.tsv', '\t')
not_100 = pd.read_csv('not_dep_summaries/100_word.tsv', '\t')
sev_10 = pd.read_csv('severe_summarizes/10_word.tsv', '\t')
sev_15 = pd.read_csv('severe_summarizes/15_word.tsv', '\t')
sev_20 = pd.read_csv('severe_summarizes/20_word.tsv', '\t')
sev_25 = pd.read_csv('severe_summarizes/25_word.tsv', '\t')
sev_30 = pd.read_csv('severe_summarizes/30_word.tsv', '\t')
sev_35 = pd.read_csv('severe_summarizes/35_word.tsv', '\t')
sev_40 = pd.read_csv('severe_summarizes/40_word.tsv', '\t')
sev_45 = pd.read_csv('severe_summarizes/45_word.tsv', '\t')
sev_50 = pd.read_csv('severe_summarizes/50_word.tsv', '\t')

original_size = train.Label.value_counts()


print(original_size['moderate'], original_size['moderate'] / original_size['moderate'])
print(original_size['severe'], original_size['moderate'] / original_size['severe'])
print(original_size['not depression'], original_size['moderate'] / original_size['not depression'])
count = 0

random.seed(42)

#pids = random.sample(train['PID'].tolist(), int(len(train['PID']) * 0.25))
#print(len(pids))

with open('75_balanced_dataset.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['PID', 'Text_data', 'Label'])

    for index, row in train.iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
        count += 1

    severe = original_size['severe']
    not_dep = original_size['not depression']
    compare = original_size['moderate'] // 2 + original_size['moderate'] // 4
    print(severe, not_dep)

    for index, row in sev_10.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_15.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_20.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_25.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_30.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_35.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_40.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    for index, row in sev_45.iterrows():
        if severe < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            severe += 1
    
    for index, row in not_10.iterrows():
        if not_dep < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            not_dep += 1
    for index, row in not_20.iterrows():
        if not_dep < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            not_dep += 1
    for index, row in not_35.iterrows():
        if not_dep < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            not_dep += 1
    for index, row in not_50.iterrows():
        if not_dep < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            not_dep += 1
    for index, row in not_100.iterrows():
        if not_dep < compare:
            tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
            not_dep += 1


print(severe, not_dep)
new = pd.read_csv('75_balanced_dataset.tsv', '\t')
print("New:", new.Label.value_counts())
