import pandas as pd
import csv
import random

train = pd.read_csv('data/train.tsv', '\t')
not_10 = pd.read_csv('not_dep_summaries/10_word.tsv', '\t')
not_20 = pd.read_csv('not_dep_summaries/20_word.tsv', '\t')
not_30 = pd.read_csv('not_dep_summaries/20_word.tsv', '\t')
not_40 = pd.read_csv('not_dep_summaries/20_word.tsv', '\t')
not_50 = pd.read_csv('not_dep_summaries/50_word.tsv', '\t')
#not_100 = pd.read_csv('not_dep_summaries/100_word.tsv', '\t')
sev_10 = pd.read_csv('severe_summarizes/10_word.tsv', '\t')
#sev_15 = pd.read_csv('severe_summarizes/15_word.tsv', '\t')
sev_20 = pd.read_csv('severe_summarizes/20_word.tsv', '\t')
sev_25 = pd.read_csv('severe_summarizes/25_word.tsv', '\t')
sev_30 = pd.read_csv('severe_summarizes/30_word.tsv', '\t')
#sev_35 = pd.read_csv('severe_summarizes/35_word.tsv', '\t')
sev_40 = pd.read_csv('severe_summarizes/40_word.tsv', '\t')
#sev_45 = pd.read_csv('severe_summarizes/45_word.tsv', '\t')
sev_50 = pd.read_csv('severe_summarizes/50_word.tsv', '\t')

original_size = train.Label.value_counts()

for index, row in not_40.sample(5).iterrows():
    print(index, row)

print(original_size['moderate'], original_size['moderate'] - original_size['moderate'])
print(original_size['severe'], original_size['moderate'] - original_size['severe'])
print(original_size['not depression'], original_size['moderate'] - original_size['not depression'])

severe_per = round((original_size['moderate'] - original_size['severe']) / 6) # 6 Because we have 6 summarized datasets
not_per = round((original_size['moderate'] - original_size['not depression']) / 5) # 6 Because we have 6 summarized datasets

print("balancing severe per class : ", severe_per)
print("balancing not depression per class : ", not_per)

count = 0

random.seed(42)

with open('50_balanced_dataset.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['PID', 'Text_data', 'Label'])

    for index, row in train.iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])

    # Severe depression balancing
    for index, row in sev_10.sample(severe_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in sev_20.sample(severe_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in sev_25.sample(severe_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in sev_30.sample(severe_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in sev_40.sample(severe_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in sev_50.sample(severe_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])

    # Not depression balancing
    for index, row in not_10.sample(not_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in not_20.sample(not_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in not_30.sample(not_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in not_40.sample(not_per).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])
    for index, row in not_50.sample(not_per - 2).iterrows():
        tsv_writer.writerow([row['PID'], row['Text_data'], row['Label']])


new = pd.read_csv('50_balanced_dataset.tsv', '\t')
print("New:", new.Label.value_counts())
