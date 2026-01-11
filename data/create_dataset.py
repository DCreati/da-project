from datasets import load_dataset
import pandas as pd

dataset = load_dataset("chengxuphd/liar2")

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

def calculate_total_statements(row):
    row["total_statements"] = (
        row['true_counts'] +
        row['mostly_true_counts'] +
        row['half_true_counts'] +
        row['mostly_false_counts'] +
        row['false_counts'] +
        row['pants_on_fire_counts']
    )
    if row["total_statements"] > 0:
        row['credibility_score'] = (row['mostly_true_counts'] + row['true_counts']) / row['total_statements']
    else:
        row['credibility_score'] = 0
    return row

train = train.map(calculate_total_statements)
validation = validation.map(calculate_total_statements)
test = test.map(calculate_total_statements)

train_df = pd.DataFrame(train)
validation_df = pd.DataFrame(validation)
test_df = pd.DataFrame(test)

combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

combined_df.to_csv("liar_credibility_score.csv", index=False)


#dataset['total_statements'] = liar_dataset[['true_counts', 'mostly_true_counts', 'half_true_counts', 'mostly_false_counts', 'false_counts', 'pants_on_fire_counts']].sum(axis=1)

#liar_dataset['Credibility Score'] = liar_dataset.apply(lambda row: (row['mostly_true_counts'] + row['true_counts']) / row['total_statements'] if row['total_statements'] > 0 else 0, axis=1)


# Remove unnecessary columns

#liar_dataset = liar_dataset.drop('total_statements', axis=1)

#liar_dataset.head()