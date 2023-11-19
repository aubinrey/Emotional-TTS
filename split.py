import json
import random

# Define the paths for the input and output JSON files
train_filelist_path = 'resources/filelists/friends/5Seasons_dataset.json'
output_train_path = 'resources/filelists/friends/train_split.json'
output_test_path = 'resources/filelists/friends/test_split.json'
output_valid_path = 'resources/filelists/friends/valid_split.json'

# Define the percentage split for train, test, and valid sets
train_split_percentage = 0.7
test_split_percentage = 0.15
valid_split_percentage = 0.15

# Function to split the data into train, test, and valid sets
def split_data(data, train_percent, test_percent, valid_percent):
    total_len = len(data)
    train_len = int(total_len * train_percent)
    test_len = int(total_len * test_percent)
    valid_len = total_len - train_len - test_len

    random.shuffle(data)

    train_data = data[:train_len]
    test_data = data[train_len:train_len + test_len]
    valid_data = data[train_len + test_len:]

    return train_data, test_data, valid_data

# Load the original JSON file with UTF-8 encoding
with open(train_filelist_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Split the data into train, test, and valid sets
train_set, test_set, valid_set = split_data(data, train_split_percentage, test_split_percentage, valid_split_percentage)

# Save the split sets to separate JSON files with UTF-8 encoding
with open(output_train_path, 'w', encoding='utf-8') as train_file:
    json.dump(train_set, train_file, indent=4)

with open(output_test_path, 'w', encoding='utf-8') as test_file:
    json.dump(test_set, test_file, indent=4)

with open(output_valid_path, 'w', encoding='utf-8') as valid_file:
    json.dump(valid_set, valid_file, indent=4)

print(f"Data split and saved to {output_train_path}, {output_test_path}, and {output_valid_path}.")
