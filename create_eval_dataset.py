import json
import random

def pick_random_samples(dataset, num_samples):
    return random.sample(dataset, min(num_samples, len(dataset)))

def main():
    # Friends dataset
    with open('./resources/filelists/friends/5Seasons_dataset.json', 'r', encoding='utf-8') as friends_file:
        friends_dataset = json.load(friends_file)

    # ESD dataset
    with open('./resources/filelists/ESD/dataset.json', 'r', encoding='utf-8') as esd_file:
        esd_dataset = json.load(esd_file)

    # 90 random samples (15 for each of the 6 character) from Friends dataset
    friends_samples = []
    for char_id in range(6):
        char_samples = [sample for sample in friends_dataset if sample['character-id'] == char_id]
        friends_samples.extend(pick_random_samples(char_samples, 15))

    # 100 random samples (2 for each of the 10 character and each of the 5 emotions) from ESD dataset
    esd_samples = []
    for char_id in range(10):
        for emotion in set(sample['emotion'] for sample in esd_dataset):
            char_emotion_samples = [sample for sample in esd_dataset if sample['character-id'] == char_id and sample['emotion'] == emotion]
            esd_samples.extend(pick_random_samples(char_emotion_samples, 2))

    evaluation_dataset = {
        "Friends": friends_samples,
        "ESD": esd_samples
    }

    # Save the evaluation dataset 
    with open('evaluation_dataset.json', 'w', encoding='utf-8') as eval_file:
        json.dump(evaluation_dataset, eval_file, indent=2)

if __name__ == "__main__":
    main()
