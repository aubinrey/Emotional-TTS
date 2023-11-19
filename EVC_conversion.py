import json

def parse_text_file(file_path):
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            spk_id, text, emotion = parts[0].split('_')[0], parts[1], parts[2]
            wav_path = f"{file_path.split('/')[0]}/{spk_id}/{parts[0].split('_')[1]}.wav"
            entry = {"path-wav": wav_path, "text": text, "character-id": int(spk_id)-11, "emotion": emotion}
            data.append(entry)

    return data

file_path = 'Emotion Speech Dataset/dataset.txt'  # Replace with the actual path to your text file
json_output_path = 'Emotion Speech Dataset/dataset.json'   # Replace with the desired output JSON file path

parsed_data = parse_text_file(file_path)

with open(json_output_path, 'w') as json_file:
    json.dump(parsed_data, json_file, indent=2)

print(f"JSON file created at: {json_output_path}")
