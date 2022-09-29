import csv
import json
import random
from typing import Dict, Text
from bbmhr.pipeline.prompting import assembly_prompt, read_source_data


def process_response(response: Text, task: Text) -> Text:
    if task == "gpt_1":
        if response.startswith(' ') or response.startswith("the seeker"):
            return response.split('\n')[0]
        else:
            return response.split(',')[1].split('\n')[0]
    elif task == "gpt_2":
        return response.split('\n')[0]

def pick_up_sample_conversations(sample_file_path, sample_number: int = 100):
    current_sample_file = open(sample_file_path, 'r', encoding='utf-8', newline='')
    current_data = {}
    for line in current_sample_file.readlines():
        print(line)
        line_data = json.loads(line)
        current_data[line_data["index"]] = line_data["content"]
    current_number = len(current_data.items())
    current_sample_file.close()

    current_sample_file = open(sample_file_path, 'a', encoding='utf-8')
    print(f"Detected {current_number} samples, will continue annotation.")

    # Create dialogue generator
    print("Loading generator...")
    prompt_path = './bbmhr/prompt_templates/nl_gpt_1.txt'
    template_file = open(prompt_path, "r", encoding="utf-8")
    fixed_prompt = template_file.read()
    source_data_path = './data/ESConv_one_speaker_one_turn.json'
    source_data = read_source_data(source_data_path)
    
    dialogue_list = []
    prompt_generator = assembly_prompt(fixed_prompt, source_data=source_data)
    for _ in range(7500):
        dialogue_list.append(next(prompt_generator).split("Conversation:")[-1])

    # Read from responses
    print("Loading responses...")
    gpt_1_response_path = './data/experiments/bbmhr/3B/gpt_1/train_response_b0_14.jsonl'
    gpt_1_response_file = open(gpt_1_response_path, 'r', encoding='utf-8')
    gpt_1_response_list = []
    for line in gpt_1_response_file.readlines():
        gpt_1_response_list.append(json.loads(line)["response"])

    gpt_2_response_path = './data/experiments/bbmhr/3B/new_gpt_2/train_response_b0_14.jsonl'
    gpt_2_response_file = open(gpt_2_response_path, 'r', encoding='utf-8')
    gpt_2_response_list = []
    for line in gpt_2_response_file.readlines():
        gpt_2_response_list.append(json.loads(line)["response"])

    ada_response_path = './data/experiments/bbmhr/3B/ada/train_response_b0_14.jsonl'
    ada_response_file = open(ada_response_path, 'r', encoding='utf-8')
    ada_response_list = []
    for line in ada_response_file.readlines():
        ada_response_list.append(json.loads(line)["response"])

    davinci_response_path = './data/experiments/bbmhr/3B/davinci/train_response_b0_14.jsonl'
    davinci_response_file = open(davinci_response_path, 'r', encoding='utf-8')
    davinci_response_list = []
    for line in davinci_response_file.readlines():
        davinci_response_list.append(json.loads(line)["response"])

    # print(len(dialogue_list), dialogue_list[0:5])
    # print(len(gpt_1_response_list), gpt_1_response_list[0:5])
    # print(len(gpt_2_response_list), gpt_2_response_list[0:5])
    # print(len(ada_response_list), ada_response_list[0:5])
    # print(len(davinci_response_list), davinci_response_list[0:5])
    # print(len(response_list), response_list[0:5])

    # do annotate if there aren't enough samples
    while current_number < sample_number:
        num = random.randint(0, 7500)
        if num in current_data:
            continue
        print("******** Dialog *************")
        print(dialogue_list[num])
        print("********* GPT-1 ************")
        print("1: " + gpt_1_response_list[num] + "\n")
        print("********* GPT-2 ************")
        print("2: " + gpt_2_response_list[num] + "\n")
        print("********* Ada ************")
        print("3: " + ada_response_list[num] + "\n")
        print("********* Davinci ************")
        print("4: " + davinci_response_list[num] + "\n")
        print("********* Human ************")
        command = input("Annotate?\n")
        if command == "x":
            continue
        else:
            human_label = input("Enter your label. \n")
            new_data = {
                "index": num,
                "content": {
                    "dialog": dialogue_list[num],
                    "gpt_1": process_response(gpt_1_response_list[num], "gpt_1"),
                    "gpt_2": process_response(gpt_2_response_list[num], "gpt_2"),
                    "ada": ada_response_list[num],
                    "davinci": davinci_response_list[num],
                    "human": human_label
                }
            }
            current_number += 1
            current_sample_file.write(json.dumps(new_data) + "\n")


def prepare_turk_data(reasoning_data_path: Text, csv_data_path: Text):
    reasoning_data_file = open(reasoning_data_path, 'r', encoding='utf-8')
    csv_data_file = open(csv_data_path, 'w+', newline='', encoding='utf-8')

    fieldnames = ["dialog", "gpt_1", "gpt_2", "ada", "davinci"]
    csv_writer = csv.DictWriter(csv_data_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for line in reasoning_data_file.readlines():
        line_data = json.loads(line.strip())
        sample = line_data["content"]
        del sample['human']
        print(sample)
        sample['dialog'] = sample['dialog'].replace('\n', '\n<br>')
        sample['dialog'] = sample['dialog'].replace("seeker:", "<strong>seeker:</strong>")
        sample['dialog'] = sample['dialog'].replace("supporter:", "<strong>supporter:</strong>")
        sample['gpt_2'] = "the seeker " + sample['gpt_2']
        sample['ada'] = "the seeker " + sample['ada']
        sample['davinci'] = "the seeker " + sample['davinci']
        csv_writer.writerow(sample)


if __name__ == "__main__":
    # pick_up_sample_conversations("./eval/reasoning_evaluation/samples.jsonl")
    prepare_turk_data(r"./eval/reasoning_evaluation/samples.jsonl", r"./eval/reasoning_evaluation/samples_no_human.csv")
