import csv
import json
import random
import numpy as np
from typing import Dict, List, Text
from collections import Counter
from bbmhr.pipeline.prompting import assembly_prompt, read_source_data


def process_response(response: Text, task: Text) -> Text:
    if task == "gpt_1":
        if response.startswith(' ') or response.startswith("the seeker"):
            return response.split('\n')[0]
        else:
            return response.split(',')[1].split('\n')[0]
    elif task == "gpt_2" or task == "distilgpt2":
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


def add_distil_data(reasoning_data_path: Text, distil_data_path: Text, distil_reasoning_path: Text):
    distil_file = open(distil_data_path, 'r', encoding='utf-8')
    reasoning_file = open(reasoning_data_path, 'r', encoding='utf-8')
    distil_reasoning_file = open(distil_reasoning_path, 'w', encoding='utf-8')

    distil_lines = distil_file.readlines()
    reasoning_lines = reasoning_file.readlines()

    for index in range(len(reasoning_lines)):
        reasoning_data = json.loads(reasoning_lines[index].strip())
        reasoning_data["content"]["distilgpt2"] = process_response(json.loads(distil_lines[index].strip())["response"], "distilgpt2")
        distil_reasoning_file.write(json.dumps(reasoning_data) + "\n")


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


def approve_reject(results_path: Text, bad_worker: Text):
    results_file = open(results_path, 'r', newline='', encoding='utf-8')
    reader = csv.reader(results_file)
    worker = {}
    fields = {}

    raw_results = []

    bad_workers = []
    if bad_worker:
        bad_worker_file = open(bad_worker, 'r', encoding='utf-8')
        for l in bad_worker_file.readlines():
            bad_workers.append(l.strip())
    
    # print(bad_workers)

    header = next(reader)
    for index in range(len(header)):
        fields[index] = header[index]
    # print(fields)

    for line in reader:
        if line[15] in bad_workers or line[21] != "":
            continue

        tmp = []
        start = 32
        while start < 68:
            for item_number in range(start, start + 3):
                if line[item_number] == "true":
                    tmp.append((item_number - 32) % 3)
            start += 3
        raw_results.append(tmp.copy())

        tmp.extend(line[68: 83])
        tmp.append(line[23])

        if line[15] in worker:
            worker[line[15]].append(tmp)
        else:
            worker[line[15]] = [tmp]

    return raw_results
    # for key, values in worker.items():
    #     print(key)
    #     for value in values:
    #         print(value)


def calculate_rates(raw: List[List[int]]):
    numbers = {
        "davinci": [],
        "gpt_1": [],
        "gpt_2": [],
        "ada": []
    }


    for item in raw:
        davinci = [item[0], item[4], item[8]]
        gpt_1 = [item[1], item[5], item[9]]
        gpt_2 = [item[2], item[6], item[10]]
        ada = [item[3], item[7], item[11]]

        numbers["davinci"].append(davinci)
        numbers["gpt_1"].append(gpt_1)
        numbers["gpt_2"].append(gpt_2)
        numbers["ada"].append(ada)
    
    for key, value in numbers.items():
        print(f"calculating {key}")

        print(f"emotion: ")
        choices_dict = Counter(np.array(value)[:, 0].tolist())
        print(choices_dict[0] / (choices_dict[0] + choices_dict[1] + choices_dict[2]))

        print(f"reason: ")
        choices_dict = Counter(np.array(value)[:, 1].tolist())
        print(choices_dict[0] / (choices_dict[0] + choices_dict[1] + choices_dict[2]))

        print(f"suggestion: ")
        choices_dict = Counter(np.array(value)[:, 2].tolist())
        print(choices_dict[0] / (choices_dict[0] + choices_dict[1] + choices_dict[2]))

        all_dict = Counter(np.array(value).flatten().tolist())
        print(all_dict[0] / (all_dict[0] + all_dict[1] + all_dict[2]))


if __name__ == "__main__":
    # pick_up_sample_conversations("./eval/reasoning_evaluation/samples.jsonl")
    # prepare_turk_data(r"./eval/reasoning_evaluation/samples.jsonl", r"./eval/reasoning_evaluation/samples_no_human.csv")
    # raw_results = approve_reject(
    #     r"./eval/Turkresults/reasoning/reasoning.csv",
    #     r"./eval/Turkresults/workers/reasoning/bad worker"
    #     )

    # calculate_rates(raw_results)
    add_distil_data(
        r"./eval/reasoning_evaluation/samples.jsonl",
        r"./data/NL_responseb_0.jsonl",
        r"./eval/reasoning_evaluation/new_samples.jsonl"
    )
