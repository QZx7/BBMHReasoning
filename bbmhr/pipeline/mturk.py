from typing import Text


import os
import csv


def generate_batch(task_name: Text):
    dest_file_name = "./eval/human_evaluation/" + task_name + ".csv"
    dest_file = open(dest_file_name, 'w+', encoding='utf-8', newline='')
    fieldnames = ['conversation_1', 'conversation_2']
    writer = csv.DictWriter(dest_file, fieldnames=fieldnames)

    for (dirpath, dirnames, filenames) in os.walk("./eval/human_evaluation/"):
        print(filenames)
    # human_academic_text = open("./eval/human_evaluation/human_academic.txt", 'r', encoding='utf-8').read()
    # human_academic_text = open("./eval/human_evaluation/human_academic.txt", 'r', encoding='utf-8').read()
    # human_academic_text = open("./eval/human_evaluation/human_academic.txt", 'r', encoding='utf-8').read()
    # human_academic_text = open("./eval/human_evaluation/human_academic.txt", 'r', encoding='utf-8').read()
    # human_academic_text = open("./eval/human_evaluation/human_academic.txt", 'r', encoding='utf-8').read()
    text_dict = {}
    for filename in filenames:
        if ".xlsx" in filename or ".csv" in filename:
            continue
        print(filename)
        text = open("./eval/human_evaluation/" + filename, 'r', encoding='utf-8').read()
        
        # process
        if "human_" not in filename and "bb_" not in filename:
            text = text.replace("seeker","<strong>seeker</strong>")
            text = text.replace("supporter",'<strong style="background-color: aqua">supporter 2</strong>')
            text = text.replace("\n","<br>\n")
        text_dict[filename] = text
    
    writer.writeheader()
    writer.writerow({"conversation_1": text_dict["human_academic.txt"], "conversation_2": text_dict[task_name + "_academic.txt"]})
    writer.writerow({"conversation_1": text_dict["human_friend.txt"], "conversation_2": text_dict[task_name + "_friend.txt"]})
    writer.writerow({"conversation_1": text_dict["human_job.txt"], "conversation_2": text_dict[task_name + "_job.txt"]})
    writer.writerow({"conversation_1": text_dict["human_ongoing.txt"], "conversation_2": text_dict[task_name + "_ongoing.txt"]})
    writer.writerow({"conversation_1": text_dict["human_partner.txt"], "conversation_2": text_dict[task_name + "_partner.txt"]})


    # writer.writerow({"conversation_1": text_dict["human_academic"], "conversation_2": text_dict["bbmh_academic"]})
    # writer.writerow({"conversation_1": text_dict["human_friend"], "conversation_2": text_dict["bbmh_friend"]})
    # writer.writerow({"conversation_1": text_dict["human_job"], "conversation_2": text_dict["bbmh_job"]})
    # writer.writerow({"conversation_1": text_dict["human_ongoing"], "conversation_2": text_dict["bbmh_ongoing"]})
    # writer.writerow({"conversation_1": text_dict["human_partner"], "conversation_2": text_dict["bbmh_partner"]})

    # writer.writerow({"conversation_1": text_dict["human_academic"], "conversation_2": text_dict["gpt_1_academic"]})
    # writer.writerow({"conversation_1": text_dict["human_friend"], "conversation_2": text_dict["gpt_1_friend"]})
    # writer.writerow({"conversation_1": text_dict["human_job"], "conversation_2": text_dict["gpt_1_job"]})
    # writer.writerow({"conversation_1": text_dict["human_ongoing"], "conversation_2": text_dict["gpt_1_ongoing"]})
    # writer.writerow({"conversation_1": text_dict["human_partner"], "conversation_2": text_dict["gpt_1_partner"]})

    # writer.writerow({"conversation_1": text_dict["human_academic"], "conversation_2": text_dict["gpt_2_academic"]})
    # writer.writerow({"conversation_1": text_dict["human_friend"], "conversation_2": text_dict["gpt_2_friend"]})
    # writer.writerow({"conversation_1": text_dict["human_job"], "conversation_2": text_dict["gpt_2_job"]})
    # writer.writerow({"conversation_1": text_dict["human_ongoing"], "conversation_2": text_dict["gpt_2_ongoing"]})
    # writer.writerow({"conversation_1": text_dict["human_partner"], "conversation_2": text_dict["gpt_2_partner"]})

    # writer.writerow({"conversation_1": text_dict["human_academic"], "conversation_2": text_dict["ada_academic"]})
    # writer.writerow({"conversation_1": text_dict["human_friend"], "conversation_2": text_dict["ada_friend"]})
    # writer.writerow({"conversation_1": text_dict["human_job"], "conversation_2": text_dict["ada_job"]})
    # writer.writerow({"conversation_1": text_dict["human_ongoing"], "conversation_2": text_dict["ada_ongoing"]})
    # writer.writerow({"conversation_1": text_dict["human_partner"], "conversation_2": text_dict["ada_partner"]})

    # writer.writerow({"conversation_1": text_dict["human_academic"], "conversation_2": text_dict["davinci_academic"]})
    # writer.writerow({"conversation_1": text_dict["human_friend"], "conversation_2": text_dict["davinci_friend"]})
    # writer.writerow({"conversation_1": text_dict["human_job"], "conversation_2": text_dict["davinci_job"]})
    # writer.writerow({"conversation_1": text_dict["human_ongoing"], "conversation_2": text_dict["davinci_ongoing"]})
    # writer.writerow({"conversation_1": text_dict["human_partner"], "conversation_2": text_dict["davinci_partner"]})

    print(text_dict)




def main():
    generate_batch("bb")
    generate_batch("gpt_1")
    generate_batch("gpt_2")
    generate_batch("ada")
    generate_batch("davinci")
    generate_batch("bbmh")


if __name__ == "__main__":
    main()
