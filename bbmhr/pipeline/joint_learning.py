import json
from xmlrpc.client import boolean

from torch import le
from prompting import read_source_data
from typing import Dict, List, Text

source_data_path = r"./data/ESConv_one_speaker_one_turn.json"
batch_data_path = r"./data/experiments/400M/test_response_b15_17.jsonl"
parlai_format_path = r"./data/experiments/no_reasoning/test_parlai_b15_17.txt"


def read_batch_data(batch_data_path: Text) -> List[Dict[str, str]]:
    batch_data_file = open(batch_data_path, "r", encoding="utf-8")
    batch_data = []
    for row in batch_data_file:
        batch_data.append(json.loads(row))

    return batch_data


def parlai_format_from_batch(
    batch_data: List[Dict[str, str]],
    source_data: List[Dict[str, str]],
    out_path: Text,
    with_annotation: boolean = True,
):
    output_file = open(out_path, "w+", encoding="utf-8", newline="")
    tmp = ""
    total_seeker_utterance_index = 0
    for dialog in source_data:
        start = 0
        # if the dialog starts from the supporter
        if dialog["conversation"][0]["speaker"] == "supporter":
            tmp += dialog["conversation"][0]["content"] + "\n".encode(
                "unicode_escape"
            ).decode("utf-8")
            start = 1
        for index in range(start, len(dialog["conversation"]) - 1, 2):
            if total_seeker_utterance_index >= len(batch_data):
                return

            text = tmp + dialog["conversation"][index]["content"]
            if with_annotation:
                annotation = batch_data[total_seeker_utterance_index]["response"].split(
                    "\n"
                )[0]
                text += " The seeker " + annotation
            label = dialog["conversation"][index + 1]["content"]
            tmp = ""
            total_seeker_utterance_index += 1
            if index >= len(dialog["conversation"]) - 3:
                output_file.write(
                    f"text:{text}"
                    + "\t"
                    + f"labels:{label}"
                    + "\t"
                    + "episode_done:True\n"
                )
                if index == len(dialog["conversation"]) - 3:
                    total_seeker_utterance_index += 1
            else:
                output_file.write(f"text:{text}" + "\t" + f"labels:{label}" + "\n")


def main():
    source_data = read_source_data(source_data_path)
    batch_data = read_batch_data(batch_data_path)
    parlai_format_from_batch(
        batch_data, source_data, parlai_format_path, with_annotation=False
    )


if __name__ == "__main__":
    main()
