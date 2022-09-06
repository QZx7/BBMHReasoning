import json
import argparse
from xmlrpc.client import boolean

from prompting import read_source_data
from typing import Dict, List, Text

source_data_path = r"./data/ESConv_one_speaker_one_turn.json"
batch_data_path = r"./data/experiments/3B/gpt/train_response_b0_14.jsonl"
parlai_format_path = r"./data/experiments/3B/gpt/train_parlai_b0_14.txt"


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
                # annotation = post_process_annotation(batch_data[total_seeker_utterance_index]["response"])
                # print(annotation)
                if annotation == "":
                    annotation = "<empty annotation>"
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
                if "<empty annotation>" not in text:
                    output_file.write(f"text:{text}" + "\t" + f"labels:{label}" + "\n")


def post_process_annotation(annotation: Text):
    if annotation[0] == " ":
        return annotation.split("\n")[0]
    else:
        if not annotation.startswith("the seeker"):
            try:
                return annotation[annotation.index(',') + 1:].split("\n")[0]
            except ValueError:
                return ""
        else:
            return annotation.split("\n")[0]


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data_path",
        type=str,
        default="./data/ESConv_one_speaker_one_turn.json",
        help="Original ESConv data path",
    )
    parser.add_argument(
        "--batch_data_path", type=str, required=True, help="Resposnes from PLMs."
    )
    parser.add_argument(
        "--parlai_format_path", type=str, required=True, help="Output parlai file path."
    )
    parser.add_argument(
        "--annotation", type=bool, default=True, help="Generate data with annotation."
    )
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = add_arguments()

    # load data
    source_data = read_source_data(args.source_data_path)
    batch_data = read_batch_data(args.batch_data_path)

    # generate parlai format file
    parlai_format_from_batch(
        batch_data, source_data, args.parlai_format_path, with_annotation=args.annotation
    )


if __name__ == "__main__":
    main()
