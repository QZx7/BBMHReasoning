from typing import Dict, List


def parlai_format_from_batch(batch_data: List[Dict[str, str]], source_data: List[Dict[str, str]]):
    tmp = ""
    for dialog in source_data:
        start = 0
        # if the dialog starts from the supporter
        if dialog["conversation"][0]["speaker"] == "supporter":
            tmp += dialog["conversation"][0]["content"] + "\n".encode("unicode_escape").decode("utf-8")
            start = 1
