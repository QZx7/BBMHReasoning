import argparse
from cgi import print_arguments
import json
import random
import logging
from statistics import mode
import openai
import os
from typing import Any, Dict, List, Optional, Text, TextIO

from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

GPT_3_log_path = os.path.join("./log", date.today().strftime("%Y%m%d") + ".log")
handler = logging.FileHandler(GPT_3_log_path, mode="a+", encoding="utf-8")
handler.setFormatter(logging.Formatter(Log_Format))
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_3_TASK_COMPLETION = "completion"
GPT_3_TASK_CLASSIFICATION = "classification"
GPT_3_TASK_CONVERSATION = "conversation"

template_path = r"./bbmhr/prompt_templates/nl_utt_level.txt"
source_data_path = r"./data/ESConv_one_speaker_one_turn.json"
test_data_path = r"./data/ESConv_test_data.json"
response_path = r"./data/NL_response"
seeker_utterances_only = r"./data/seeker_only"


def read_prompt(prompt_path: Text) -> Text:
    """Read a prompt template from text file.

    Args:
        prompt_path (Text): Path to the prompt template file.

    Returns:
        Text: Prompt template as plain text.
    """
    template_file = open(prompt_path, "r", encoding="utf-8")
    return template_file.read()


def get_gpt_result(
    task: Text,
    gpt_prompt: Optional[Text] = "",
    query: Optional[Text] = "",
    stop_words: Optional[List[Text]] = [],
) -> Dict:
    """Get response from the gpt by prompt.

    Args:
        task (Text): task to finsih, choose from "completion", "classification" and "conversation".
        gpt_prompt (Optional[Text], optional): The prompt text. Defaults to "".
        query (Optional[Text], optional): classification task only. Defaults to "".
        stope_words (Optional[List[Text]], optional): stops to break the generation and prepare for next quest.

    Returns:
        Dict: the result dict
    """
    logger.debug("Now requesting GPT-3")
    response = None
    if task == GPT_3_TASK_COMPLETION:
        if not gpt_prompt:
            print("need to provide prompt")
        else:
            response = openai.Completion.create(
                engine="davinci",
                prompt=gpt_prompt,
                temperature=0.7,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_words,
            )
    elif task == GPT_3_TASK_CLASSIFICATION:
        if not query:
            print("need to provide query")
        else:
            response = openai.Classification.create(
                file="file-GVK7z8A0vGQmPvKydNavhkyi",
                query=query,
                search_model="ada",
                model="curie",
                max_examples=3,
            )
    elif task == GPT_3_TASK_CONVERSATION:
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=gpt_prompt,
            temperature=0.5,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=stop_words,
        )
    logger.debug("Finished requesting GPT-3")
    return response


def reformat_source_data(
    source_path: Text, reformat_source_path: Text
) -> Dict[str, str]:
    """If the source is not in required format, reformat.

    Args:
        source_path (Text): Path of existing source file.
        reformat_source_path (Text): Path of reformatted file.

    Returns:
        Dict: Reformatted data dictionary.
    """
    source_file = open(source_path, "r", encoding="utf-8")
    source_data = json.load(source_file)
    reformated_file = open(reformat_source_path, "w+", encoding="utf-8")
    assert not isinstance(source_data[0]["conversation"], str)

    for data in source_data:
        tmp = ""
        for utterance in data["conversation"]:
            tmp += utterance["speaker"] + ": " + utterance["content"] + "\n"
        data["conversation"] = tmp

    json.dump(source_data, reformated_file, indent=4)
    return source_data


def read_source_data(source_path: Text) -> List[Dict[str, str]]:
    """Read as source data ESConv to fill in the prompt template.

    Args:
        source_path (Text): Path to the source data file.

    Returns:
        Dict: Return the source as data.
    """
    source_file = open(source_path, "r", encoding="utf-8")
    source_data = json.load(source_file)
    return source_data


def pick_up_examples(
    test_data: List[Dict[str, str]], number: int
) -> List[Dict[str, str]]:
    """Randomly pick up certain number of examples as prompt instances.

    Args:
        test_data (List[Dict]): Pool of all data to choose examples from.
        number (int): The number of examples to choose.

    Returns:
        List[Dict]: A list of data examples selected.
    """
    examples = []
    while len(examples) < number:
        rand_index = random.randrange(len(test_data))
        if len(examples) == 0:
            examples.append(test_data[rand_index])
        else:
            selected = False
            for example in examples:
                if example["emotion_type"] == test_data[rand_index]["emotion_type"]:
                    if example["problem_type"] == test_data[rand_index]["problem_type"]:
                        selected = True
                        break
            if not selected:
                examples.append(test_data[rand_index])
    return examples


def assembly_prompt(
    prompt: Text,
    seeker_only_file: TextIO,
    test_data: List[Dict[str, str]],
    source_data: List[Dict[str, Any]],
) -> str:
    """Assembly the final prompt with the given template and the source data.
       The method yield one prompt at a time. Each time, a user turn with previous
       dialogue history will be assigned as question of current prompt.
    Args:
        template (Text): A template with main factors representated by tokens with <>.
        text_data (List[Dict[str, str]]): Some annotated data pool from which examples are chosen.
        source_data (List[Dict[str, Any]]): The dialogue data to annotate.

    Returns:
        str: A prompt.
    """
    # template_file = open(template, "r", encoding="utf-8")
    # prompt = template_file.read()
    instances = pick_up_examples(test_data, 3)

    # Assembly prompt instances depending on the number of instances
    for index in range(len(instances)):
        prompt = prompt.replace(
            f"<conversation_{index}>", instances[index]["conversation"]
        )
        # replace <> tokens with real content as prompt instances
        prompt = prompt.replace(f"<feel_{index}>", instances[index]["feel"])
        prompt = prompt.replace(f"<reason_{index}>", instances[index]["reason"])
        prompt = prompt.replace(f"<suggestion_{index}>", instances[index]["suggestion"])

    # Assembly question and yield one prompt one time
    for data in source_data:
        current_dialog = ""
        for utterance in data["conversation"]:
            current_dialog += utterance["speaker"] + ": " + utterance["content"] + "\n"
            if utterance["speaker"] == "seeker":
                seeker_only_file.write(
                    json.dumps({"utterance": utterance["content"]}) + "\n"
                )
                dy_prompt = prompt.replace("<conversation>", current_dialog)
                yield dy_prompt


def load_large_model(model_name: Text):
    """Load the large language model based on model name.

    Args:
        model_name (Text): The given model name.

    Returns:
        _type_: Return the model file and tokenizer.
    """
    if "gpt-j" == model_name:
        model_name = "EleutherAI/gpt-j-6B"
        print(f"loading model from {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model configuration: {model.config}")
    elif "gpt-2" == model_name:
        model_name = "distilgpt2"
        print(f"loading model from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Model configuration: {model.config}")
    elif "gpt" == model_name:
        model_name = "openai-gpt"
        print(f"loading model from {model_name}")
        model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
        print(f"Model configuration: {model.config}")
    else:
        return None
    return model, tokenizer


def gpt_text_generate(prompt: Text, model, tokenizer) -> str:
    """Generate text with GPT-J 6B model using the given prompt.

    Args:
        prompt (Text): The prompt input.

    Returns:
        str: The generated answer.
    """
    # prompt = (
    #     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    #     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    #     "researchers was the fact that the unicorns spoke perfect English."
    # )

    # add padding token
    # print(tokenizer.truncation_side)
    sequence = tokenizer(
        prompt,
        return_tensors="pt",
        # truncation=True,
        # max_length=600,
    )
    input_ids = sequence["input_ids"]
    attention_mask = sequence["attention_mask"]
    model.config.pad_token_id = model.config.eos_token_id
    # print(tokenizer.eos_token)

    try:
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=80,
            attention_mask=attention_mask,
        )
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
    except RuntimeError:
        gen_text = "<padding> <padding> <padding> <padding> <padding>"
    return gen_text


def process_prompt_length(
    prompt: Text, allowed_dialog_length: int, tokenizer, single_utterance: bool = False
) -> Text:
    current_dialog = prompt
    utterances = current_dialog.split("\n")
    tmp_text = "\n".join(utterances)
    if "Conversation:" in prompt:
        current_dialog = prompt.split("Conversation:")[-1].split(
            "In this conversation,"
        )[0][1:]
        print(f"dialog before processing: \n{current_dialog}")
        utterances = current_dialog.split("\n")[:-1]
        utterances.pop(0)
        tmp_text = "\n".join(utterances)

    print(len(tokenizer(tmp_text)["input_ids"]))
    if not single_utterance:
        while len(tokenizer(tmp_text)["input_ids"]) > allowed_dialog_length:
            utterances.pop(0)
            tmp_text = "\n".join(utterances)
    else:
        if len(utterances) >= 2:
            tmp_text = utterances[-2] + "\n" + utterances[-1]
        else:
            tmp_text = utterances[0]

    print(f"dialog after processing: \n{tmp_text}")
    print(len(tokenizer(tmp_text)["input_ids"]))
    return tmp_text + "\n"


def prompting(prompt: Text, model_name: Text, model, tokenizer) -> str:
    # load model with given name
    if "gpt-2" == model_name:
        return gpt_text_generate(prompt, model, tokenizer)
    return


def dump_response(message: Text, response_file: Text) -> None:
    response_file.write(json.dumps({"response": message}) + "\n")


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-j",
        help="select model name from ['gpt-j', 'gpt-2', 'gpt']",
    )
    parser.add_argument(
        "--sample_number", type=int, default=0, help="how many samples to generate"
    )
    parser.add_argument(
        "--prompt_template", type=str, required=True, help="prompt template to use"
    )
    parser.add_argument("--start_index", type=int, default=0, help="where to start")
    parser.add_argument(
        "--response_suffix",
        type=str,
        default="default_batch",
        help="folder to save the results",
    )
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = add_arguments()
    # load data
    template_file = open(args.prompt_template, "r", encoding="utf-8")
    fixed_prompt = template_file.read()
    source_data = read_source_data(source_data_path)
    test_data = read_source_data(test_data_path)
    response_file = open(
        response_path + args.response_suffix + ".jsonl", "w+", encoding="utf-8"
    )
    seeker_only_file = open(
        seeker_utterances_only + args.response_suffix + ".jsonl", "w+", encoding="utf-8"
    )
    # load model
    model_name = args.model_name
    model, tokenizer = load_large_model(model_name)

    # get fixed template length
    fixed_sequence = tokenizer(fixed_prompt)
    fixed_length = len(fixed_sequence["input_ids"])
    allowed_dialog_length = 900 - 80 - fixed_length - 1
    if model_name == "gpt":
        allowed_dialog_length = 500 - 80 - fixed_length - 1

    # load prompt generator
    prompt_generator = assembly_prompt(
        fixed_prompt,
        seeker_only_file,
        test_data,
        source_data,
    )

    for _ in range(args.start_index):
        next(prompt_generator)

    if args.sample_number == 0:
        for i in prompt_generator:
            prompt = next(prompt_generator)
            logger.debug(len(prompt))

            current_length = len(tokenizer(prompt)["input_ids"]) - fixed_length + 1
            print(current_length)
            if current_length > allowed_dialog_length:
                print(
                    f"current dialog length {current_length} is longer than allowed dialog length {allowed_dialog_length}. The beginning part of the conversation will be removed adaptively."
                )
                prompt = prompt.replace(
                    "<conversation>",
                    process_prompt_length(prompt, allowed_dialog_length, tokenizer),
                )
                print(prompt)

            response = gpt_text_generate(prompt, model, tokenizer)
            response = response[len(prompt) :]
            logger.info(response)
            dump_response(response, response_file)
    else:
        for i in range(args.sample_number):
            prompt = next(prompt_generator)
            logger.debug(len(prompt))

            current_length = len(tokenizer(prompt)["input_ids"]) - fixed_length + 1
            print(current_length)
            if current_length > allowed_dialog_length:
                print(
                    f"current dialog length {current_length} is longer than allowed dialog length {allowed_dialog_length}. The beginning part of the conversation will be removed adaptively."
                )
                prompt = fixed_prompt.replace(
                    "<conversation>",
                    process_prompt_length(prompt, allowed_dialog_length, tokenizer),
                )
                print(prompt)

            response = gpt_text_generate(prompt, model, tokenizer)
            response = response[len(prompt) :]
            logger.info(response)
            dump_response(response, response_file)


def inference(model_name, model, tokenizer, prompt_template: Text, current_dialog) -> Text:
    """Inference gpt models in real time.

    Args:
        model_name (_type_): Name of reasoning model.
        model (_type_): Reasoning model
        tokenizer (_type_): Tokenizer of reasonin model.
        prompt_template (Text): Prompt template.
        current_dialog (_type_): Current updating conversation.

    Returns:
        Text: The reasoning response from the reasoning model.
    """
    template_file = open(prompt_template, "r", encoding="utf-8")
    fixed_prompt = template_file.read()
    # model, tokenizer = load_large_model(model_name)

    fixed_sequence = tokenizer(fixed_prompt)
    fixed_length = len(fixed_sequence["input_ids"])
    allowed_dialog_length = 1000 - 80 - fixed_length - 1
    if model_name == "gpt":
        allowed_dialog_length = 500 - 80 - fixed_length - 1


    prompt = fixed_prompt.replace(
        "<conversation>",
        process_prompt_length(current_dialog, allowed_dialog_length, tokenizer),
    )
    response = gpt_text_generate(prompt, model, tokenizer)
    # response = response[len(prompt) :]
    print(f"Original response: {response}")
    response = response.split("In this conversation,")[-1].split(":")[0].replace("\nsupporter", "")
    logger.info(response)
    return response


if __name__ == "__main__":
    main()
