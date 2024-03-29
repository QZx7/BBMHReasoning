import argparse
import json
import random
import logging
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
dialog_data_path = r"./eval/reasoning_evaluation/samples.jsonl"
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
    model_type: Optional[Text] = "ada"
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
            model_name = model_name = "text-" + model_type + "-001"
            max_length = 80
            if model_type == 'davinci':
                logger.info("GPT-3 type: %s", model_type)
                model_name = "text-" + model_type + "-002"
                max_length = 120
            response = openai.Completion.create(
                engine=model_name,
                prompt=gpt_prompt,
                temperature=0.7,
                max_tokens=max_length,
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
                search_model=model_type,
                model="curie",
                max_examples=3,
            )
    elif task == GPT_3_TASK_CONVERSATION:
        model_name = model_name = "text-" + model_type + "-001"
        if model_type == 'davinci':
            model_name = "text-" + model_type + "-002"
        response = openai.Completion.create(
            engine=model_name,
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


def read_dialog_data(dialog_path: Text) -> List[Text]:
    """Read dialog directly from dialog style files.

    Args:
        dialog_path (Text): Path to the dialog data file path.

    Returns:
        List[Text]: Return the dialog data as list of strings.
    """
    dialog_data = []
    with open(dialog_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            dialog_data.append(json.loads(line.strip())["content"]["dialog"])
    return dialog_data


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
    source_data: List[Dict[Text, Any]],
    seeker_only_file: Optional[TextIO] = None,
) -> str:
    """Assembly the final prompt with the given template and the source data.
       The method yield one prompt at a time. Each time, a user turn with previous
       dialogue history will be assigned as question of current prompt.
    Args:
        template (Text): A template with main factors representated by tokens with <>.
        seeker_only_file (Optional[TextIO]): The file IO stream to write the seeker utterances.
        source_data (List[Dict[str, Any]]): The dialogue data to annotate.

    Returns:
        str: A prompt.
    """
    # Assembly question and yield one prompt one time
    for data in source_data:
        current_dialog = ""
        for utterance in data["conversation"]:
            current_dialog += utterance["speaker"] + ": " + utterance["content"] + "\n"
            if utterance["speaker"] == "seeker":
                if seeker_only_file:
                    seeker_only_file.write(
                        json.dumps({"utterance": utterance["content"]}) + "\n"
                    )
                dy_prompt = prompt.replace("<conversation>", current_dialog)
                yield dy_prompt


def prompt_from_dialog_data(prompt: Text, dialog_data: List[Text]) -> str:
    """Assembly prompt, however from existing dialog style data instead of utterance based
       data.

    Args:
        prompt (Text): A template with main factors represented by tokens with <>.
        dialog_data (List[Text]): A list of dialogues for prompting.

    Returns:
        str: A prompt.
    """
    for data in dialog_data:
        data = data.replace("\nIn this conversation, the seeker", "\n")
        dy_prompt = prompt.replace("<conversation>", data)
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
        model_name = "gpt2"
        print(f"loading model from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Model configuration: {model.config}")
    elif "distilgpt2" == model_name:
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
    elif model_name in ["ada", "davinci", "gpt-3"]:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = None
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

    sequence = tokenizer(
        prompt,
        return_tensors="pt",
        # truncation=True,
        # max_length=600,
    )
    input_ids = sequence["input_ids"]
    attention_mask = sequence["attention_mask"]
    model.config.pad_token_id = model.config.eos_token_id

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

        utterances = current_dialog.split("\n")[:-1]
        utterances.pop(0)
        tmp_text = "\n".join(utterances)

    if not single_utterance:
        while len(tokenizer(tmp_text)["input_ids"]) > allowed_dialog_length:
            utterances.pop(0)
            tmp_text = "\n".join(utterances)
    else:
        if len(utterances) >= 2:
            tmp_text = utterances[-2] + "\n" + utterances[-1]
        else:
            tmp_text = utterances[0]

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
        help="select model name from ['gpt-j', 'gpt-2', 'gpt', 'distilgpt2', 'gpt-3]",
    )
    parser.add_argument(
        "--model_type", type=str, default='ada', help="Specific model type for gpt-3"
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
    parser.add_argument("--use_dialog", type=bool, default=False, help="weather to use dialog data")
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = add_arguments()
    # load data
    template_file = open(args.prompt_template, "r", encoding="utf-8")
    fixed_prompt = template_file.read()
    source_data = read_source_data(source_data_path)
    dialog_data = read_dialog_data(dialog_data_path)
    # test_data = read_source_data(test_data_path)
    response_file = open(
        response_path + args.response_suffix + ".jsonl", "w+", encoding="utf-8"
    )
    seeker_only_file = open(
        seeker_utterances_only + args.response_suffix + ".jsonl", "w+", encoding="utf-8"
    )
    # load model
    model_name = args.model_name
    model, tokenizer = load_large_model(model_name)
    logger.info("loaded tokenizer and model from %s", model_name)
    # get fixed template length
    fixed_sequence = tokenizer(fixed_prompt)
    fixed_length = len(fixed_sequence["input_ids"])
    logger.info("loaded fixed template length %s", fixed_length)
    max_input_length = 1000
    response_length = 80
    if args.model_name == "gpt-3":
        if args.model_type == "davinci":
            max_input_length = 3000
            response_length = 120
        else:
            max_input_length = 2000
            response_length = 80
    if args.model_name == "gpt":
        max_input_length = 500
        response_length = 80
    allowed_dialog_length = max_input_length  - response_length - fixed_length - 1
    logger.info("Set max input length to %s, response length to %s and allowed dialog length to %s", max_input_length, response_length, allowed_dialog_length)
    # load prompt generator
    prompt_generator = None

    if args.use_dialog:
        prompt_generator = prompt_from_dialog_data(
            fixed_prompt,
            dialog_data
        )
    else:
        prompt_generator = assembly_prompt(
            fixed_prompt,
            seeker_only_file,
            source_data,
        )

    for _ in range(args.start_index):
        next(prompt_generator)

    # total_token_num = 0
    # sample_index = 0
    # if generate number is 0, generate until the end.
    if args.sample_number == 0:
        for prompt in prompt_generator:
            # prompt = next(prompt_generator)
            current_length = len(tokenizer(prompt)["input_ids"]) - fixed_length + 1
            if current_length > allowed_dialog_length:
                print(
                    f"current dialog length {current_length} is longer than allowed dialog length {allowed_dialog_length}. The beginning part of the conversation will be removed adaptively."
                )
                prompt = fixed_prompt.replace(
                    "<conversation>",
                    process_prompt_length(prompt, allowed_dialog_length, tokenizer),
                )
                print(prompt)
            if "gpt-3" == model_name:
                response = get_gpt_result("completion", prompt, stop_words=['\n'], model_type=args.model_type)
                response = response["choices"][0]["text"]
            else:
                response = gpt_text_generate(prompt, model, tokenizer)
                response = response[len(prompt) :]
            logger.info(response)
            dump_response(response, response_file)
    # if generate number is not 0, generate the given number of samples.
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
            # total_token_num += len(tokenizer(prompt)["input_ids"])
            # sample_index += 1
            if "gpt-3" == model_name:
                response = get_gpt_result("completion", prompt, stop_words=['\n'], model_type=args.model_type)
                response = response["choices"][0]["text"]
            else:
                response = gpt_text_generate(prompt, model, tokenizer)
                response = response[len(prompt) :]
            logger.info(response)
            dump_response(response, response_file)
        # print(sample_index, total_token_num)


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
    max_input_length = 1000
    response_length = 80
    if model_name == "davinci":
        max_input_length = 3000
        response_length = 120
    elif model_name == "ada":
        max_input_length = 2000
        response_length = 80
    elif model_name == "gpt":
        max_input_length = 500
        response_length = 80
    allowed_dialog_length = max_input_length  - response_length - fixed_length - 1
    prompt = fixed_prompt.replace(
        "<conversation>",
        process_prompt_length(current_dialog, allowed_dialog_length, tokenizer),
    )
    print(prompt)

    response = ""
    if model_name == "gpt":
        response = gpt_text_generate(prompt, model, tokenizer)
        # print(response)
        # response = response.split("in this conversation, the seeker")[-1].split(":")[0].replace("\nsupporter", "").replace("\nConversation", "")
        response = response.split("in this conversation, the seeker")[-1].split("\n")[0]
    elif model_name == "gpt-2":
        response = gpt_text_generate(prompt, model, tokenizer)
        # print(response)
        response = response.split("In this conversation, the seeker")[-1].split("\n")[0]
    elif model_name in ["ada", "davinci"]:
        response = get_gpt_result("completion", prompt, stop_words=['\n'], model_type=model_name)
        response = response["choices"][0]["text"]
    print(response)
    logger.info(response)
    return " The seeker " + response


if __name__ == "__main__":
    main()
