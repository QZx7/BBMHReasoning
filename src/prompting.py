import json
import random
from typing import Any, Dict, List, Text

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

template_path = r"./src/prompt_templates/nl_no_token.txt"
source_data_path = r"./data/ESConv_one_speaker_one_turn.json"
test_data_path = r"./data/ESConv_test_data.json"
response_path = r"./data/NL_response.jsonl"


def read_prompt(prompt_path: Text) -> Text:
    """Read a prompt template from text file.

    Args:
        prompt_path (Text): Path to the prompt template file.

    Returns:
        Text: Prompt template as plain text.
    """
    template_file = open(prompt_path, "r", encoding="utf-8")
    return template_file.read()


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
    template: Text, test_data: List[Dict[str, str]], source_data: List[Dict[str, Any]]
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
    template_file = open(template, "r", encoding="utf-8")
    prompt = template_file.read()
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
    elif "gpt-2" == model_name:
        model_name = "distilgpt2"
        print(f"loading model from {model_name}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
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

    sequence = tokenizer(prompt, return_tensors="pt")
    input_ids = sequence["input_ids"]
    attention_mask = sequence["attention_mask"]
    model.config.pad_token_id = model.config.eos_token_id
    # print(tokenizer.eos_token)

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.7,
        max_length=1500,
        attention_mask=attention_mask,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text


def prompting(prompt: Text, model_name: Text, model, tokenizer) -> str:
    # load model with given name
    if "gpt-2" == model_name:
        return gpt_text_generate(prompt, model, tokenizer)
    return


def dump_response(message: Text, response_file: Text) -> None:
    response_file.write(json.dumps({"response": message}) + "\n")


def main():
    source_data = read_source_data(source_data_path)
    test_data = read_source_data(test_data_path)
    response_file = open(response_path, "a+", encoding="utf-8")
    # load model
    model_name = "gpt-j"
    model, tokenizer = load_large_model(model_name)
    # load prompt generator
    prompt_generator = assembly_prompt(template_path, test_data, source_data)

    for i in range(5):
        prompt = next(prompt_generator)
        print(prompt)
        print("==================")
        response = gpt_text_generate(prompt, model, tokenizer)
        response = response[len(prompt):]
        print("*******************")
        dump_response(response, response_file)


if __name__ == "__main__":
    main()
