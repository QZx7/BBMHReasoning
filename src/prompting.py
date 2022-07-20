import json
import random
from typing import Dict, List, Text

from transformers import AutoModelForCausalLM, AutoTokenizer

template_path = r'./src/prompt_templates/natural_language.txt'
source_data_path = r'./data/ESConv_prompt_style.json'
test_data_path = r'./data/ESConv_test_data.json'

def read_prompt(prompt_path: Text) -> Text:
    """Read a prompt template from text file.

    Args:
        prompt_path (Text): Path to the prompt template file.

    Returns:
        Text: Prompt template as plain text.
    """
    template_file = open(prompt_path, 'r', encoding='utf-8')
    return template_file.read()


def reformat_source_data(source_path: Text, reformat_source_path: Text) -> Dict[str, str]:
    """If the source is not in required format, reformat.

    Args:
        source_path (Text): Path of existing source file.
        reformat_source_path (Text): Path of reformatted file.

    Returns:
        Dict: Reformatted data dictionary.
    """
    source_file = open(source_path, 'r', encoding='utf-8')
    source_data = json.load(source_file)
    reformated_file = open(reformat_source_path, 'w+', encoding='utf-8')
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
    source_file = open(source_path, 'r', encoding='utf-8')
    source_data = json.load(source_file)
    return source_data


def pick_up_examples(test_data: List[Dict[str, str]], number: int) -> List[Dict[str, str]]:
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
                if example['emotion_type'] == test_data[rand_index]['emotion_type']:
                    if example['problem_type'] == test_data[rand_index]['problem_type']:
                        selected = True
                        break
            if not selected:
                examples.append(test_data[rand_index])
    return examples


def assembly_prompt(template: Text, test_data: List[Dict[str, str]], source_data: List[Dict[str, str]]) -> str:
    """Assembly the final prompt with the given template and the source data.

    Args:
        template (Text): A template with main factors representated by tokens with <>.
        text_data (List[Dict[str, str]]): Some annotated data pool from which examples are chosen.
        source_data (List[Dict[str, str]]): The dialogue data to annotate.

    Returns:
        str: A prompt.
    """
    template_file = open(template, 'r', encoding='utf-8')
    prompt = template_file.read()
    instances = pick_up_examples(test_data, 3)

    # Assembly prompt instances depending on the number of instances
    for index in range(len(instances)):
        prompt = prompt.replace(f"<conversation_{index}>", instances[index]["conversation"])
        prompt = prompt.replace(f"<feel_{index}>", instances[index]["feel"])
        prompt = prompt.replace(f"<reason_{index}>", instances[index]["reason"])
        prompt = prompt.replace(f"<suggestion_{index}>", instances[index]["suggestion"])
    
    # Assembly question and yield one prompt one time
    for data in source_data:
        dy_prompt = prompt.replace("<conversation>", data['conversation'])
        yield dy_prompt


def load_large_model(model_name: Text):
    """Load the large language model based on model name.

    Args:
        model_name (Text): The given model name.

    Returns:
        _type_: Return the model file and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def gpt_j_text_generate(prompt: Text, model, tokenizer) -> str:
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

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text


def prompting(prompt: Text, model_name: Text) -> str:
    # load model with given name
    if "gpt-j" == model_name:
        return gpt_j_text_generate(prompt)


def main():
    source_data = read_source_data(source_data_path)
    test_data = read_source_data(test_data_path)
    prompt_generator = assembly_prompt(template_path, test_data, source_data)
    load_large_model("EleutherAI/gpt-j-6B")
    while True:
        input_text = input("command\n")
        if input_text == "n":
            prompt = next(prompt_generator)
            print("=========================")
            response = prompting(prompt)
            print(response)


if __name__ == "__main__":
    main()
