import json
from typing import List, Text, Union
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from evaluate import load

from bart_score import BARTScorer
from transformers import DebertaTokenizer, RobertaTokenizer
from transformers import DebertaForSequenceClassification, RobertaForSequenceClassification
from transformers import pipeline

def read_reasoning_data(reasoning_path: Text, task_name: Text, metric_style: Text = "bleu"):
    reasoning_file = open(reasoning_path, 'r', encoding='utf-8')
    preds = []
    refes = []
    for line in reasoning_file.readlines():
        line_data = json.loads(line.strip())
        preds.append(line_data["content"][task_name])
        if metric_style == "bleu":
            refes.append(["the seeker " + line_data["content"]['human']])
        elif metric_style in ["rouge", "bert", "entailment"]:
            refes.append("the seeker " + line_data['content']['human'])

    return preds, refes


def calculate_score(preds: List[Text], refes: Union[List[Text], List[List[Text]]], metric: Text = "bleu"):
    if metric == "bleu":
        metric = BLEUScore(n_gram=4)
    elif metric == "rouge":
        metric = ROUGEScore(rouge_keys='rougeL')
    elif metric == "bert":
        bertscore = load('bertscore')
        results = bertscore.compute(predictions=preds, references=refes, model_type="roberta-large", lang="en")
        average_f1 = sum(results["f1"]) / len(results["f1"])
        return average_f1
    elif metric == "bart":
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        bart_scorer.load(path=r"D:\\project\\dataset\\bart_score\\bart_score.pth")
        results = bart_scorer.score(preds, refes, batch_size=4)
        print(results)
        return results
    return metric(preds, refes)


def entailment_score(preds: List[Text], refes: List[Text], model_name: Text = "roberta"):
    label_dict = {
            'CONTRADICTION': [],
            'NEUTRAL': [],
            'ENTAILMENT': []
    }
    labels = ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']
    entailment_scores = []
    model = None
    tokenizer = None

    if model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
    elif model_name == "deberta":
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base-mnli')
        model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base-mnli')
    
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)
    for index in range(len(preds)):
        sample = ""
        if model_name == "roberta":
            sample = "<s> " + preds[index] + " </s> <s> " + refes[index] + "</s>"
        elif model_name == "deberta":
            sample = "[CLS] " + preds[index] + " [SEP] " + refes[index] + " [SEP]"
        result = classifier(sample)[2]
        # print(result)
        entailment_scores.append(result['score'])

    print(sum(entailment_scores) / len(entailment_scores))


if __name__ == '__main__':

    # for metric in ["bleu", "rouge", "bert"]:
    #     print(f"score: {metric}")
    #     for task in ["gpt_1", "gpt_2", "ada", "davinci"]:
    #         preds, refes = read_reasoning_data(r"./eval/reasoning_evaluation/samples.jsonl", task)
    #         print(f"task: {task}")
    #         print(calculate_score(preds, refes, metric))

    for task in ["gpt_1", "gpt_2", "ada", "davinci"]:
        preds, refes = read_reasoning_data(r"./eval/reasoning_evaluation/samples.jsonl", task, "entailment")
        print(f"task: {task}")
        print(entailment_score(preds, refes, model_name="roberta"))
