# BBMHR
This is the repository for Paper: 

**Ask an Expert: Leveraging Language Models to Improve Strategic Reasoning in Goal-Oriented Dialogue Models**
(https://aclanthology.org/2023.findings-acl.417/)

## Abstract
Existing empathetic conversational models can
frequently generate vacuous responses. Adding
reasoning processes is promising for a model
to generate more professional and strategic re-
sponses in the goal-oriented settings. This
work explores the possibility of consulting pre-
trained language models for reasoning pro-
cesses and demonstrates how adding such rea-
soning can help a model generate suggestive,
specific, and engaging responses in the domain
of mental support. Prompting is used to an-
notate dialogues in a mental support conversa-
tion dataset with reasoning processes on emo-
tional status, reason, and possible suggestions.
We use the state-of-the-art empathetic dialogue
model Blenderbot 2.7B for the dialogue model
and assess the model through human evalua-
tion. We show that adding reasoning processes
is able to lead to an overall improvement of
9.02% ± 3.6% in terms of being engaging and
suggestive compared to the baseline without
reasoning.

## Data
Files for data that is used `/data`. Where the data file for `bbmh/no_reasoning` is converted from dataset [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation).

Data file for `bbmhr` are reasoning annotated dialogue data by different expert models.

## Results
Results for the auto evaluation on reasoning annotations.

![image](https://github.com/QZx7/BBMHReasoning/assets/62750920/50ad3ac8-e679-4fd5-8c46-53bfffc0e3fc)

Results for the human evaluation on reasoning annotations.

![image](https://github.com/QZx7/BBMHReasoning/assets/62750920/6131d9dd-3a59-4507-9bdd-6eae2bd2a8c7)

Results for the human evaluation on dialogues by different BBMHR models.

![image](https://github.com/QZx7/BBMHReasoning/assets/62750920/d7712cf8-13dc-4b89-b92b-6f46db03aff6)

## Train
1. Install `parlai` by `pip install parlai`
2. Install `transformers` by `pip install transformers`
3. clone this repo and run the training script with data annotated with specific expert model.
e.g., to train a model with reasoning annotation from gpt_2 expert, run:
```
python3 ~/anaconda3/envs/persona/lib/python3.8/site-packages/parlai/scripts/train_model.py \
-t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues,fromfile:parlaiformat \
--fromfile_datapath ../data/3B/new_gpt_2/train_parlai_b0_14.txt \
--multitask-weights 1,3,3,3,1 \
-veps 0.25 \
--attention-dropout 0.0 \
--batchsize 8 \
--model transformer/generator \
--embedding-size 2560 \
--ffn-size 10240 \
--variant prelayernorm \
--n-heads 32 \
--n-positions 128 \
--n-encoder-layers 2 \
--n-decoder-layers 24 \
--history-add-global-end-token end \
--delimiter '  ' \
--dict-tokenizer bytelevelbpe \
--dropout 0.1 \
--fp16 True \
--init-model zoo:blender/reddit_3B/model \
--dict-file zoo:blender/reddit_3B/model.dict \
--label-truncate 128 \
--log_every_n_secs 10 \
-lr 7e-06 \
--lr-scheduler reduceonplateau \
--lr-scheduler-patience 3 \
--optimizer adam \
--relu-dropout 0.0 \
--activation gelu \
--model-parallel true \
--save-after-valid True \
--text-truncate 128 \
--truncate 128 \
--warmup_updates 100 \
--fp16-impl mem_efficient \
--update-freq 2 \
--gradient-clip 0.1 \
--skip-generation True \
-vp 10 \
-vmt ppl \
-vmm min \
--model-file ../model/bbmhr/new_gpt_2/3B/bbmhr_27B
```

## Interactive
To interactive with the trained BBMHR model, run:
```
python3 ./bbmhr/parlai/scripts/interactive_bbmhr.py \
-t blended_skill_talk \
-mf /path/to/model/file \
--use_gpt True \
--prompt_path ./bbmhr/prompt_templates/nl_gpt_3.txt \
--reasoning_model_name davinci \
--save_history_path ./eval/human_evaluation/davinci_job.txt
```
where `reasoning_model_name` is the name of the expert and `prompt_path` is the path of the prompt to use.

## Cite as
```
@inproceedings{zhang-etal-2023-ask,
    title = "Ask an Expert: Leveraging Language Models to Improve Strategic Reasoning in Goal-Oriented Dialogue Models",
    author = "Zhang, Qiang  and
      Naradowsky, Jason  and
      Miyao, Yusuke",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.417",
    doi = "10.18653/v1/2023.findings-acl.417",
    pages = "6665--6694",
    abstract = "Existing dialogue models may encounter scenarios which are not well-represented in the training data, and as a result generate responses that are unnatural, inappropriate, or unhelpful. We propose the {``}Ask an Expert{''} framework in which the model is trained with access to an {``}expert{''} which it can consult at each turn. Advice is solicited via a structured dialogue with the expert, and the model is optimized to selectively utilize (or ignore) it given the context and dialogue history. In this work the expert takes the form of an LLM.We evaluate this framework in a mental health support domain, where the structure of the expert conversation is outlined by pre-specified prompts which reflect a reasoning strategy taught to practitioners in the field. Blenderbot models utilizing {``}Ask an Expert{''} show quality improvements across all expert sizes, including those with fewer parameters than the dialogue model itself. Our best model provides a {\textasciitilde}10{\%} improvement over baselines, approaching human-level scores on {``}engingingness{''} and {``}helpfulness{''} metrics.",
}
```
