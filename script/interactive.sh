#!/bin/sh
python3 ./bbmhr/parlai/scripts/interactive_bbmhr.py \
-t blended_skill_talk \
-mf ~/dialogue_base/blenderbot/model/bbmhr/davinci/3B/bbmhr_27B \
--use_gpt True \
--prompt_path ./bbmhr/prompt_templates/nl_gpt_3.txt \
--reasoning_model_name davinci \
--save_history_path ./eval/human_evaluation/davinci_job.txt