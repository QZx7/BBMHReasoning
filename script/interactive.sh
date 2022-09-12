#!/bin/sh
python3 ./bbmhr/parlai/scripts/interactive_bbmhr.py \
-t blended_skill_talk \
-mf ~/dialogue_base/blenderbot/model/bbmhr/new_gpt_2/3B/bbmhr_27B \
--use_gpt True \
--prompt_path ./bbmhr/prompt_templates/nl_zero.txt \
--reasoning_model_name gpt-2 \
--save_history_path ./eval/human_evaluation/gpt_2_job.txt