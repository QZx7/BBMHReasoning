#!/bin/sh
python3 ./bbmhr/parlai/scripts/interactive_bbmhr.py \
-t blended_skill_talk \
-mf ~/dialogue_base/blenderbot/model/bbmhr/gpt_1/3B/bbmhr_27B \
-use_gpt True \
--prompt_path ./bbmhr/prompt_templates/nl_utt_level.txt \
--reasoning_model_name gpt