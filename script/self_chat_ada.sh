#!/bin/sh
python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 30 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmhr/ada/3B/bbmhr_27B \
--use-reasoning True \
--reasoning-model-name ada \
--prompt-path ./bbmhr/prompt_templates/nl_gpt_3.txt \
--outfile ./eval/self_chat/ada_friend.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_1/friend.txt