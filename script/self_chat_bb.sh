#!/bin/sh
python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 15 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmh/27B/bbmh_27B \
--outfile ./eval/self_chat/bbmh_academic.txt \
--seed-messages-from-file ./eval/self_chat/seeds/bbmh_academic.txt