#!/bin/sh
python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmhr/gpt_2/3B/bbmhr_27B \
--use-reasoning True \
--reasoning-model-name gpt-2 \
--prompt-path ./bbmhr/prompt_templates/nl_zero.txt \
--outfile ./eval/self_chat/gpt_2_academic.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_1/academic.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmhr/gpt_2/3B/bbmhr_27B \
--use-reasoning True \
--reasoning-model-name gpt-2 \
--prompt-path ./bbmhr/prompt_templates/nl_zero.txt \
--outfile ./eval/self_chat/gpt_2_job.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_1/job.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmhr/gpt_2/3B/bbmhr_27B \
--use-reasoning True \
--reasoning-model-name gpt-2 \
--prompt-path ./bbmhr/prompt_templates/nl_zero.txt \
--outfile ./eval/self_chat/gpt_2_friend.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_1/friend.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmhr/gpt_2/3B/bbmhr_27B \
--use-reasoning True \
--reasoning-model-name gpt-2 \
--prompt-path ./bbmhr/prompt_templates/nl_zero.txt \
--outfile ./eval/self_chat/gpt_2_ongoing.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_1/ongoing.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmhr/gpt_2/3B/bbmhr_27B \
--use-reasoning True \
--reasoning-model-name gpt-2 \
--prompt-path ./bbmhr/prompt_templates/nl_zero.txt \
--outfile ./eval/self_chat/gpt_2_partner.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_1/partner.txt