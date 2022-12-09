#!/bin/sh
python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmh/27B/bbmh_27B \
--outfile ./eval/self_chat/bbmh_academic.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/academic.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmh/27B/bbmh_27B \
--outfile ./eval/self_chat/bbmh_job.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/job.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmh/27B/bbmh_27B \
--outfile ./eval/self_chat/bbmh_friend.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/friend.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmh/27B/bbmh_27B \
--outfile ./eval/self_chat/bbmh_ongoing.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/ongoing.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file ~/dialogue_base/blenderbot/model/bbmh/27B/bbmh_27B \
--outfile ./eval/self_chat/bbmh_partner.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/partner.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file zoo:blender/blender_3B/model \
--outfile ./eval/self_chat/bb_academic.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/academic.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file zoo:blender/blender_3B/model \
--outfile ./eval/self_chat/bb_job.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/job.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file zoo:blender/blender_3B/model \
--outfile ./eval/self_chat/bb_friend.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/friend.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file zoo:blender/blender_3B/model \
--outfile ./eval/self_chat/bb_ongoing.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/ongoing.txt

python3 bbmhr/parlai/scripts/self_chat_mental.py \
--model-file ~/dialogue_base/blenderbot/model/bbmh/seeker/bbmh_27B \
--task blended_skill_talk \
--num-self-chats 1 \
--selfchat-max-turns 20 \
--partner-model-file zoo:blender/blender_3B/model \
--outfile ./eval/self_chat/bb_partner.txt \
--seed-messages-from-file ./eval/self_chat/seeds/seed_2/partner.txt