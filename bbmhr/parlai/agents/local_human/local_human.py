#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent does gets the local keyboard input in the act() function.

Example: parlai eval_model -m local_human -t babi:Task1k:1 -dt valid
"""
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.utils.misc import display_messages, load_cands
from parlai.utils.strings import colorize

from bbmhr.pipeline.prompting import read_prompt, get_gpt_result


class LocalHumanReasoningAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group("Local Human Arguments")
        agent.add_argument(
            "-fixedCands",
            "--local-human-candidates-file",
            default=None,
            type=str,
            help="File of label_candidates to send to other agent",
        )
        agent.add_argument(
            "--single_turn",
            type="bool",
            default=False,
            help="If on, assumes single turn episodes.",
        )
        agent.add_argument(
            "--prompt_path", required=True, type=str, help="File for the prompt."
        )
        agent.add_argument(
            "--use_gpt",
            default=False,
            type="bool",
            help="Whether to use GPT for real time inference or local data.",
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = "localHumanReasoning"
        self.episodeDone = False
        self.finished = False
        self.fixedCands_txt = load_cands(self.opt.get("local_human_candidates_file"))
        self.prompt_prefix = read_prompt(self.opt.get("prompt_path"))
        self.history = ""
        print(
            colorize(
                "Enter [DONE] if you want to end the episode, [EXIT] to quit.",
                "highlight",
            )
        )

    def epoch_done(self):
        return self.finished

    def observe(self, msg):
        self.history += "supporter: " + msg["text"] + "\n"
        print(
            display_messages(
                [msg],
                add_fields=self.opt.get("display_add_fields", ""),
                prettify=self.opt.get("display_prettify", False),
                verbose=self.opt.get("verbose", False),
            )
        )

    def act(self):
        reply = Message()
        reply["id"] = self.getID()
        try:
            reply_text = input(colorize("Enter Your Message:", "text") + " ")
            if self.opt.get("use_gpt"):
                self.history += f"seeker: {reply_text}\n"
                prompt = self.prompt_prefix.replace("<conversation>", self.history)
                gpt_response = get_gpt_result("completion", prompt, stop_words=["\n"])[
                    "choices"
                ][0]["text"]
                print(gpt_response)
                reply_text += gpt_response
            # print(reply_text)
        except EOFError:
            self.finished = True
            return {"episode_done": True}

        reply_text = reply_text.replace("\\n", "\n")
        reply["episode_done"] = False
        if self.opt.get("single_turn", False):
            reply.force_set("episode_done", True)
        reply["label_candidates"] = self.fixedCands_txt
        if "[DONE]" in reply_text:
            # let interactive know we're resetting
            raise StopIteration
        reply["text"] = reply_text
        if "[EXIT]" in reply_text:
            self.finished = True
            raise StopIteration
        return reply

    def episode_done(self):
        return self.episodeDone
