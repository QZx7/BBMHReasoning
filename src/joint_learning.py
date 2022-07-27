import random

from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

class RepeatLabelAgent(Agent):
    def __init__(self, opt):
        self.id = "RepeatLabel"
    
    def observe(self, observation):
        self.observation = observation
        return observation

    def act(self):
        reply = {"id": self.id}
        if "labels" in self.observation:
            reply["text"] = ', '.join(self.observation["labels"])
        elif 'label_candidates' in self.observation:
            cands = self.observation['label_candidates']
            reply['text'] = random.choice(list(cands))
        else:
            reply["text"] = "I don't know."
        return reply

parser = ParlaiParser()
opt = parser.parse_args()

agent = RepeatLabelAgent(opt)
world = create_task(opt, agent)

for _ in range(10):
    world.parley()
    print(world.display())
    if world.epoch_done():
        print("Done")
        break
