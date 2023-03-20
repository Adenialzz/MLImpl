import torch
from gpt import GPT, GPTConfig

config = GPTConfig()
gpt_model = GPT(config)
gpt_model.eval()


sequence = [2, 3, 11, 4]
inp = torch.tensor(sequence).unsqueeze(dim=0)
print(inp.shape)
out = gpt_model.generate(inp, 20)
print(out[0].shape)




