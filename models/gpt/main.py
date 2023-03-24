import torch
import tiktoken
from gpt import GPT

n_samples = 10
max_new_tokens = 50
top_k = 16
temperature = 0.8
prompt = 'I want to earn more money, so I'
device = 'cpu'

gpt_model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
gpt_model.eval()
gpt_model.to(device)

eos_token = '<|endoftext|>'
enc = tiktoken.get_encoding('gpt2')
encode = lambda s: enc.encode(s, allowed_special={eos_token})
decode = lambda l: enc.decode(l)

start_ids = encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
with torch.no_grad():
    for k in range(n_samples):
        y = gpt_model.generate(x, max_new_tokens, temperature, top_k)
        print(decode(y[0].tolist()))
        print('-' * 100)


