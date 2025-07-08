import os
from transformers import Qwen2Config, AutoTokenizer
from qwen import Qwen2_5
from safetensors.torch import load_file

# model_root = 'Qwen/Qwen2.5-0.5B'
model_root = 'hf_models/qwen25_0_5b_instruct'
device = 'cuda'

config = Qwen2Config.from_pretrained(model_root)
model = Qwen2_5(**config.to_dict()).to(device)
model = model.eval().requires_grad_(False)

state_dict = load_file(os.path.join(model_root, 'model.safetensors'), device=device)
compatible_state_dict = {k[len('model.'): ]: v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
print(len(missing), len(unexpected))     # 3, 0
print(missing)  # ['cos_cached', 'sin_cached', 'lm_head.weight']

# tied embedding 按说不是应该加载完参数，再在这里 tie embedding weight 吗？但是不 tie，输出结果也是正常的。
# if config.tie_word_embeddings:
#     model.lm_head.weight.copy_(model.embed_tokens.weight)
#     # model.lm_head.weight = model.embed_tokens.weight
#     print(model.lm_head.weight[0, :10])
#     print(model.embed_tokens.weight[0, :10])

tokenizer = AutoTokenizer.from_pretrained(model_root)
prompt = "介绍一下大语言模型。"

messages = [
    {'role': 'system', "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {'role': 'user', "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors='pt').to(device)

input_ids = model_inputs['input_ids']

generated_ids = model.generate(input_ids, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, top_k=None, temperature=0.6) # topk 有点问题

response_ids = generated_ids[0][len(input_ids[0]): ]
response = tokenizer.decode(response_ids, skip_special_tokens=True)

print('Prompt:')
print(prompt)

print('Response:')
print(response)
