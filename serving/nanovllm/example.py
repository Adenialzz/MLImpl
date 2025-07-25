

from nanovllm import SamplingParams, LLM

from transformers import AutoTokenizer

# 加载、配置LLM
path = "huggingface/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 准备prompt
prompts = [
    "介绍一下 vLLM。"
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    for prompt in prompts
]

# 推理，输出回答
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")

