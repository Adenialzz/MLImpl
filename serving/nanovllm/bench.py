import time
from random import randint, seed
from typing import Literal


def main(
    model_path: str,
    engine: Literal['vllm', 'nanovllm'],
    tp: int = 1,
    repeat: int = 20,
):

    if engine == 'nanovllm':
        from nanovllm import LLM, SamplingParams
    elif engine == 'vllm':
        from vllm import LLM, SamplingParams
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    llm = LLM(model_path, enforce_eager=False, max_model_len=4096, tensor_parallel_size=tp)
    total_tokens_list, time_list, throughput_list = [], [], []
    for idx in range(repeat):
        # 生成随机输入
        seed(idx)
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
        if engine == 'vllm':
            prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

        llm.generate(["Benchmark: "], SamplingParams())    # 这一行是干嘛的？ 额外的 warmup ？

        # 生成
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = (time.time() - t)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t

        total_tokens_list.append(total_tokens)
        time_list.append(t)
        throughput_list.append(throughput)
        print(f"[{engine}] {idx+1}/{repeat}, Total: {total_tokens}tok Time: {t:.2f}s Throughput: {throughput:.2f}tok/s")

    print(
        f"[{engine}] Total: "
        f"{sum(total_tokens_list)/repeat:.2f}tok "
        f"Time: {sum(time_list)/repeat:.2f}s ",
        f"Throughput: {sum(throughput_list)/repeat:.2f}tok/s"
    )


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
