import atexit
from dataclasses import fields
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from tqdm import tqdm
import time

from ..config import Config
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence
from ..sampling_params import SamplingParams

class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fileds = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []

        ctx = mp.get_context('spawn')
        # model_runner 是主从架构，相当于实现了一个简单的 RPC 框架
        # rank0 （即 self.model_runner）负责接收函数名和参数，将函数名和参数写入共享内存。当然，rank0 本身也执行一部分模型推理。并且，采样 sample 也是由 rank0 执行
        # rank>0 （即新建的多个 Process）从共享内容读取函数名和参数，并进行实际模型推理
        # 每个 rank 负责张量并行的一部分
        for i in range(1, config.tensor_parallel_size):    # 初始化 rank>0 的 modelrunner，是实际的推理进程
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))

            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)   # 初始化 rank0

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call('exit')
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call('run', seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True
    ):
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}

        prefill_throughput, decode_throughput = 0, 0

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc='Generating', dynamic_ncols=True)
        while not self.is_finished():
            t = time.perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:    # 用正负标识 prefill 还是 decode，:)
                    prefill_throughput = num_tokens / (time.perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (time.perf_counter() - t)

                pbar.set_postfix({
                    "prefill": f"{int(prefill_throughput)}tok/s",
                    'decode': f"{int(decode_throughput)}/tok/s"
                })

                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{'text': self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()

        return outputs
