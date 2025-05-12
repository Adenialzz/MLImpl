import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .utils import logical_or_without_broadcasting

class Actor:
    # actor model 和 reference model 都用这个
    def __init__(
        self,
        model: nn.Module,      # need `.forward` and `.generate` is implemented
        pad_token_id: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        max_length: int = 1024
    ):
        self.model = model
        self.pad_token_id = pad_token_id
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_length = max_length


    def get_rollouts(self, input_ids, input_mask):
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=input_mask,
                pad_token_id=self.pad_token_id,
                do_sample=self.do_sample,
                max_length=self.max_length
            )

        # pad output_ids to be (batch, max_length) in case all completions stopped early
        # 如果所有回答都早停了（eos，没走到 max_length），将 output_ids pad 成 (batch, max_length) 的形状
        pad_n = self.max_length - output_ids.shape[1]
        output_ids = F.pad(output_ids, (0, pad_n), value=self.pad_token_id)

        # Generate output_mask procedurally (must be a vectorised way)
        # 生成 output_mask
        output_mask = torch.zeros_like(output_ids)
        
        start_idx = input_ids.shape[1]
        for i in range(output_ids.shape[0]):
            for j in range(start_idx, output_ids.shape[1]):  # 每条回答从回答的第一个 token 开始看
                if output_ids[i, j] != self.pad_token_id:
                    output_mask[i, j] = 1
                else:
                    output_mask[i, j] = 1   # 到 pad_token_id 了，说明是最后一个生成的 token，对应 mask 标成 1 之后 break，后面都是 0
                    break

        return output_ids, output_mask

    def get_logits(self, output_ids, input_mask, output_mask):
        """
        Inputs: output_ids: Tensor(batch, output_seq), input_mask: Tensor(batch, input_seq), output_mask: Tensor(batch, output_seq)
        Outputs: logprobs: Tensor(batch, output_seq), logprobs_mask: Tensor(batch, output_seq)
        """

        # Pad input mask to same size as output_mask and OR
        # OR'd mask is now all non padding / eos_tokens

        # 把 input_mask 填充成 output_mask 的形状，并执行逻辑或。得到的 mask 作为 attention_mask，可以计算所有回答 token 的 logits
        ord_mask = logical_or_without_broadcasting(input_mask, output_mask)

        # 执行 model.forward 获取 logits
        outputs = self.model(output_ids, attention_mask=ord_mask)

        # logits[i, j, k] = outputs.logits[i, j-1, k] for j>0
        # logits[i, 0, k] = 1e10 for k = output_ids[i, 0]
        # logits[i, 0, k] = 0 otherwise

        logits = torch.empty_like(outputs.logits)
        logits[:, 1:, :] = outputs.logits[:, :-1, :]      # 错开一位，就是输入、输出
        logits[
            list(range(output_ids.shape[0])),
            0,
            output_ids[:, 0]
        ] = 1e10

        # Collapse logits final dim from vocab -> 1 by
        # selecting the logit for the token we used
        pi = Categorical(logits=logits)
        logprobs = pi.log_prob(output_ids)

        # logprobs[i, j] is now the log prob of generating token output_ids[i,j]
        # logprobs[:, 0] = 0 and shouldn't be used
        # output_mask gets around this and the "input_ids"
        # using logprobs.masked_select(output_mask.to(torch.bool)) will select all valid logprobs (and flatten to avoid being a ragged tensor)

        return logprobs, pi


    def to(self, device):
        self.model.to(device)

    @property
    def device(self):
        return self.model.device