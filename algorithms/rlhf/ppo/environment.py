import torch
from .utils import logical_or_without_broadcasting

class RLHFEnvironment:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_input_prompt(self):
        raise NotImplementedError

    def score_generation(self, text: str):
        raise NotImplementedError

    def sample_prompts(self, batch_size: int):
        batch_prompt = [self.get_input_prompt() for _ in range(batch_size)]
        inputs = self.tokenizer(batch_prompt, truncation=True, padding=True, return_tensors='pt')
        return inputs.input_ids, inputs.attention_mask

    def get_rewards(self, output_ids: torch.Tensor, input_mask: torch.Tensor, output_mask: torch.Tensor):
        full_mask = logical_or_without_broadcasting(input_mask, output_mask)
        texts = []

        for output_id, mask in zip(output_ids, full_mask):
            ids = output_id.masked_select(mask.to(torch.bool))
            texts.append(self.tokenizer.decode(ids, skip_special_token=True))
        
        scores = [self.score_generation(text) for text in texts]

        # Rewards[i, j] = 
        #   reward score    if j is last generated token of ith example
        #   0               otherwise

        idxs = -output_mask.flip(-1).argmax(-1) - 1 # index of last generated token # TODO: Make less cryptic
        rewards = torch.zeros_like(output_mask, dtype=torch.float32)
        rewards[list(range(rewards.shape[0])), idxs] = torch.as_tensor(scores)

        return rewards






