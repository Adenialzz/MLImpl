import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLConfig
from qwen_vl_utils import process_vision_info
from vit import VisionTransformer
from glob import glob
from qwen25.qwen import Qwen2_5
from qwen25.gqa import CausalGroupQueryAttention
from qwen25.rope import MultimodalRotaryPositionEmbedding, apply_multimodal_rotary_pos_emb
from safetensors.torch import load_file as load_safetensors
from typing import Optional
from functools import partial


def set_mrope_func(model: nn.Module, func):
    for child in model.children():
        if isinstance(child, CausalGroupQueryAttention):
            child: CausalGroupQueryAttention
            child.rop_apply_func = func
        else:
            set_mrope_func(child, func)

class Qwen2_5_VL_LanguageModel(Qwen2_5):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(**config.to_dict())
        self.config = config
        self.rot_pos_emb = MultimodalRotaryPositionEmbedding(
            base=config.rope_theta,
            rope_dim=config.text_config.hidden_size//config.text_config.num_attention_heads,
            max_position_embeddings=None      # 没用到 max pos emb
        )

        mrope_func = partial(apply_multimodal_rotary_pos_emb, mrope_section=config.rope_scaling['mrope_section'])
        set_mrope_func(self, mrope_func)
        
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    def forward(self, input_ids, input_embeds, image_grid_thw):

        if input_ids is not None and input_embeds is None:
            x = self.embed_tokens(input_ids)        # (B, T, D)
        else:
            x = input_embeds

        position_ids, mrope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=None
        )
        cos, sin = self.rot_pos_emb(position_ids)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits



class Qwen2_5_VL(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        config: Qwen2_5_VLConfig = Qwen2_5_VLConfig.from_pretrained(model_path)

        self.config = config
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.block_size = config.max_position_embeddings

        with torch.device(device):
            self.vit: VisionTransformer = VisionTransformer(**config.vision_config.to_dict())
            self.language_model: Qwen2_5_VL_LanguageModel = Qwen2_5_VL_LanguageModel(config)

        state_dict = {}
        state_dict_files = glob(f"{model_path}/*.safetensors")

        for f in state_dict_files:
            print(f"loading model shard {os.path.basename(f)}")
            sd = load_safetensors(f, device=device)
            state_dict.update(sd)

        vit_state_dict = {k[len('visual.'): ]: v for k, v in state_dict.items() if k.startswith('visual.')}
        lm_state_dict = {k[len('model.'): ]: v for k, v in state_dict.items() if k.startswith('model.')}

        self.vit.load_state_dict(vit_state_dict)
        missing, unexpected = self.language_model.load_state_dict(lm_state_dict, strict=False)
        print(missing, unexpected)  # missing ("cos_cached", "sin_cached", "lm_head.weight")

    def get_image_features(self, pixel_vlaues, image_grid_thw):
        '''
        return tuple of tensors
        length == num_images
        '''
        pixel_vlaues = pixel_vlaues.type(self.vit.patch_embed.proj.weight.dtype)
        image_embeds = self.vit(pixel_vlaues, image_grid_thw)

        # image_embeds = torch.load('vv_image_embeds.pt').to(pixel_vlaues.dtype)

        return image_embeds

    def forward(self, input_ids, pixel_values, image_grid_thw):
        # 这个应该只在 prefill 阶段搞一次吧 text/image -> token
        text_embeds = self.language_model.embed_tokens(input_ids)
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        # image_embeds = torch.cat(image_embeds, dim=0)

        image_mask = input_ids == self.image_token_id

        num_image_tokens = image_mask.sum()
        num_image_features = image_embeds.shape[0]

        assert num_image_tokens == num_image_features, f"num_image_tokens: {num_image_tokens} not equal to num_image_features: {num_image_features}"
        image_mask = image_mask.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
        input_embeds = text_embeds.masked_scatter(image_mask, image_embeds)

        logits = self.language_model(
            input_ids=input_ids,      # 用于计算 MRoPE
            input_embeds=input_embeds,
            image_grid_thw=image_grid_thw  # 计算 MRoPE
        )
        return logits

    @torch.no_grad()
    def generate(self, input_ids, pixel_values, grid_thw, max_new_tokens, eos_token_id=None, temperature=1.0, top_k=50):
        idx = input_ids

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size: ]    # 截取最近的 max_pos_ids(block_size) 个 token
            logits = self(idx_cond, pixel_values, grid_thw)
            logits = logits[:, -1, :] / temperature     # last hidden state

            if top_k is not None:
                v = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)

            if eos_token_id is not None:
                if idx_next == eos_token_id:
                    print('end due to eos token')
                    break

        return idx

