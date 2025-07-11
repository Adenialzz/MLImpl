from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLConfig
from qwen_vl_utils import process_vision_info
from qwen25_vl import Qwen2_5_VL


model_path = 'hf_models/qwen25_vl_3b/'
device = 'cuda'

processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

img_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'

messages = [{
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": img_url
        },
        {
            "type": "text",
            "text": "描述这张图片的内容。"
            # "text": "图中的有几个人？"
        }
    ]
}]


text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to(device)


# for k, v in inputs.items():
#     print(k, v.shape)


config: Qwen2_5_VLConfig = Qwen2_5_VLConfig.from_pretrained(model_path)

qwen25vl = Qwen2_5_VL(model_path, device)

generated_ids = qwen25vl.generate(
    inputs['input_ids'],
    inputs['pixel_values'],
    inputs['image_grid_thw'],
    max_new_tokens=512,
    eos_token_id=processor.tokenizer.eos_token_id,
    temperature=0.01,
    top_k=None
)
response_ids = generated_ids[0][len(inputs['input_ids'][0]): ]
response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

print('Response:')
print(response)

