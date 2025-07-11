from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

model_path = 'hf_models/qwen25_vl_3b'
device = 'cuda'

processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).to(device)

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
        }
    ]
}]


text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to(device)

for _ in range(10):
    generated_ids = model.generate(**inputs, temperature=0.01, max_new_tokens=512)[0].tolist()[len(inputs['input_ids'][0]): ]
    print(processor.decode(generated_ids))

