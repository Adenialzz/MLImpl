

手写 Qwen 2.5 VL 可加载开源权重推理


- Qwen 2.5 纯语言模型。本身也可以加载开源权重并推理，[见这里](./qwen25/)

- Qwen 2.5 VL ViT。实现了 window attention 和 2D RoPE

- Qwen 2.5 VL 语言模型。整体继承自 Qwen 2.5 纯语言模型，更换其 1D RoPE 为 3D Multimodal RoPE

- Qwen 2.5 VL 本身支持视频输入，本项目简单起见没有实现，只支持了图像输入。有兴趣可以进一步补充。

Run ！

```bash
python main.py
```
