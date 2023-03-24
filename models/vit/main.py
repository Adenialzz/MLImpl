import torch
from vision_transformer import VisionTransformer

vit = VisionTransformer()

x = torch.randn(4, 3, 224, 224)
out = vit(x)
print(out.shape)
