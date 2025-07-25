
from torch import nn, Tensor
import os
from safetensors import safe_open
from glob import glob

def default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', {})
    # packed_modules_mapping 是 vLLM（或类似大模型推理框架）在加载权重时用的一张“映射表”，专门用来解决 “算子融合” 或 “量化权重打包” 带来的“文件里的权重名字 ≠ 模型里的参数名字”的问题。
    for file in glob(os.path.join(path, '*.safetensors')):
        with safe_open(file, 'pt', 'cpu') as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, 'weight_loader')
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break

                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

