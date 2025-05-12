
from https://github.com/thomfoster/minRLHF

```bash
cd MLImpl/algorithms/rlhf
python train_ppo.py
```


TODOs

1. 找一个有实际意义的数据集进行训练

2. 支持向 Env 中传入 reward funcs/model

3. 将data(dict) 封装一下 BufferItem、Experience（可参考 OpenRLHF）
