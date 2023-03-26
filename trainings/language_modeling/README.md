# language modeling training code

- from https://github.com/karpathy/nanoGPT
- gpt2 model gpt.py is soft link of ../../models/gpt/gpt.py

## quick start

**prepare data**

```shell
python data/shakespeare/prepare.py
```

**start triaining**

single gpu

```shell
python main.py --data_dir data/shakespeare
```

multi gpu

```shell
torchrun --standalone --nproc_per_node=8 main.py --data_dir data/shakespeare
```

more details please refer to https://github.com/karpathy/nanoGPT



