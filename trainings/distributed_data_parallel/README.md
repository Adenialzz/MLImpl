# distributed data parallel

## example
- image classification as example
- engine.py is soft link with ../image_classification/engine.py
- main.py is minimum modification from ../image_classification/main.py

## Refs
- Chinese Blog Ref: https://blog.csdn.net/weixin_44966641/article/details/121872773
- Code Ref: https://github.com/karpathy/nanoGPT/blob/master/train.py

## test

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
```shell
python main.py --batch_size=32
```

you can also run the script on single GPU with ddp manner:

```shell
torchrun --standalone --nproc_per_node=1 main.py
```

To run with DDP on 4 gpus on 1 node, example:

```shell
torchrun --standalone --nproc_per_node=4 main.py
```

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:

  ```shell
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 main.py
  ```

- Run on the worker node:

  ```shell
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 main.py
  ```

  (If your cluster does not have Infiniband interconnect prepend `NCCL_IB_DISABLE=1`)
