set -x

qwen3_0_6b_path=""
qwen3_4b_path=""
qwen3_8b_path=""
qwen3_14b_path=""

model_specs=("0_6b" "4b" "8b" "14b")

for ms in "${model_specs[@]}"; do
    key="qwen3_${ms}_path"   # 得到 qwen3_8b_path
    model_path="${!key}"

    echo "Qwen3 $ms nanovllm"
    python bench.py --model_path $model_path --tp 1 --engine nanovllm > nanovllm_$ms.log

    echo "Qwen3 $ms vllm"
    python bench.py --model_path $model_path --tp 1 --engine vllm > vllm_$ms.log

    echo "**********************************"
done
