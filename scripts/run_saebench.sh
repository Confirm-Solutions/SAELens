#!/bin/bash

shopt -s nullglob

PATHS=(/workspace/SAELens/checkpoints/topk-sae-resid-post-160m-1b-sweep/topk_*/final_*)

first=1
for path in "${PATHS[@]}"; do
    echo "launching absorption for $path"
    hook_layer=$(python3 -c "import yaml; print(yaml.safe_load(open('$path/hydra_config.yaml'))['hook_layer'])")
    echo "hook_layer: $hook_layer"
    if [ "$first" -eq 1 ]; then
        ray job submit --entrypoint-num-gpus=2 --no-wait -- python -m sae_bench.evals.absorption.main --local_sae_path="$path" \
            --sae_block_pattern="blocks.$hook_layer.hook_resid_post" \
            --model_name="pythia-160m-deduped" \
            --output_folder="$path/absorption_results"
        first=0
    else
        ray job submit --entrypoint-num-gpus=0.25 --no-wait -- python -m sae_bench.evals.absorption.main --local_sae_path="$path" \
            --sae_block_pattern="blocks.$hook_layer.hook_resid_post" \
            --model_name="pythia-160m-deduped" \
            --output_folder="$path/absorption_results"
    fi
done

# run `vllm serve --model google/gemma-3-12b-it --port 8000` for the server

# export CUDA_VISIBLE_DEVICES=0

# for path in "${PATHS[@]}"; do
#     echo "launching autointerp for $path"
#     hook_layer=$(python3 -c "import yaml; print(yaml.safe_load(open('$path/hydra_config.yaml'))['hook_layer'])")
#     python -m sae_bench.evals.autointerp.main --local_sae_path="$path" \
#         --sae_block_pattern="blocks.$hook_layer.hook_resid_post" \
#         --model_name="pythia-160m-deduped" \
#         --output_folder="$path/autointerp_results" \
#         --local \
#         --llm_api_model="google/gemma-3-12b-it" \
#         --llm_batch_size=512 \
#         --llm_dtype="bfloat16"
# done
