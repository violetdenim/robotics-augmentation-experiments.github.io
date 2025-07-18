
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
    export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
    export NUM_GPU="${NUM_GPU:=1}"
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py         --checkpoint_dir $CHECKPOINT_DIR         --video_save_folder 2024_08_15__19_55_49_18000_30_LA01/output         --controlnet_specs 2024_08_15__19_55_49_18000_30_LA01/spec.json         --offload_text_encoder_model         --offload_guardrail_models         --offload_prompt_upsampler         --num_gpus $NUM_GPU

