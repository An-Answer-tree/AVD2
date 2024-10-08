CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
OMPI_COMM_WORLD_SIZE="4" \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=45946 src/tasks/run_adapt.py \
        --config src/configs/VidSwinBert/BDDX_multi_default.json \
        --train_yaml BDDX/training_32frames.yaml \
        --val_yaml BDDX/testing_32frames.yaml \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 4 \
        --num_train_epochs 49 \
        --learning_rate 0.0001 \
        --max_num_frames 32 \
        --pretrained_2d 0 \
        --backbone_coef_lr 0.05 \
        --mask_prob 0.5 \
        --max_masked_token 125 \
        --zero_opt_stage 1 \
        --mixed_precision_method deepspeed \
        --deepspeed_fp16 \
        --gradient_accumulation_steps 4 \
        --learn_mask_enabled \
        --loss_sparse_w 0.1 \
        --use_sep_cap \
        --multitask \
        --signal_types course speed \
        --loss_sensor_w 0.05 \
        --max_grad_norm 1 \
        --output_dir ./output/multitask/sensor_course_speed