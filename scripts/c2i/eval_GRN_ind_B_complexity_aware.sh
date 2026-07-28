cache_dir=./checkpoints
batch_size=64
exp_name=GRN_ind_B
CKPT_DIR=${cache_dir}/checkpoints/${exp_name}
generation_dir=${cache_dir}/generation/${exp_name}
CKPT_FILE='/weights/GRN_ind_B_ep599.pth'
data_path=./data/imagenet/imagenet_train_val
vae_path=./weights/HBQ_image_tokenizer_16dim_M4.ckpt

export EXP_NAME=${exp_name}
export PROJECT="GRN"
export CKPT_FILE=${CKPT_FILE}

for tau in 1.31
do
    for cfg in 2.35
    do
        for interval_min in 0.44
        do
            torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=29502 \
            c2i_train_infer.py \
            --model GRN_B \
            --proj_dropout 0.0 \
            --P_mean -0.8 --P_std 0.8 \
            --img_size 256 --noise_scale 1.0 \
            --batch_size ${batch_size} --blr 5e-5 \
            --epochs 600 --warmup_epochs 5 \
            --gen_bsz ${batch_size} --num_images 50000 --tau ${tau} --cfg ${cfg} --interval_min ${interval_min} --interval_max 1.0 \
            --output_dir ${CKPT_DIR} \
            --method GRN_ind \
            --hbq_round 4 \
            --in_channels 256 \
            --vae_path ${vae_path} \
            --wandb 0 \
            --data_path ${data_path} \
            --resume ${CKPT_FILE} \
            --generation_dir ${generation_dir} \
            --sampling_method complexity_aware \
            --evaluate_gen \
            --complexity_aware_wp 5 \
            --complexity_aware_k 700 \
            --complexity_aware_b -620 \
            --complexity_aware_tmin 20 \
            --num_sampling_steps 50
        done
    done
done
