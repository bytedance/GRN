exp_name=GRN_ind_B
cache_dir=./checkpoints
OUTPUT_DIR=${cache_dir}/${exp_name}

data_path=./data/imagenet/imagenet_train_val
vae_path=./weights/HBQ_image_tokenizer_16dim_M4.ckpt

export EXP_NAME=${exp_name}
export PROJECT="GRN"
mkdir -p ${OUTPUT_DIR}

if [[ "$*" == *"--debug"* ]]; then
    ARNOLD_WORKER_NUM=1
    ARNOLD_WORKER_GPU=8
    NUM_WORKERS=0
    ARNOLD_ID=${ARNOLD_ID:-0}
    ARNOLD_WORKER_0_HOST='localhost'
    ARNOLD_WORKER_0_PORT=10101
    master_port=${ARNOLD_WORKER_0_PORT}
    # wandb offline
    wandb online
else
    MASTER_NODE_ID=0
    nnodes=${ARNOLD_WORKER_NUM}
    node_rank=${ARNOLD_ID}
    master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
    master_addr=${!master_addr}
    master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
    master_port=${!master_port}
    ports=(`echo $master_port | tr ',' ' '`)
    master_port=${ports[0]}
    pip3 install einops==0.8.0
fi

torchrun \
    --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM --master_addr=$ARNOLD_WORKER_0_HOST \
    --node_rank=$ARNOLD_ID --master_port=$master_port \
    c2i_train_infer.py \
    --model GRN_B \
    --proj_dropout 0.0 \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 128 --blr 5e-5 \
    --epochs 600 --warmup_epochs 5 \
    --gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --method GRN_ind \
    --hbq_round 4 \
    --in_channels 256 \
    --sampling_method euler \
    --vae_path ${vae_path} \
    --wandb 1 \
    --data_path ${data_path} \
    --online_eval 0 \
    --lr_schedule constant \
    --min_lr 1e-5 \
    --clip_grad_norm 0.1
