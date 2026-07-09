# megabench benchmark run -n nano-gpt -p "--num_steps=100"

wandb online
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export spatial_compress_rate=16
export MAX_FRAMES_1M=9

NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ "$*" == *"--debug"* ]]; then
    ARNOLD_WORKER_NUM=1
    ARNOLD_WORKER_GPU=1
    NUM_WORKERS=0
    # TORCHINDUCTOR_COMPILE_THREADS=1
    # ulimit -n 1024768
    ARNOLD_ID=${ARNOLD_ID:-0}
    ARNOLD_WORKER_0_HOST='localhost'
    # ARNOLD_WORKER_0_PORT=${ARNOLD_WORKER_0_PORT:-'9597'}
    ARNOLD_WORKER_0_PORT=10101
    # NCCL_NVLS_ENABLE=0
    wandb offline
else
    ARNOLD_WORKER_NUM=1
    ARNOLD_WORKER_GPU=8
    NUM_WORKERS=16
    ARNOLD_ID=${ARNOLD_ID:-0}
    ARNOLD_WORKER_0_HOST='localhost'
    ARNOLD_WORKER_0_PORT=9592
    # sudo apt-get install libgl1-mesa-glx -y
fi

port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
echo $port


latent_channels=256
quant_method=hierarchical_binary_quant_round_4
exp_name=${quant_method}_dim${latent_channels}
data_root=[data_root]
username=[username]
# rm -rf ${exp_name}

torchrun \
    --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM --master_addr=$ARNOLD_WORKER_0_HOST \
    --node_rank=$ARNOLD_ID --master_port=$port \
    train.py --num_workers $NUM_WORKERS \
    --tokenizer 'hbq_tokenizer' \
    --resolution 256 --batch_size 2 --dataset_list high-quality-v5 \
    --dataaug "resizecrop" --sequence_length 33 --fps 3 6 8 16 \
    --optim_type AdamW --lr 1e-5 --dis_lr_multiplier 1 --max_steps 60000000 \
    --norm_type rms \
    --disc_layers 3 --activation_in_disc leaky_relu --apply_noise  \
    --l1_weight 1 --perceptual_weight 1 --kl_weight 0 \
    --discriminator_iter_start -1 --image_disc_weight 1.0 --image_gan_weight 0.3 --gan_image4video yes --video_disc_weight 0 --video_gan_weight 0 --remove_disc 0 \
    --disc_temporal_compress no \
    --use_checkpoint --compile no --ema yes \
    --default_root_dir ${exp_name} --log_every 2 --ckpt_every 500 \
    --entropy_loss_weight 0.1 \
    --quantizer_type MultiScaleBSQTP --schedule_mode infinity_video_two_pyramid_full_time_elegant_spatial_down32 \
    --pretrained_mode weights --pretrained_ema yes \
    --codebook_dim_low 4 \
    --use_stochastic_depth --keep_first_quant --drop_rate 0.5 --random_short_schedule --skip_detail_scales_prob -1 \
    --use_multi_scale 1 --quant_not_rely_256 0 \
    --semantic_num_lvl 2 --semantic_scale_dim ${latent_channels} \
    --detail_num_lvl 2 --detail_scale_dim ${latent_channels} \
    --middle_scale_dim 16 \
    --semantic_scales 1000 \
    --use_feat_proj 0 \
    --div_delta_t 0 \
    --multi_scale_freq '' \
    --video_scale_repetition_times 4 \
    --video_scale_repetition_prob 1.0 \
    --dataaug_video "resizecrop" \
    --data_root ${data_root} \
    --username ${username} \
    --train_continuous 0 \
    --elementwise_enlarge_factor 1 \
    --quant_method ${quant_method} \
    --lfq_weight 0 \
    --dec_dim 256 \
    --latent_channels ${latent_channels} \
    --dim_mult 1 2 4 4 \
    --num_res_blocks 2 \
    --temperal_downsample 0 1 1 \
    --dropout_z 0. \
    --quant_unit 256 \
    --remove_enlarge_factors 0 \
    --use_channelwise_std 1 \
    --encoder_out_type feature_tanh \
    --enable_online_download 0 \
    --seed 1938
    "$@"
