#!/usr/bin/env bash

set -x

# set dist args
nproc_per_node=8

if [ ! -z "$SINGLE" ] && [ "$SINGLE" == "1" ]; then
  echo "[single node alone] local, one gpu"
  nnodes=1
  node_rank=0
  nproc_per_node=1
  master_addr=127.0.0.1
  master_port=12344
  wandb offline
  fsdp_warp_mode='trans_block'
  restrict_data_size=2000
  enable_hybrid_shard=0
elif [ ! -z "$SINGLE" ] && [ "$SINGLE" == "2" ]; then
  echo "[single node alone] local, multiple gpus"
  nnodes=1
  node_rank=0
  nproc_per_node=8
  master_addr=127.0.0.1
  master_port=12345
  wandb online
  # wandb offline
  fsdp_warp_mode='full'
  restrict_data_size=10000
  enable_hybrid_shard=0
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
  wandb online
  # wandb offline
  fsdp_warp_mode='full'
  restrict_data_size=-1
  enable_hybrid_shard=1
fi

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=infinity/models/:$PYTHONPATH

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"
git config --global --add safe.directory "$(dirname "$(dirname "$(readlink -f "$0")")")"

exp_name=GRN_T2IV_2B
cache_dir=/dev/shm
text_encoder_path=./weights/umt5-xxl
pn_list="[\"0.41M\"]" # 0.06M is 256x256 / 352x192; 0.41M is 640x640 / 848x480; 0.91M is 960x960 / 1280x720; 1M is 1024x1024;
pn_probs="[1.0]" # sample probs for each resolution
meta_folders="[\"data/toy_data/jsonls\"]"
meta_folder_repeats="[1]"
meta_folder_identifiers="[\"\"]"
vae_path=./weights/HBQ_tokenizer_64dim_M4.ckpt
token_cache_dir=./tmp/token_cache
bed_path=checkpoints
LOCAL_OUT=checkpoints
mkdir -p $LOCAL_OUT
bed_path=${bed_path}/${exp_name}/
Ct5=4096
t5_path=${text_encoder_path}
vae_latent_dim=64
hbq_round=4
num_lvl=2
vae_encoder_out_type='feature_tanh'
num_of_label_value=${num_lvl}
video_scale_repetition='[54]'
video_scale_probs='[1]'
refine_mode='ar_discrete_GRN_bit'
train_h_div_w_list='[]'
alpha=1001
local_out_path=$LOCAL_OUT/${exp_name}
video_fps=16
video_frames=209
i2v_ratio=0.

rm -rf ${local_out_path}
rm -rf ${bed_path}

sudo apt-get install libgl1-mesa-glx -y

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=$((master_port - 3)) \
t2iv_train.py \
--ep=300 \
--opt=adamw \
--ada=0.9_0.999 \
--sche=lin0 \
--fp16=2 \
--tclip=0.1 \
--flash=0 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--bed=${bed_path} \
--meta_folders="${meta_folders}" \
--meta_folder_repeats="${meta_folder_repeats}" \
--meta_folder_identifiers="${meta_folder_identifiers}" \
--exp_name=${exp_name} \
--tlr=2e-4 \
--twd=0.01 \
--pn_list="${pn_list}" \
--pn_probs="${pn_probs}" \
--model=GRN0b \
--video_fps=${video_fps} \
--video_frames=${video_frames} \
--workers=0 \
--hdfs_mode='only' \
--short_cap_prob 0.0 \
--drop_condition_prob 0.1 \
--online_t5=1 \
--use_streaming_dataset 1 \
--Ct5=${Ct5} \
--t5_path=${t5_path} \
--vae_latent_dim=${vae_latent_dim} \
--vae_path=${vae_path} \
--hbq_round=${hbq_round} \
--wp 0.001 \
--wpe=1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 0.5 \
--rope_type='3d' \
--rope2d_normalized_by_hw 2 \
--always_training_scales 1000 \
--num_of_label_value=${num_of_label_value} \
--zero=3 \
--fsdp_warp_mode=${fsdp_warp_mode} \
--save_model_iters_freq 100 \
--log_freq=1 \
--project_name=magic_dc \
--checkpoint_type='torch' \
--noise_apply_layers=1000 \
--apply_spatial_patchify=0 \
--dynamic_scale_schedule=GRN_vae_stride16 \
--use_slice=1 \
--use_flex_attn=True \
--pad=128 \
--use_vae_token_cache=0 \
--save_vae_token_cache=0 \
--allow_online_vae_feature_extraction=1 \
--cache_check_mode=0 \
--mask_video_first_frame=0 \
--down_size_limit=100 \
--casual_up_repeat=0 \
--firstframe_quantized_padding_type=zero \
--token_cache_dir=${token_cache_dir} \
--video_caption_type=tarsier2_caption \
--learn_residual=0 \
--tlen 512 \
--min_scale_ind=3 \
--duration_resolution=0.25 \
--seq_pack_bucket=1000 \
--drop_long_video=1 \
--image_scale_repetition=${video_scale_repetition} \
--video_scale_repetition=${video_scale_repetition} \
--video_scale_probs=${video_scale_probs} \
--append_duration2caption=0 \
--wp_it=50 \
--simple_text_proj=1 \
--min_video_frames=-1 \
--fsdp_save_flatten_model=1 \
--restrict_data_size=${restrict_data_size} \
--allow_less_one_elem_in_seq=0 \
--allow_neglect_largest_scale=0 \
--train_max_token_len=40960 \
--semantic_num_lvl=${num_lvl} \
--semantic_scale_dim=${vae_latent_dim} \
--detail_num_lvl=${num_lvl} \
--detail_scale_dim=${vae_latent_dim} \
--use_fsq_cls_head=1 \
--use_feat_proj=0 \
--use_clipwise_caption=0 \
--use_ada_layer_norm=0 \
--use_fsdp_model_ema 1 \
--model_ema_decay 0.999 \
--add_scale_token 1 \
--add_class_token 0 \
--vae_encoder_out_type=${vae_encoder_out_type} \
--loop_data_per_epoch=1 \
--refine_mode=${refine_mode} \
--alpha=${alpha} \
--train_h_div_w_list=${train_h_div_w_list} \
--pred_flip_label=0 \
--log_norm_mean=-1.3 \
--log_norm_sigma=1.3 \
--i2v_ratio=${i2v_ratio} \
--dense_ratio4seqpack=0.4 \
--enable_hybrid_shard=${enable_hybrid_shard} \
--inner_shard_degree=${nproc_per_node} \
--rush_resume=[pretrained_GRN_model_path]
