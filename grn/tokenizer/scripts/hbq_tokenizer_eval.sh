export spatial_compress_rate=16

for iters in 124999
do
    for return_res in 0.06M
    do
        for latent_channels in 256
        do
            video_scale_repetition=50
            ema=yes
            num_lvl=3
            test_type=detail
            quant_method=hierarchical_binary_quant_round_4
            vqgan_ckpt=/mnt/bn/foundation-ads/hanjian.thu123/VideoVAE/hj_video_vae_results/loom_vae_imagenet_dim_256_hierarchical_binary_quant_round_4_gan_loss_image_gan4video_d1p0_g0p3/checkpoints/model_step_${iters}.ckpt

            if [ "$return_res" = "0.40M" ]; then
                h=480
                w=864
            elif [ "$return_res" = "raw" ]; then
                h=480
                w=864
            elif [ "$return_res" = "0.06M" ]; then
                h=256
                w=256
            elif [ "$return_res" = "0.25M" ]; then
                h=384
                w=688
            elif [ "$return_res" = "1M" ]; then
                h=768
                w=1360
            else
                echo "wrong return_res"
                h=0
                w=0
            fi

            exp_name=$(echo "$vqgan_ckpt" | rev | cut -d'/' -f3 | rev)
            save_dir=results/${exp_name}/checkpoints/h${h}_w${w}/model_step_${iters}_256_${return_res}__ema_${ema}_${test_type}
            echo ${vqgan_ckpt}
            echo ${save_dir}

            rm -rf ${save_dir}
            mkdir -p ${save_dir}
            log_file=${save_dir}/log.txt

            python3 sample.py \
            --inference_type 'image' \
            --dataaug "resizecrop" \
             --default_root_dir test \
            --dataset_list imagenet --resolution ${h} ${w} \
            --sequence_length 81 \
            --ema ${ema} --intermediate_tensor \
            --schedule_mode infinity_video_two_pyramid_full_time_elegant --save_prediction --codebook_dim_low 4 \
            --skip_detail_scales_prob -1 \
            --semantic_scales 1000 --return_res ${return_res} --use_multi_scale 1 \
            --vqgan_ckpt ${vqgan_ckpt} \
            --video_scale_repetition_times ${video_scale_repetition} \
            --video_scale_repetition_prob 1.0 \
            --semantic_num_lvl ${num_lvl} --semantic_scale_dim ${latent_channels} \
            --detail_num_lvl ${num_lvl} --detail_scale_dim ${latent_channels} \
            --middle_scale_dim 16 \
            --quant_not_rely_256 0 \
            --use_feat_proj 0 \
            --div_delta_t 0 \
            --data_root sg \
            --cal_norm \
            --train_continuous 0 \
            --test_type ${test_type} \
            --elementwise_enlarge_factor 1 \
            --quant_method ${quant_method} \
            --quant_unit 256 \
            --normed_tgt_std -1 \
            --encoder_out_type feature_tanh \
            --tokenizer 'hbq_tokenizer' \
            --dec_dim 256 \
            --latent_channels ${latent_channels} \
            --dim_mult 1 2 4 4 \
            --num_res_blocks 2 \
            --temperal_downsample 0 1 1 \
            --dropout_z 0. \
            --debug \
            --batch_size 32 \
            --save_dir ${save_dir} 2>&1 | tee ${log_file}
        done
    done
done
