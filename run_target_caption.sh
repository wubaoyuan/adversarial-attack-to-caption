
# ./run_target_caption.sh [GPU_ID] [RESULTS_SAVE_DIR] [LOG_DIR] [GEM(0) or SSVM(1)] [STAR_IDX] [END_IDX] \
#                         [ID] [CAPTION_MODEL] [PRETRAINED_MODEL]

# START_IDX and END_IDX mean the index of val_data (0 ~ 1,000)
# CAPTION_MODEL is selected from show_tell, show_attend_tell, att2in2
# PRETRAINED_MODEL: show_tell (data/pretrained_models/st_model-best.pth)
#                   show_attend_tell (data/pretrained_models/sat_model-best.pth)
#                   att2in2 (data/pretrained_models/rl_model-best.pth)
# e.g. ./run_target_caption.sh 0 save_dir/log save_dir/logs/log 1 0 1000 sat show_attend_tell \
#                              data/pretrained_models/sat_model-best.pth


if [ $# -ne 9 ]; then
  echo "invaild parameters..."
  exit 1;
fi

save_dir_name=$(dirname $2)
log_dir_name=$(dirname $3)

if [ ! -d $save_dir_name ];then
    mkdir -p $save_dir_name
fi

if [ ! -d $log_dir_name ];then
    mkdir -p $log_dir_name
fi

if [ $4 -eq 0 ]; then
  echo "using EM algorithm..."
  CUDA_VISIBLE_DEVICES=$1 python2.7 target_caption_attack.py --id $7 \
        --caption_model $8 \
        --pretrain_model $9 \
        --batch_size 10 \
        --val_images_use 5000 \
        --language_eval 1 \
        --save_dir $2"_"$5"_"$6 \
        --start_num $5 \
        --end_num $6 | tee $3"_"$5"_"$6".txt"
elif [ $4 -eq 1 ]; then
  echo "using structural SVM..."
  CUDA_VISIBLE_DEVICES=$1 python2.7 target_caption_attack.py --id $7 \
        --caption_model $8 \
        --pretrain_model $9 \
        --batch_size 10 \
        --val_images_use 5000 \
        --language_eval 1 \
        --save_dir $2"_"$5"_"$6 \
        --is_ssvm \
        --start_num $5 \
        --end_num $6 | tee $3"_"$5"_"$6".txt"
else
  echo "invalid parameter..."
  exit 1;
fi
