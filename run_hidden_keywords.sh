
# ./run_hidden_keywords.sh [GPU_ID] [RESULTS_SAVE_DIR] [LOG_DIR] [GEM(0) or SSVM(1)] [NUM_HIDDEN] \
#                          [STAR_IDX] [END_IDX] \
#                          [ID] [CAPTION_MODEL] [PRETRAINED_MODEL]

# NUM_HIDDEN: number of hidden places in a sentence, suggest 1, 2, 3
# e.g. ./run_hidden_keywords.sh 0 save_dir/log save_dir/logs/log 0 2 0 1000 st show_tell \
#                               data/pretrained_models/st_model-best.pth


if [ $# -ne 10 ]; then
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
  CUDA_VISIBLE_DEVICES=$1 python2.7 hidden_keywords_attack.py --id $8 \
        --caption_model $9 \
        --pretrain_model ${10} \
        --batch_size 10 \
        --val_images_use 5000 \
        --language_eval 1 \
        --save_dir $2"_"$6"_"$7 \
        --num_hidden $5 \
        --times 1 \
        --start_num $6 \
        --end_num $7 | tee $3"_"$6"_"$7".txt"
elif [ $4 -eq 1 ]; then
  echo "using structural SVM..."
  CUDA_VISIBLE_DEVICES=$1 python2.7 hidden_keywords_attack.py --id $8 \
        --caption_model $9 \
        --pretrain_model ${10} \
        --batch_size 10 \
        --val_images_use 5000 \
        --language_eval 1 \
        --save_dir $2"_"$6"_"$7 \
        --num_hidden $5 \
        --times 1 \
        --is_ssvm \
        --start_num $6 \
        --end_num $7 | tee $3"_"$6"_"$7".txt" 
else
  echo "invalid parameter..."
  exit 1;
fi
