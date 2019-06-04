
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
