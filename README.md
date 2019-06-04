Attack Image Captioning System (accepted by CVPR2019)
====


This repository contains the code for how to exact attack image captioning system introduced in following paper

[Exact Adversarial Attack to Image Captioning via Structured Output Learning with Latent Variables](https://arxiv.org/pdf/1905.04016.pdf)

Yan Xu*, Baoyuan Wu*, Fumin Shen, Yanbo Fan, Yong Zhang, Heng Tao Shen, Wei Liu (* Authors contributed equally)


### Prerequisites ###

This code is implemented based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). (Only Support Pyhton2.7)

1. Clone this repo: `git clone --recurtsive https://github.com/xuyan1115/caption-attack.git`

2. Download the pretrained models (CNN part and RNN part) from [here]() and put them into directory `data/pretained_models/`

3. Download the coco2014 dataset(train and val) from [here](http://cocodataset.org/#download). You should put the folder `train2014/` and `val2014/` to the directory `data/images/`

4. Download the preprocessed COCO captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage and unzip it to directory `data/`

5. Run the following command to filter words and create a vocabulary and discretized caption data, which are dumped into `data/cocotalk.json` and `data/cocotalk_label.h5`, respectively.
  ```
  python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
  ```


### Usage ###

We proposed two attack methods (GEM and SSVM) on three popular image captioning system. (Show-Tell, Show-Attend-Tell, self-critical sequence training(SCST))

1. Run `run_target_caption.sh` for attacking targeted complete captions.
```
./run_target_caption.sh [GPU_ID] [RESULTS_SAVE_DIR] [LOG_DIR] [GEM(0) or SSVM(1)] [STAR_IDX] [END_IDX] \
                        [ID] [CAPTION_MODEL] [PRETRAINED_MODEL]
                        
# START_IDX and END_IDX mean the index of val_data (0 ~ 1,000)
# CAPTION_MODEL is selected from show_tell, show_attend_tell, att2in2
# PRETRAINED_MODEL: show_tell (data/pretrained_models/st_model-best.pth)
#                   show_attend_tell (data/pretrained_models/sat_model-best.pth)
#                   att2in2 (data/pretrained_models/rl_model-best.pth)
# e.g. ./run_target_caption.sh 0 save_dir/log save_dir/logs/log 1 0 1000 sat show_attend_tell \
#                              data/pretrained_models/sat_model-best.pth
```

2. Run `run_hidden_keywords.sh` for attacking targeted partial captions with some specific hidden places.
```
./run_hidden_keywords.sh [GPU_ID] [RESULTS_SAVE_DIR] [LOG_DIR] [GEM(0) or SSVM(1)] [NUM_HIDDEN] \
                        [STAR_IDX] [END_IDX] \
                        [ID] [CAPTION_MODEL] [PRETRAINED_MODEL]
                        
# NUM_HIDDEN: number of hidden places in a sentence, suggest 1, 2, 3
# e.g. ./run_hidden_keywords.sh 0 save_dir/log save_dir/logs/log 0 2 0 1000 st show_tell \
#                              data/pretrained_models/st_model-best.pth
```

3. Run `run_observed_keywords.sh` for attacking targeted partial captions with some specific observed places.
```
./run_observed_keywords.sh [GPU_ID] [RESULTS_SAVE_DIR] [LOG_DIR] [GEM(0) or SSVM(1)] [NUM_OBSERVED] \
                        [STAR_IDX] [END_IDX] \
                        [ID] [CAPTION_MODEL] [PRETRAINED_MODEL]
                        
# NUM_OBSERVED: number of observed places in a sentence, suggest 1, 2, 3
# e.g. ./run_observed_keywords.sh 0 save_dir/log save_dir/logs/log 1 1 0 1000 rl att2in2 \
#                              data/pretrained_models/rl_model-best.pth
```

### Citation ###

If you find our approach is useful in your research, please consider citing:
  
  ```
  @inproceedings{yan2019attack,
  title={Exact Adversarial Attack to Image Captioning via Structured Output Learning with Latent Variables},
  author={Yan Xu and Baoyuan Wu and Fumin Shen and Yanbo Fan and Yong Zhang and Heng Tao Shen and Wei Liu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
  ```
