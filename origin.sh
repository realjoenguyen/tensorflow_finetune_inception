#!/bin/bash
. ./set_path.sh

python origin_retrain.py \
--image_dir ${IMAGE_DIR} \
--how_many_training_steps 8000 \
--train_batch_size 100 \
--validation_batch_size -1 \
--test_batch_size -1 \
--learning_rate 0.05 \
--validation_percentage 1 \
--testing_percentage 1 \
--flip_left_right True \
--random_brightness 10 \
--random_scale 10 \
--print_misclassified_test_images True > log_8000_distort.txt

#--image_dir /home/ta/Projects/brand_safety/crawler/images/finetune_inception/flower_photos \

#python retrain_model_classifier.py 
