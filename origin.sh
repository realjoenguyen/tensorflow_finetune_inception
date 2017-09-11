
#!/bin/bash
python origin_retrain.py \
--image_dir /home/ta/Projects/brand_safety/crawler/images/data/total_images/ \
--how_many_training_steps 100000 \
--train_batch_size 10 \
--validation_batch_size 10 \
--test_batch_size 10 \
--learning_rate 0.05 \
--validation_percentage 1 \
--testing_percentage 15

#--image_dir /home/ta/Projects/brand_safety/crawler/images/finetune_inception/flower_photos \

#python retrain_model_classifier.py 
