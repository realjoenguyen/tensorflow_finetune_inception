
#!/bin/bash
python retrain_official.py \
--train_dir /home/ta/Projects/brand_safety/crawler/images/data/train/ \
--dev_dir /home/ta/Projects/brand_safety/crawler/images/data/dev/ \
--test_dir /home/ta/Projects/brand_safety/crawler/images/data/test/ \
--how_many_epochs 10 \
--train_batch_size 10 \
--validation_batch_size 10 \
--test_batch_size 10 \
--learning_rate 0.05

#--image_dir /home/ta/Projects/brand_safety/crawler/images/finetune_inception/flower_photos \

#python retrain_model_classifier.py 
