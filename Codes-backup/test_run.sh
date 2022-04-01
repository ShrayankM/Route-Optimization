#!/bin/sh
python test_segmentation.py --ts_img /home/shrayank_mistry/Modules/data/my_data_v2/test_images/ \
--ts_msk /home/shrayank_mistry/Modules/data/my_data_v2/test_masks/ \
--m_path /home/shrayank_mistry/Modules/model_stats/Unet_resnet50/ \
--bs 1 \
--res 256 
