#!/bin/sh
python train_segmentation.py --tr_img /home/shrayank_mistry/Modules/data/my_data_v2/train_images/ \
--tr_msk /home/shrayank_mistry/Modules/data/my_data_v2/train_masks/ \
--v_img /home/shrayank_mistry/Modules/data/my_data_v2/val_images/ \
--v_msk /home/shrayank_mistry/Modules/data/my_data_v2/val_masks/ \
--ts_img /home/shrayank_mistry/Modules/data/my_data_v2/test_images/ \
--ts_msk /home/shrayank_mistry/Modules/data/my_data_v2/test_masks/ \
--m_dir /home/shrayank_mistry/Modules/model_stats/ \
--bs 2 \
--opt Adam \
--lr 0.001 \
--eps 3 \
--res 256