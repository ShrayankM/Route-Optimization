#!/bin/sh
python train_segmentation.py --tr_img /home/shrayank_mistry/Modules/project-mum-pune/data/train_images/ \
--tr_msk /home/shrayank_mistry/Modules/project-mum-pune/data/train_masks/ \
--v_img /home/shrayank_mistry/Modules/project-mum-pune/data/val_images/ \
--v_msk /home/shrayank_mistry/Modules/project-mum-pune/data/val_masks/ \
--ts_img /home/shrayank_mistry/Modules/project-mum-pune/data/test_images/ \
--ts_msk /home/shrayank_mistry/Modules/project-mum-pune/data/test_masks/ \
--m_dir /home/shrayank_mistry/Modules/model_stats/ \
--bs 16 \
--opt Adam \
--lr 0.001 \
--eps 3 \
--res 64