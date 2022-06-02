#!/bin/sh
python train_segmentation.py --tr_img data/train_images/ \
--tr_msk data/train_masks/ \
--v_img data/val_images/ \
--v_msk data/val_masks/ \
--ts_img data/test_images/ \
--ts_msk data/test_masks/ \
--m_dir model_stats/ \
--bs 16 \
--opt Adam \
--lr 0.001 \
--eps 3 \
--res 64
