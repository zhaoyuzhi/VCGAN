python train.py \
--pre_train True \
--save_mode 'epoch' \
--save_by_epoch 1 \
--save_by_iter 10000 \
--save_path './models' \
--sample_path './samples' \
--load_name './trained_models/First_Stage_final.pth' \
--feature_extractor_path './trained_models/resnet50_fc_in_epoch150_bs256.pth' \
--pwcnet_path './trained_models/pwcNet-default.pytorch' \
--perceptual_path './trained_models/vgg16_pretrained.pth' \
--video_class_txt './txt/DAVIS_videvo_train_class.txt' \
--video_imagelist_txt './txt/DAVIS_videvo_train_imagelist.txt' \
--multi_gpu True \
--cudnn_benchmark True \
--epochs 101 \
--batch_size 16 \
--lr_g 1e-4 \
--lr_d 1e-4 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0 \
--lr_decrease_mode 'epoch' \
--lr_decrease_epoch 10 \
--lr_decrease_iter 100000 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--gan_mode 'no' \
--lambda_l1 10 \
--lambda_percep 5 \
--lambda_gan 1 \
--lambda_flow 0 \
--lambda_flow_short 3 \
--lambda_flow_long 5 \
--mask_para 50 \
--lambda_gp 10 \
--pad 'reflect' \
--activ_g 'lrelu' \
--activ_d 'lrelu' \
--norm 'in' \
--init_type 'xavier' \
--init_gain 0.02 \
--baseroot '/mnt/lustre/zhaoyuzhi/dataset/ILSVRC2012_train_256' \
--iter_frames 5 \
--sample_size 1 \
--crop_size 256 \
--crop_size_h 256 \
--crop_size_w 448 \
--geometry_aug False \
--angle_aug False \
--scale_min 1 \
--scale_max 1 \