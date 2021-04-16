refpath=/mnt/lustre/zhaoyuzhi/dataset/DAVIS_videvo_test

basepath1=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l110_per10_gan1_short3_long5_lr5e-5
basepath2=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l110_per5_gan1_short3_long10_lr5e-5
basepath3=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l110_per5_gan1_short6_long10_lr5e-5
basepath4=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l110_per5_gan1_short6_long5_lr5e-5
basepath5=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l110_per5_gan2_short3_long5_lr5e-5
basepath6=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l11_per1_gan1_short1_long1_lr5e-5
basepath7=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l120_per10_gan2_short3_long5_lr5e-5
basepath8=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train/ablation_study_loss_parameter/models_l120_per5_gan1_short3_long5_lr5e-5

basepath9=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train_without_global_feature_extractor/ablation_study_architecture/without_global_feature_extractor
basepath10=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train_without_placeholder_feature_extractor/ablation_study_architecture/without_placeholder_feature_extractor
basepath11=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train_without_both_feature_extractors/ablation_study_architecture/without_both_feature_extractors

basepath12=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/ChromaGAN
basepath13=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/ChromaGAN+BVTC
basepath14=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/3DVC
basepath15=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/SCGAN
basepath16=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/SCGAN+BVTC

basepath17=/mnt/lustre/zhaoyuzhi/ieeetmmvcgan/train_with_normal_long_term_loss/results_l1_lp_lg_lt

python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath1}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath2}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath3}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath4}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath5}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath6}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath7}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath8}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath9}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath10}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath11}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath12}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath13}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath14}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath15}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath16}
python NRMSE_PSNR_SSIM_bylistname_align_DAVIS.py --refpath ${refpath} --basepath ${basepath17}
