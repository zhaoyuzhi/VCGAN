import os
import time

base_path = '/media/user/data/VCGAN_ablation_study'
results_480p = ['/media/user/data/VCGAN_ablation_study/ablation_full_result',
                '/media/user/data/VCGAN_ablation_study/ablation_only_two_losses_result',
                '/media/user/data/VCGAN_ablation_study/ablation_only_three_losses_result',
                '/media/user/data/VCGAN_ablation_study/ablation_no_lp_result',
                '/media/user/data/VCGAN_ablation_study/ablation_no_gan_result',
                '/media/user/data/VCGAN_ablation_study/ablation_no_lst_result',
                '/media/user/data/VCGAN_ablation_study/ablation_no_llt_result',
                '/media/user/data/VCGAN_ablation_study/ablation_normal_llt_result',
                '/media/user/data/VCGAN_ablation_study/ablation_no_lo_result',
                ]
dataset = 'videvo'
# test_gpu = '0'

for result in results_480p:

    # cal warp error

    result_path = os.path.join(result, dataset)
    pwcnet_path = './models/pwcNet-default.pytorch'
    report_path = './report_%s.txt' % dataset
    if dataset == 'DAVIS':
        imagelist_txt = './DAVIS_test_imagelist_without_first_frame.txt'
        class_txt = './DAVIS_test_class.txt'

    else:
        imagelist_txt = './videvo_test_imagelist_without_first_frame.txt'
        class_txt = './videvo_test_class.txt'

    os.system(r"python WarpError_by_txt.py --base_path=%s --imagelist_txt=%s --class_txt=%s --pwcnet_path=%s --report_path=%s" % (result_path, imagelist_txt, class_txt, pwcnet_path, report_path))
