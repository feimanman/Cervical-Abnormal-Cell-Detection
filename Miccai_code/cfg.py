batch_size =30
patch_size = [1024, 1024]
epochs = 500
#6 0.001-0.0005
start_lr = 0.0001
lr_power = 0.9
weight_decay = 0.0001
num_worker = 0

alpha = 0.25
gamma = 2
patch_num=5

backend = "retinanet"

# data_path = "../../train/"
# train_sample_path = '../../data/t/'
# train_sample_path = '/mnt/data/feimanman/npzdata/train_quda_fuhe/'
train_sample_path = '/mnt/data/feimanman/npzdata/train/'
# train_sample_path='/mnt/nas/data/20192020_3D_train_17334/'
val_img_path = "../data/val/"
val_gt_json_path = "../data/val_gt/"
val_predict_json_path = '../json/val100/'
val_predict_json_path2 = '../json/val/'



visual_sample_path = ""  # change to validation sample path (including .npz files)
# checkpoint_path = "../checkpoint_resize/"
checkpoint_path = '../checkpoint_fmm_refine_1/'
# checkpoint_path = "../c/"
log_path = '../log_refine_1/'
result_path = "../result/"

