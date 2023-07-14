import time
import argparse
from skimage import io
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from data_loader import CervicalDataset, collater,CervicalDataset1
from augmentation import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90, Normalizer, UnNormalizer
from torch.utils.data import Dataset, DataLoader
from util import *
from losses1 import FocalLoss,SupFocalLoss,RankLoss,ContrastLoss
from build_network import build_network, lr_poly
from extract_patch import cropper,croppatch

from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
    ToGray,
    HueSaturationValue,
)
IMG_SHAPE = (224, 224, 3)
def generate_transforms(image_size):

    train_transform = Compose(
        [
            # Resize(height=image_size[0], width=image_size[1]),
            OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            # ShiftScaleRotate(
            #     shift_limit=0.2,
            #     scale_limit=0.2,
            #     rotate_limit=20,
            #     interpolation=cv2.INTER_LINEAR,
            #     border_mode=cv2.BORDER_REFLECT_101,
            #     p=1,
            # ),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    val_transform = Compose(
        [
            Resize(height=image_size[0], width=image_size[1]),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    return {"train_transforms": train_transform, "val_transforms": val_transform}

def load_data1(filename):
    test_data =pd.read_csv(filename)

    return test_data

def prepare_input(image):
    image = image.copy()
    # print('pre_img',image)
    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    # print('_img',image)
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True).cuda()
def generate_transforms_syt(image_size):

    train_transform = Compose(
        [
            # Resize(height=image_size[0], width=image_size[1]),
            OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            ToGray(p=0.5),
        #     HueSaturationValue(        hue_shift_limit=20,
        # sat_shift_limit=30,
        # val_shift_limit=20,
        # p=0.5),
            # OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
            # VerticalFlip(p=0.5),
            # HorizontalFlip(p=0.5),
            # ShiftScaleRotate(
            #     shift_limit=0.2,
            #     scale_limit=0.2,
            #     rotate_limit=20,
            #     interpolation=cv2.INTER_LINEAR,
            #     border_mode=cv2.BORDER_REFLECT_101,
            #     p=1,
            # ),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    val_transform = Compose(
        [
            Resize(height=image_size[0], width=image_size[1]),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    return {"train_transforms": train_transform, "val_transforms": val_transform}
def train_epoch(model, train_loader, criterion, optimizer, epoch, n_epochs, start_lr, lr_power,
                print_freq=1):
    batch_time = AverageMeter()
    cls_loss_all = AverageMeter()
    reg_loss_all = AverageMeter()
    loss_all = AverageMeter()

    model.train()
    end = time.time()
    np.random.seed()
    for idx, data in enumerate(train_loader):
        annotations = data['label'].cuda()
        classification, regression, anchors = model(data['img'].cuda())
        # print(classification.size())
        cls_loss, reg_loss = criterion(classification, regression, anchors, annotations)
        # print(type(cls_loss),type(reg_loss))
        loss = cls_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_loss_all.update(cls_loss.item(), annotations.size(0))
        reg_loss_all.update(reg_loss.item(), annotations.size(0))
        loss_all.update(loss.item(), annotations.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = lr_poly(start_lr, epoch, n_epochs, lr_power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print stats
        if idx % print_freq == 0:
            res = '\t'.join(['Train, Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                             'Iter: [%d/%d]' % (idx + 1, len(train_loader)),
                             'Time: %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                             'Cls_loss %.4f (%.4f)' % (cls_loss_all.val, cls_loss_all.avg),
                             'Reg_loss %.4f (%.4f)' % (reg_loss_all.val, reg_loss_all.avg),
                             'Loss %.4f (%.4f)' % (loss_all.val, loss_all.avg),
                             'Lr %.8f' % lr,
                             ])
            print(res)

        # update learning rate


    return cls_loss_all.avg, reg_loss_all.avg, loss_all.avg



def train_epoch_supress(model_retina,model_refine, train_loader, criterion_retina, criterion_supress,criterion_contrast,optimizer, epoch, n_epochs, start_lr, lr_power,
                print_freq=1):
    batch_time = AverageMeter()
    cls_loss_all = AverageMeter()
    reg_loss_all = AverageMeter()
    rank_loss_all=AverageMeter()
    rank_reg_loss_all=AverageMeter()
    rank_bce_loss_all = AverageMeter()
    rank_con_loss_all = AverageMeter()
    loss_all = AverageMeter()
    model_retina.cuda()
    model_retina.train()
    model_refine.cuda()
    model_refine.eval()
    end = time.time()
    np.random.seed()
    img_Path=args.img_Path
    mkdir(img_Path)

    for idx, data in enumerate(train_loader):
        annotations = Variable(data['label']).cuda()
        classification, regression, anchors,ROI_features = model_retina(data['img'].cuda(),'train')
        target, bbox_id, imgname,patch_result = croppatch(classification, regression, anchors, data['img'].cuda(), annotations, idx,img_Path, data['path'],epoch)
        target=np.array(target)
        imgname=np.array(imgname)
        batch_num=classification.shape[0]
        test_preds = torch.rand([target.shape[0],2])
        img_features = []
        with torch.no_grad():
            for img_id in range(target.shape[0]):

                img = io.imread(img_Path+imgname[img_id])
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                inputs = prepare_input(img)
                test_score,img_feature=model_refine(inputs)
                img_features.append(img_feature)

                test_preds[img_id]=torch.softmax(test_score, dim=1)

        patch_features=torch.cat(img_features)
        target=torch.Tensor(target)
        scores_all1=test_preds

        cls_loss, reg_loss = criterion_retina(classification, regression, anchors, annotations)

        rank_loss,rank_reg_loss=criterion_supress(test_preds,classification,bbox_id,batch_num,regression, anchors, annotations,target,patch_result)
        contrast_loss=criterion_contrast(ROI_features,patch_features)

        loss = cls_loss + reg_loss + rank_loss * 0.5 + rank_reg_loss * 0.25 + contrast_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cls_loss_all.update(cls_loss.item(), annotations.size(0))
        reg_loss_all.update(reg_loss.item(), annotations.size(0))
        rank_reg_loss_all.update(rank_reg_loss.item(),scores_all1.size(0))

        rank_con_loss_all.update(contrast_loss.item(),scores_all1.size(0))
        rank_loss_all.update(rank_loss.item(),scores_all1.size(0))
        loss_all.update(loss.item(), annotations.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = lr_poly(start_lr, epoch, n_epochs, lr_power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print stats
        if idx % print_freq == 0:
            res = '\t'.join(['Train, Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                             'Iter: [%d/%d]' % (idx + 1, len(train_loader)),
                             'Time: %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                             'Cls_loss %.4f (%.4f)' % (cls_loss_all.val, cls_loss_all.avg),
                             'Reg_loss %.4f (%.4f)' % (reg_loss_all.val, reg_loss_all.avg),
                             'Rank_Loss %.4f (%.4f)' % (rank_loss_all.val, rank_loss_all.avg),
                             'Rank_Reg_Loss %.4f (%.4f)' % (rank_reg_loss_all.val, rank_reg_loss_all.avg),
                             'Rank_con_Loss %.4f (%.4f)' % (rank_con_loss_all.val, rank_con_loss_all.avg),
                             'Loss %.4f (%.4f)' % (loss_all.val, loss_all.avg),
                             'Lr %.8f' % lr,
                             ])
            print(res)


    return cls_loss_all.avg, reg_loss_all.avg,rank_loss_all.avg,rank_reg_loss_all.avg,rank_con_loss_all.avg,rank_bce_loss_all.avg,loss_all.avg
from utils_sample_diagnose import *
from sample_diagnose_train import CoolSystem
def train(args):

    hparams = init_hparams()
    Refine_model = CoolSystem(hparams)
    Refine_model.load_state_dict(torch.load('/mnt/data/feimanman/Sample_Test1/classification.ckpt')["state_dict"])
    Refine_model.to("cuda")
    for param in Refine_model.parameters():
        param.requires_grad = False
    Retina_model, start_epoch_retina = build_network(snapshot='/mnt/data/feimanman/global_local_net/checkpoint_fmm_retina/retina_model/model_at_epoch_001.dat', backend=cfg.backend)

    Retina_optimizer = torch.optim.Adam(Retina_model.parameters(), lr=args.start_lr, weight_decay=cfg.weight_decay)
    Supress_criterion=RankLoss()
    Contrast_criterion=ContrastLoss()
    Supretina_criterion=FocalLoss(alpha=cfg.alpha, gamma=cfg.gamma)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_sample_path='/mnt/data/feimanman/npzdata/train/'
    sample_path = [os.path.join(train_sample_path, x) for x in os.listdir(train_sample_path) if '.npz' in x]
    train_data = CervicalDataset(sample_path, cfg.patch_size,
                                 transform=transforms.Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotate90(0.5), Normalizer()]))

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, drop_last=False,
                              collate_fn=collater, num_workers=cfg.num_worker, worker_init_fn=worker_init_fn)
    epochs = cfg.epochs
    i = 0
    for epoch in range(start_epoch_retina, epochs):
        train_cls_loss,train_reg_loss,train_rank_loss,train_rank_reg_loss,train_rank_con_loss,train_rank_bce_loss,train_loss = train_epoch_supress(
            model_retina=Retina_model,
            model_refine=Refine_model,
            train_loader=train_loader,
            criterion_retina=Supretina_criterion,
            criterion_supress=Supress_criterion,
            criterion_contrast=Contrast_criterion,
            optimizer=Retina_optimizer,
            epoch=epoch,
            n_epochs=epochs,
            start_lr=args.start_lr,
            lr_power=cfg.lr_power,
            print_freq=1,
        )
        i += 1

        checkpoint_path=args.checkpoint_path
        log_path = args.log_path
        datpath=save_model(Retina_model, epoch, checkpoint_path)
        mkdir(log_path)

        with open(log_path + 'train_log.csv', 'a') as log_file:
            log_file.write(
                '%03d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f\n' %
                ((epoch + 1),
                 train_cls_loss,
                 train_reg_loss,
                 train_rank_loss,
                 train_rank_reg_loss,
                 train_rank_con_loss,
                 train_rank_bce_loss,
                 train_loss)
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None,
                        type=str, help='snapshot')
    parser.add_argument('--checkpoint_path', default='../checkpoint_fmm_supress_suf_1113_rank_bce_train_all_lr_0.001_1_C4_rank/',
                        type=str, help='snapshot')
    parser.add_argument('--log_path', default='../log_supress_suf_1113_rank_bce_train_all_lr_0.001_C4_rank/',
                        type=str, help='snapshot')
    parser.add_argument('--start_lr', default=0.0001,
                        type=str, help='snapshot')
    parser.add_argument('--img_Path', default='/mnt/data/feimanman/global_local_net/result/train_1113_rank_bce_train_all_C4_rank/',
                        type=str, help='snapshot')

    args = parser.parse_args()
    train(args)
