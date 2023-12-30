import argparse
import torch
from torch.utils.data import DataLoader
from data.get_datasets import get_class_splits
from data.get_datasets import get_datasets
from data.augmentations import get_transform
from model import ContrastiveLearningViewGenerator, DINOHead
from train import test

def get_eval_args(dataset = 'cifar100'):
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=dataset, help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # PARSE
    # ----------------------
    args = parser.parse_args('')
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    args.eval_funcs = ['v2']

    return args




def get_eval_data_loaders(dataset = 'cifar100'):
    args = get_eval_args(dataset)

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)


    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                            train_transform,
                                                                                            test_transform,
                                                                                            args)
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    
    return train_loader, test_loader_unlabelled 


def load_projector(ckpt_pth, in_dim=768, out_dim=100, nlayers=3):
    ckpt  = torch.load(ckpt_pth)
    head_state_dict= {x[2:]:ckpt['model'][x] for x in ckpt['model'].keys() if x.startswith('1')}

    projector = DINOHead(in_dim=in_dim, out_dim=out_dim, nlayers=nlayers)
    projector.load_state_dict(head_state_dict)

    return projector

def get_backbone(ckpt_pth):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    ckpt = torch.load(ckpt_pth)
    backbone.load_state_dict(ckpt)

    return backbone

def eval_fusion(backbone_path, head_path):
    args = get_eval_args()
    head = load_projector(head_path)
    backbone = get_backbone(backbone_path)
    model = torch.nn.Sequential(backbone, head).cuda()
    train_loader, test_loader = get_eval_data_loaders()

    all_acc, old_acc, new_acc = test(model, test_loader, 0,'', args)

    return all_acc, old_acc, new_acc
