import os
import pdb
import torch
import numpy as np


def get_masked_input(x, tensor_x, args, device):
    if len(x.shape) == 2:
        B, T = x.shape
        mask = torch.rand((B, T)).to(device)
    elif len(x.shape) == 3:
        B, N, T = x.shape
        mask = torch.rand((B, N, T)).to(device)

    if args.mask_type == 'random':
        mask[mask <= args.mask_ratio] = 0
        mask[mask > args.mask_ratio] = 1
    elif args.mask_type == 'end':
        if len(x.shape) == 2:
            mask[:, int(T * args.mask_ratio):] = 0
            mask[:, :int(T * args.mask_ratio)] = 1
        elif len(x.shape) == 3:
            mask[:, :, int(T * args.mask_ratio):] = 0
            mask[:, :, :int(T * args.mask_ratio)] = 1
    elif args.mask_type == 'end_single':
        if len(x.shape) == 2:
            mask[:, -1] = 0
            mask[:, :-1] = 1
        elif len(x.shape) == 3:
            mask[:, :, -1] = 0
            mask[:, :, :-1] = 1

    inp = tensor_x.masked_fill(mask == 0, -1)
    return inp, mask


def validate_dataset_name(dataset):
    allowed_datasets = [
            'weather',
            'electricity',
            'traffic',
            'ETTh1',
            'ETTm1',
            'ETTh2',
            'ETTm2',
            'all',
            'real_fish_day_12ms',
            'real_fish_day_12ms_seq_len_720',
            'real_fish_day_12ms_eods_only_seq_len_720',
            'real_fish_day_12ms_eods_only_seq_len_640',
            'real_fish_day_12ms_eods_only_seq_len_64',
            'real_fish_day_12ms_eods_only_12hrs_seq_len_64',
            'real_fish_day_12ms_eods_only_12hrs_seq_len_64_fish_all',
            'real_fish_day_12ms_eods_only_12hrs_seq_len_64_fish_a',
            'real_fish_day_12ms_eods_only_12hrs_seq_len_64_fish_b',
            'real_fish_day_12ms_eods_only_12hrs_seq_len_64_fish_c',
            'real_fish_day_12ms_eods_only_12hrs_seq_len_64_fish_d',
    ]
    assert dataset in allowed_datasets, f"dataset must be one of the allowed datasets: {allowed_datasets}"


def create_dataloaders(batchsize=100,
                       dataset="dummy",
                       base_path='dummy',
                       revined_data=False,
                       ):
    validate_dataset_name(dataset)
    print(f"Creating dataloaders for dataset: {dataset}")
    full_path = base_path + '/' + dataset

    if revined_data == 'False':
        train_data = np.load(os.path.join(full_path, "train_notrevin_x.npy"), allow_pickle=True)
        val_data = np.load(os.path.join(full_path, "val_notrevin_x.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test_notrevin_x.npy"), allow_pickle=True)

    elif revined_data == 'True':
        print('Using revined data')
        breakpoint()  # currently not recommended
        train_data = np.load(os.path.join(full_path, "train_revin_x.npy"), allow_pickle=True)
        val_data = np.load(os.path.join(full_path, "val_revin_x.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test_revin_x.npy"), allow_pickle=True)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=10,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


def create_test_dataloader_dataset(batchsize=100, dataset="dummy", base_path='dummy', revined_data=False):
    if revined_data:
        print('Testing on revined data not advised')
        breakpoint()

    validate_dataset_name(dataset)
    print(f"Creating test dataloader for dataset: {dataset}")
    full_path = base_path + '/' + dataset
    test_data = np.load(os.path.join(full_path, "test_notrevin_x.npy"), allow_pickle=True)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    return test_dataloader, test_data
