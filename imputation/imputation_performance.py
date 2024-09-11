import argparse
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import sklearn
import sklearn.metrics
import pandas as pd


def one_loop(loader, vqvae_model, device, args):
    mse = []
    mae = []
    preds = []
    trues = []
    codes = []
    code_ids = []
    inps = []

    for i, batch_x in enumerate(loader):
        batch_x = batch_x.float().to(device)
        batch_x = torch.unsqueeze(batch_x, dim=1)  # expects time to be dim [bs x nvars x time]

        # random mask
        B, N, T = batch_x.shape
        mask = torch.rand((B, N, T)).to(device)
        mask[mask <= args.mask_ratio] = 0  # masked
        mask[mask > args.mask_ratio] = 1  # remained
        # inp = batch_x.masked_fill(mask == 0, 0)
        inp = batch_x.masked_fill(mask == 0, -1)

        x_codes, x_code_ids, codebook = revintime2codes(inp, args.compression_factor, vqvae_model.encoder, vqvae_model.vq)
        # print('codebook', codebook)
        # expects code to be dim [bs x nvars x compressed_time]
        x_predictions_revin_space = codes2timerevin(x_code_ids, codebook, args.compression_factor, vqvae_model.decoder)

        batch_x_masky = batch_x[mask == 0]
        pred_x_masky = np.swapaxes(x_predictions_revin_space, 1, 2)[mask == 0]
        preds.append(pred_x_masky)
        trues.append(batch_x_masky)
        inps.append(inp)
        codes.append(x_codes.detach().cpu().numpy())
        code_ids.append(x_code_ids.detach().cpu().numpy())


        mse.append(nn.functional.mse_loss(batch_x_masky, pred_x_masky).item())
        mae.append(nn.functional.l1_loss(batch_x_masky, pred_x_masky).item())

    seed = args.trained_vqvae_model_path.split('seed')[1].split('_')[0]

    print('MSE:', np.mean(mse))
    print('MAE:', np.mean(mae))
    import joblib
    joblib.dump(preds, f'/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/{args.dataset}_mr{args.mask_ratio}_seed{seed}_f{args.compression_factor}_preds.pkl')
    joblib.dump(trues, f'/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/{args.dataset}_mr{args.mask_ratio}_seed{seed}_f{args.compression_factor}_trues.pkl')
    joblib.dump(codes, f'/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/{args.dataset}_mr{args.mask_ratio}_seed{seed}_f{args.compression_factor}_codes.pkl')
    joblib.dump(code_ids, f'/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/{args.dataset}_mr{args.mask_ratio}_seed{seed}_f{args.compression_factor}_code_ids.pkl')
    joblib.dump(codebook.detach().cpu().numpy(), f'/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/{args.dataset}_mr{args.mask_ratio}_seed{seed}_f{args.compression_factor}_codebook.pkl')
    joblib.dump(inps, f'/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/{args.dataset}_mr{args.mask_ratio}_seed{seed}_f{args.compression_factor}_inps.pkl')

    concatenated_preds = torch.cat(preds, dim=0)
    concatenated_trues = torch.cat(trues, dim=0)
    thresholded_preds_all = np.where(concatenated_preds.cpu() > 0.5, 1, 0)
    trues_all = concatenated_trues.cpu().numpy()
    metrics = sklearn.metrics.precision_recall_fscore_support(trues_all, thresholded_preds_all, average=None)
    metrics_df = pd.DataFrame({
        'Precision': metrics[0],
        'Recall': metrics[1],
        'F1-Score': metrics[2],
        'Support': metrics[3]
    })
    print(metrics_df.round(3))
    f1 = metrics_df['F1-Score'].mean()
    print('Overall F1-Score:', f1.round(3))



def revintime2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    '''
    Args:
        revin_data: [bs x nvars x pred_len or seq_len]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients

    Returns:
        codes: [bs, nvars, code_dim, compressed_time]
        code_ids: [bs, nvars, compressed_time]
        embedding_weight: [num_code_words, code_dim]

    Helpful VQVAE Comments:
        # Into the vqvae encoder: batch.shape: [bs x seq_len] i.e. torch.Size([256, 12])
        # into the quantizer: z.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # into the vqvae decoder: quantized.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # out of the vqvae decoder: data_recon.shape: [bs x seq_len] i.e. torch.Size([256, 12])
        # this is if your compression factor=4
    '''

    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    T = revin_data.shape[2]  # this can be either the prediction length or the sequence length
    compressed_time = int(T / compression_factor)  # this can be the compressed time of either the prediction length or the sequence length

    with torch.no_grad():
        flat_revin = revin_data.reshape(-1, T)  # flat_y: [bs * nvars, T]
        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)  # latent_y: [bs * nvars, code_dim, compressed_time]
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)  # quantized: [bs * nvars, code_dim, compressed_time]
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, nvar, code_dim,
                                  compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
        code_ids = encoding_indices.view(bs, nvar, compressed_time)  # code_ids: [bs, nvars, compressed_time]

    return codes, code_ids, embedding_weight


def codes2timerevin(code_ids, codebook, compression_factor, vqvae_decoder):
    '''
    Args:
        code_ids: [bs x nvars x compressed_pred_len]
        codebook: [num_code_words, code_dim]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients
        x_or_y: if 'x' use revin_denorm_x if 'y' use revin_denorm_y
    Returns:
        predictions_revin_space: [bs x original_time_len x nvars]
        predictions_original_space: [bs x original_time_len x nvars]
    '''
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        # scatter the label with the codebook
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device),1)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)  # quantized: [bs * nvars, compressed_pred_len, code_dim]
        quantized_swaped = torch.swapaxes(quantized, 1,2)  # quantized_swaped: [bs * nvars, code_dim, compressed_pred_len]
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)  # prediction_recon: [bs * nvars, pred_len]
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])  # prediction_recon_reshaped: [bs x nvars x pred_len]
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1,2)  # prediction_recon_nvars_last: [bs x pred_len x nvars]

    return predictions_revin_space


def create_NONrevin_dataloaders(batchsize=100, dataset="dummy", base_path='dummy'):

    if dataset == 'weather':
        print('weather')
        full_path = base_path + '/weather'

    elif dataset == 'electricity':
        print('electricity')
        full_path = base_path + '/electricity'

    elif dataset == 'traffic':
        print('traffic')
        full_path = base_path + '/traffic'

    elif dataset == 'ETTh1':
        print('ETTh1')
        full_path = base_path + '/ETTh1'

    elif dataset == 'ETTm1':
        print('ETTm1')
        full_path = base_path + '/ETTm1'

    elif dataset == 'ETTh2':
        print('ETTh2')
        full_path = base_path + '/ETTh2'

    elif dataset == 'ETTm2':
        print('ETTm2')
        full_path = base_path + '/ETTm2'

    elif dataset == 'all':
        print('all')
        full_path = base_path + '/all'

    elif dataset == 'neuro2':
        print('neuro2')
        full_path = base_path + '/neuro2'

    elif dataset == 'neuro5':
        print('neuro5')
        full_path = base_path + '/neuro5'

    elif dataset == 'saugeen':
        print('saugeen')
        full_path = base_path + '/saugeen'

    elif dataset == 'us_births':
        print('us_births')
        full_path = base_path + '/us_births'

    elif dataset == 'sunspot':
        print('sunspot')
        full_path = base_path + '/sunspot'

    elif dataset == 'real_fish_day_12ms':
            print('real_fish_day_12ms')
            full_path = base_path + '/real_fish_day_12ms'

    elif dataset == 'real_fish_day_12ms_seq_len_720':
            print('real_fish_day_12ms_seq_len_720')
            full_path = base_path + '/real_fish_day_12ms_seq_len_720'
    
    elif dataset == 'real_fish_day_12ms_eods_only_seq_len_720':
            print('real_fish_day_12ms_eods_only_seq_len_720')
            full_path = base_path + '/real_fish_day_12ms_eods_only_seq_len_720'

    elif dataset == 'real_fish_day_12ms_eods_only_seq_len_640':
            print('real_fish_day_12ms_eods_only_seq_len_640')
            full_path = base_path + '/real_fish_day_12ms_eods_only_seq_len_640'

    else:
        print('Not done yet')
        pdb.set_trace()

    test_data = np.load(os.path.join(full_path, "test_notrevin_x.npy"), allow_pickle=True)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    return test_dataloader, test_data


def main(args):
    device = 'cuda:' + str(args.gpu)

    test_loader, test_data = create_NONrevin_dataloaders(batchsize=4096*10, dataset=args.dataset, base_path=args.base_path)
    # run_baselines(test_data)
    # return

    vqvae_model = torch.load(args.trained_vqvae_model_path)
    vqvae_model.to(device)
    vqvae_model.eval()
    print('TEST')
    one_loop(test_loader, vqvae_model, device, args)
    print('-------------')

def get_sklearn_report(preds, trues):
    metrics = sklearn.metrics.precision_recall_fscore_support(trues, preds, average=None)

    metrics_df = pd.DataFrame({
        'Precision': metrics[0],
        'Recall': metrics[1],
        'F1-Score': metrics[2],
        'Support': metrics[3]
    })
    print(metrics_df.round(3))
    f1 = metrics_df['F1-Score'].mean()
    print('Overall F1-Score:', f1.round(3))


def run_baselines(test_data):
    test_data = test_data.flatten()
    all_0_preds = np.zeros_like(test_data)
    all_1_preds = np.ones_like(test_data)

    p_1 = np.sum(test_data) / test_data.size
    random_preds = np.random.choice([0, 1], size=test_data.size, p=[1-p_1, p_1])
    
    print('All 0s')
    get_sklearn_report(all_0_preds, test_data)
    print('All 1s')
    get_sklearn_report(all_1_preds, test_data)
    print('Random')
    get_sklearn_report(random_preds, test_data)

    

    
    

    
    print('-------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    parser.add_argument('--dataset', type=str, required=True, help='')

    parser.add_argument('--trained_vqvae_model_path', type=str, required=True, help='')
    parser.add_argument('--compression_factor', type=int, required=True, help='compression_factor')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--base_path', type=str, help='which data to perform oracle on', default=False)
    parser.add_argument('--mask_ratio', type=float, help='amount of data that is masked', default=False)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    main(args)
