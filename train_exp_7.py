# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params_exp_7
from model import GradTTS
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

train_filelist_path = params_exp_7.train_filelist_path
valid_filelist_path = params_exp_7.valid_filelist_path
cmudict_path = params_exp_7.cmudict_path
add_blank = params_exp_7.add_blank
n_spks = params_exp_7.n_spks
spk_emb_dim = params_exp_7.spk_emb_dim
n_emotions = params_exp_7.n_emotions
emotion_emb_dim = params_exp_7.emotion_emb_dim

log_dir = params_exp_7.log_dir
n_epochs = params_exp_7.n_epochs
batch_size = params_exp_7.batch_size
out_size = params_exp_7.out_size
learning_rate = params_exp_7.learning_rate
random_seed = params_exp_7.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params_exp_7.n_enc_channels
filter_channels = params_exp_7.filter_channels
filter_channels_dp = params_exp_7.filter_channels_dp
n_enc_layers = params_exp_7.n_enc_layers
enc_kernel = params_exp_7.enc_kernel
enc_dropout = params_exp_7.enc_dropout
n_heads = params_exp_7.n_heads
window_size = params_exp_7.window_size

n_feats = params_exp_7.n_feats
n_fft = params_exp_7.n_fft
sample_rate = params_exp_7.sample_rate
hop_length = params_exp_7.hop_length
win_length = params_exp_7.win_length
f_min = params_exp_7.f_min
f_max = params_exp_7.f_max

dec_dim = params_exp_7.dec_dim
beta_min = params_exp_7.beta_min
beta_max = params_exp_7.beta_max
pe_scale = params_exp_7.pe_scale


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    test_dataset = TextMelSpeakerDataset(valid_filelist_path, cmudict_path, add_blank,
                                         n_fft, n_feats, sample_rate, hop_length,
                                         win_length, f_min, f_max)
    test_batch_collate = TextMelSpeakerBatchCollate()
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             collate_fn=test_batch_collate, drop_last=True,
                             num_workers=8, shuffle=False)

    print('Initializing model...')
    model = GradTTS(nsymbols, n_spks, spk_emb_dim, 
                    n_emotions, emotion_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()

    model_trained_weights = torch.load('./checkpts/grad-tts-libri-tts.pt', map_location=lambda loc, storage: loc)
    
    # Load only the compatible layers
    unloaded_keys = []
    for key in model.state_dict().keys():
        try:
            model.state_dict()[key].copy_(model_trained_weights[key])
        except:
            state_dict = model.state_dict()
            tensor_for_key = state_dict[key]
            num_parameters_for_key = tensor_for_key.numel()
            unloaded_keys.append(key)

    print('Number of parameters with failed load = '+str(unloaded_keys))
    total_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f'Total number of parameters in the model encoder: {total_params_encoder}')
    total_params_decoder = sum(p.numel() for p in model.decoder.parameters())
    print(f'Total number of parameters in the model encoder: {total_params_decoder}')

    print('Initializing optimizer...')

    for name, param in model.named_parameters():
        if "encoder" in name:
            if "_w" not in name:
                param.requires_grad = False
        if "decoder" in name:
            if "mid" not in name:
                param.requires_grad = False
        if name in unloaded_keys or "_mlp" in name: #spk+emo emb
            param.requires_grad = True

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    untrainable_param_names = [name for name, param in model.named_parameters() if not param.requires_grad]
    
    percentage_selected_encoder = (sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) / total_params_encoder) * 100
    print(f'Percentage of parameters selected for encoder training: {percentage_selected_encoder:.2f}%')
    percentage_selected_decoder = (sum(p.numel() for p in model.decoder.parameters() if p.requires_grad) / total_params_decoder) * 100
    print(f'Percentage of parameters selected for decoder training: {percentage_selected_decoder:.2f}%')

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params_exp_7.test_size)
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        i = int(spk.cpu())

    print('Start training and testing...')
    iteration = 0
    for epoch in range(1, n_epochs + 1):
        model.eval()
        print('Synthesis for test dataset...')
        with torch.no_grad():
            # Testing phase
            test_dur_losses = []
            test_prior_losses = []
            test_diff_losses = []
            with tqdm(test_loader, total=len(test_dataset)//batch_size) as test_progress_bar:
                for test_batch in test_progress_bar:
                    x_test, x_lengths_test = test_batch['x'].cuda(), test_batch['x_lengths'].cuda()
                    y_test, y_lengths_test = test_batch['y'].cuda(), test_batch['y_lengths'].cuda()
                    spk_test = test_batch['spk'].cuda()
                    emotion_test = test_batch['emotion'].cuda()
                    test_dur_loss, test_prior_loss, test_diff_loss = model.compute_loss(x_test, x_lengths_test,
                                                                                    y_test, y_lengths_test,
                                                                                    spk=spk_test, emotion=emotion_test,
                                                                                    out_size=out_size)
                    test_dur_losses.append(test_dur_loss.item())
                    test_prior_losses.append(test_prior_loss.item())
                    test_diff_losses.append(test_diff_loss.item())

            test_msg = 'Epoch %d: test duration loss = %.3f ' % (epoch, np.mean(test_dur_losses))
            test_msg += '| test prior loss = %.3f ' % np.mean(test_prior_losses)
            test_msg += '| test diffusion loss = %.3f\n' % np.mean(test_diff_losses)
            with open(f'{log_dir}/test.log', 'a') as test_f:
                test_f.write(test_msg)

        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                spk = batch['spk'].cuda()
                emotion = batch['emotion'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     spk=spk, emotion=emotion, 
                                                                    out_size=out_size)
                loss = sum([dur_loss, 10*diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 
                                                            max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 
                                                            max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss,
                                global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss,
                                global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                global_step=iteration)
                
                msg = f'Epoch: {epoch}, ite: {iteration} | dur_loss: {dur_loss.item():.2f}, prior_loss: {prior_loss.item():.2f}, diff_loss: {diff_loss.item():.4f}'
                progress_bar.set_description(msg)
                ckpt = model.state_dict()
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                iteration += 1

        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)

        if epoch % params_exp_7.save_every > 0:
            continue
        
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/w_{epoch}.pt")