import argparse
import json
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write
import soundfile as sf
# For Grad-TTS
import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

# For HiFi-GAN
import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
import torch

def compute_metrics(original_audio, generated_audio):
    # Ensure both audio arrays have the same length
    min_length = min(len(original_audio), len(generated_audio))
    original_audio = original_audio[:min_length]
    generated_audio = generated_audio[:min_length]
    # Compute the power of the original and noise (original - generated)
    power_original = np.sum(original_audio ** 2)
    power_noise = np.sum((original_audio - generated_audio) ** 2)
    epsilon = 1e-10
    # Calculate SNR in dB
    snr = 10 * np.log10(max(epsilon, power_original / max(power_noise, epsilon)))
    mse = np.mean((original_audio - generated_audio) ** 2)
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10(1.0 / max(mse, epsilon))
    return [snr,mse,rmse,psnr]

###########HYPERPARAM#############
exp_n = 1
model = 'w_50'
dataset = "ESD"
n_emotions = 1

spk_emb = 64
emo_emb = 64
path_weights = './checkpts/grad-tts.pt'
n_spks = 1
##################################

#path_weights = './logs/final_exp/'+str(exp_n)+'/'+model+'.pt'

# if dataset == "ESD":
#      n_spks = 10
# elif dataset == "Friends":
#      n_spks = 6

with open('evaluation_dataset.json', 'r', encoding='utf-8') as eval_file:
    evaluation_dataset = json.load(eval_file)

# Load TTS model
generator = GradTTS(len(symbols)+1, n_spks, spk_emb, n_emotions, emo_emb,
                    params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                    pe_scale=1000)  
generator.load_state_dict(torch.load(path_weights, map_location=lambda loc, storage: loc))
_ = generator.cuda().eval()


cmu = cmudict.CMUDict('./resources/cmu_dictionary')
with open('./checkpts/hifigan-config.json') as f:
    h = AttrDict(json.load(f))
hifigan = HiFiGAN(h)
hifigan.load_state_dict(torch.load('./checkpts/hifigan.pt', map_location=lambda loc, storage: loc)['generator'])
_ = hifigan.cuda().eval()
hifigan.remove_weight_norm()


metrics_results = {"samples": []}
print("Evaluating "+dataset+" : ")
num_samples = len(evaluation_dataset[dataset])
progress_bar = tqdm(total=num_samples, desc='Processing Samples', unit='sample')

all_emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
# Iterate over ESD samples
mean_metrics = [0,0,0,0]
for sample in evaluation_dataset[dataset]:
    text_prompt = sample["text"]
    x = torch.LongTensor(intersperse(text_to_sequence(text_prompt, dictionary=cmu), len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    # Generate speech with prompt
    y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=50, temperature=1.3,
                                       stoc=False, 
                                       #spk=torch.LongTensor([sample["character-id"]]).cuda(),
                                       spk=None,
                                       emotion=None if n_emotions==1 else torch.LongTensor([int(all_emotions.index(sample["emotion"]))]).cuda(),
                                       length_scale=0.91)
                                       
    generated_audio = hifigan.forward(y_dec)

    original_audio_path = sample["path-wav"]
    original_audio, sample_rate = sf.read(original_audio_path)
    if dataset == "Friends":
        original_audio = original_audio[:,0]
    # Compute metrics
    metrics = compute_metrics(original_audio, generated_audio.cpu().detach().numpy())
    mean_metrics = [a + b for a, b in zip(mean_metrics, metrics)]
    metrics_results["samples"].append({
        "text": text_prompt,
        "original_audio_path": original_audio_path,
        "metrics": metrics
    })
    progress_bar.update(1)
mean_metrics = [x / num_samples for x in mean_metrics]
progress_bar.close()

# Save the metrics
# with open('logs/final_exp/'+str(exp_n)+'/metrics_'+dataset+'_'+str(n_emotions)+"emo_"+str(spk_emb)+"emb_"+model+'_results.json'
# , 'w', encoding='utf-8') as metrics_file:
#     json.dump(metrics_results, metrics_file, indent=2)


with open('logs/final_exp/'+str(exp_n)+'/mean_metrics_'+dataset+'_'+str(n_emotions)+"emo_"+str(spk_emb)+"emb_"+model+'_results.json'
, 'w', encoding='utf-8') as metrics_file:
    json.dump(mean_metrics, metrics_file, indent=2)