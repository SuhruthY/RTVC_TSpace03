import struct
import librosa
import webrtcvad
import numpy as np 

import torch
import torch.nn as nn

from pathlib import Path
from scipy.ndimage.morphology import binary_dilation

TARG_dBFS = -30
TARG_FS = 16000 
INT16_MAX = (2**15) - 1

VAD_WNDW_LEN = 30  
VAD_WNDW_AVG_WIDTH = 8
VAD_MAX_SILENCE_LEN = 6

MEL_WNDW_LEN = 25  
MEL_WNDW_STP = 10  
MEL_N_CHANNELS = 40 

PAR_N_FRAMES=160
MEL_N_CHANNELS=40
EMBEDDINGS_SIZE=256

## INFERENCE CLASS
class EncoderInference:
    def __init__(self, device=None):
        self.model=None
        self.device=device

    def load_model(self, weights_fpath, device=None):   
        self.weights_fpath = Path(weights_fpath)     
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)

        self.model = SpeakerEncoder(self.device, torch.device("cpu"))
        checkpoint = torch.load(self.weights_fpath, self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print(f"Loaded encoder from {self.weights_fpath.name} trained to step {checkpoint['step']}")

    def is_loaded(self):
        return self.model is not None
    
    def predict(self, fpath):
        arr = preprocess(fpath)
        return self.embed_utterance(arr)
    
    def embed_utterance(self, wav, using_partials=True, return_partials=False):
        if not using_partials:
            frames = generate_frames(wav)
            embed = self.embed_frames_batch(frames[None, ...])[0]
            if return_partials:
                return embed, None, None
            return embed

        wave_slices, mel_slices = self.compute_partial_slices(len(wav))
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        frames = generate_frames(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        partial_embeds = self.embed_frames_batch(frames_batch)

        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials: return embed, partial_embeds, wave_slices
        return embed

    def embed_frames_batch(self, frames_batch):  
        if self.model is None:
            raise Exception("Model was not loaded. Call load_model() before inference.")

        frames = torch.from_numpy(frames_batch).to(self.device)
        embed = self.model.forward(frames).detach().cpu().numpy()
        return embed

    @staticmethod
    def compute_partial_slices(n_samples, partial_utterance_n_frames=PAR_N_FRAMES,
                            min_pad_coverage=0.75, overlap=0.5):
        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        samples_per_frame = int((TARG_FS * MEL_WNDW_STP / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

# UTILS
def preprocess(file_path, normalize=True, trim_silences=True):
        arr, _ = librosa.load(file_path) 
        if normalize: arr = normalize_volume(arr)
        if trim_silences: arr = trim_long_silences(arr)
        return arr

def normalize_volume(wav, target_dBFS=TARG_dBFS, increase_only=False, decrease_only=False):
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

def trim_long_silences(wav, sampling_rate=TARG_FS):
    samples_per_window = (VAD_WNDW_LEN * sampling_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))

    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2], sample_rate=sampling_rate))
    audio_mask = moving_average(voice_flags, VAD_WNDW_AVG_WIDTH)
    audio_mask = np.round(audio_mask).astype(np.bool_)
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_SILENCE_LEN + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask == True]

def moving_average(array, width):
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width

def generate_frames(preprocessed_wav, sampling_rate=TARG_FS):
        frames = librosa.feature.melspectrogram(
        y=preprocessed_wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * MEL_WNDW_LEN / 1000),
        hop_length=int(sampling_rate * MEL_WNDW_STP / 1000),
        n_mels=MEL_N_CHANNELS
        )
        return frames.astype(np.float32).T

class SpeakerEncoder(nn.Module):
    def __init__(self, 
        device, 
        loss_device,
        hidden_size=256,
        num_layers=3,
        learning_rate =1e-4,
        ):
        super().__init__()
        self.loss_device = loss_device
        
        self.lstm = nn.LSTM(input_size=MEL_N_CHANNELS,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=EMBEDDINGS_SIZE).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(self.loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(self.loss_device)

        self.loss_fn = nn.CrossEntropyLoss().to(self.loss_device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def do_gradient_ops(self):
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)      
        embeds_raw = self.relu(self.linear(hidden[-1]))       
        return embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
    
    def similarity_matrix(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        return sim_matrix * self.similarity_weight + self.similarity_bias
    
    def loss(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer

