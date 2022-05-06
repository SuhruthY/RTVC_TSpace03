import torch
import torch.nn as nn

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq

from params import mel_n_channels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_DEVICE = torch.device("cpu")

class SpeakerEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3, learning_rate=1e-4):
        super().__init__()
        self.loss_device = LOSS_DEVICE
        
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True).to(DEVICE)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=EM_SIZE).to(DEVICE)
        self.relu = torch.nn.ReLU().to(DEVICE)
        
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
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
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
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer