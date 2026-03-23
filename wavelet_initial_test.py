# ==============================================================================
# IMPORTS ET CONFIGURATION
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import numpy as np
import pywt
import ptwt
import logging
import datetime
import os

# Configuration du Logging
log_filename = f"wlm_sst2_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Exécution sur le device : {device}")
logging.info(f"Démarrage de l'expérience. Device: {device}")

# ==============================================================================
# 1. ARCHITECTURE WLM (Composants Fondamentaux)
# ==============================================================================

class TokenToVolumetric(nn.Module):
    """
    Sliding window et Channel Expansion.
    Extrait le signal maximal en ignorant le Padding.
    """
    def __init__(self, embed_dim=768, C=1, D=8, H=8, W=8, window_size=3):
        super().__init__()
        self.C, self.D, self.H, self.W = C, D, H, W
        
        # Projection très légère (768 -> 512 dimensions)
        self.sliding_window = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=C * D * H * W, # 1 * 8 * 8 * 8 = 512
            kernel_size=window_size, 
            padding=window_size // 2
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, E, L)
        x_windowed = self.sliding_window(x) # -> (B, 512, L)
        
        # Le Max Pooling sauve le modèle : il extrait les mots et ignore le [PAD]
        x_vol = F.adaptive_max_pool1d(x_windowed, 1).squeeze(-1)
        
        return x_vol.view(x.size(0), self.C, self.D, self.H, self.W)

class BandwiseInteraction(nn.Module):
    """
    Simule des portes logiques (AND) entre les sous-bandes directionnelles des ondelettes[cite: 378, 385].
    """
    def __init__(self):
        super().__init__()
        self.gamma_r = nn.Parameter(torch.tensor(1.0))
        self.lambda_r = nn.Parameter(torch.tensor(0.01))

    def forward(self, cD_dict):
        keys = list(cD_dict.keys())
        if len(keys) >= 2:
            k1, k2 = keys[0], keys[1]
            interaction = self.gamma_r * F.relu(cD_dict[k1] * cD_dict[k2] - self.lambda_r)
            cD_dict[k1] = cD_dict[k1] + interaction
            cD_dict[k2] = cD_dict[k2] + interaction
        return cD_dict

class SpectralNonLinearity(nn.Module):
    """
    Applique la non-linéarité spectrale avec seuillage doux et modulation de phase[cite: 32, 146].
    Formule : \phi(z) = \gamma * sign(z) * max(|z| - \lambda, 0) * cos(\theta)[cite: 146].
    """
    def __init__(self, channels):
        super().__init__()
        self.lambda_A = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.05))
        self.lambda_D = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.05))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.theta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def _phi(self, z, lambd):
        soft_threshold = F.relu(torch.abs(z) - lambd)
        return self.gamma * torch.sign(z) * soft_threshold * torch.cos(self.theta)

    def forward(self, cA, cD):
        cA_out = self._phi(cA, self.lambda_A)
        cD_out = {k: self._phi(v, self.lambda_D) for k, v in cD.items()}
        return cA_out, cD_out

class SoftBasisSelectorLayer(nn.Module):
    def __init__(self, channels, D, H, W, candidates=['haar', 'db4', 'sym6', 'bior1.3']):
        super().__init__()
        self.valid_candidates = []
        self.reasoning = BandwiseInteraction()
        
        dummy_tensor = torch.randn(1, channels, D, H, W)
        
        for wav_name in candidates:
            try:
                wavelet = pywt.Wavelet(wav_name)
                coeffs = ptwt.wavedec3(dummy_tensor, wavelet, level=1, mode='reflect')
                rec = ptwt.waverec3(coeffs, wavelet)
                self.valid_candidates.append(wav_name)
            except Exception:
                pass 
                
        self.candidates = self.valid_candidates
        msg = f"Couche WLM initialisée. Bases conservées pour {D}x{H}x{W} : {self.candidates}"
        print(f"[*] {msg}")
        logging.info(msg)
        
        if len(self.candidates) == 0:
            raise ValueError(f"Aucune ondelette n'est compatible avec la taille {D}x{H}x{W}.")

        self.basis_logits = nn.Parameter(torch.randn(len(self.candidates)) * 0.1)
        self.spectral_nl = SpectralNonLinearity(channels)
        
    def _real_dwt3d(self, x, wavelet_name):
        wavelet = pywt.Wavelet(wavelet_name)
        coeffs = ptwt.wavedec3(x, wavelet, level=1, mode='reflect')
        return coeffs[0], coeffs[1]
        
    def _real_idwt3d(self, cA, cD_dict, wavelet_name, target_shape):
        wavelet = pywt.Wavelet(wavelet_name)
        x_rec = ptwt.waverec3([cA, cD_dict], wavelet)
        return x_rec[:, :, :target_shape[2], :target_shape[3], :target_shape[4]]

    def forward(self, x, dilation_scale=1):
        w = F.softmax(self.basis_logits, dim=0)
        x_reconstructed = 0
        
        for i, wavelet_name in enumerate(self.candidates):
            cA, cD_dict = self._real_dwt3d(x, wavelet_name)
            cA_prime, cD_prime = self.spectral_nl(cA, cD_dict)
            cD_prime = self.reasoning(cD_prime)
            x_hat_k = self._real_idwt3d(cA_prime, cD_prime, wavelet_name, x.shape)
            x_reconstructed = x_reconstructed + w[i] * x_hat_k
            
        return x_reconstructed, w
        
class WaveletLogicModel(nn.Module):
    def __init__(self, embed_dim, D, H, W, num_hops=3):
        super().__init__()
        # On utilise C=1 canal pour garder un cube unique et peu de paramètres
        self.pipeline = TokenToVolumetric(embed_dim, C=1, D=D, H=H, W=W)
        self.layers = nn.ModuleList([
            SoftBasisSelectorLayer(1, D, H, W) for _ in range(num_hops) # 1 canal spatial
        ])
        self.T_d = 2     
        self.s_max = 4   
        
    def forward(self, x, current_epoch):
        s_t = min(current_epoch // self.T_d, self.s_max)
        vol = self.pipeline(x)
        all_weights = []
        
        for layer in self.layers:
            vol, w = layer(vol, dilation_scale=s_t)
            all_weights.append(w)
            
        # On aplatit le cube 8x8x8 pour le classifieur final (B, 512)
        vol_flat = vol.view(vol.shape[0], -1)
        return vol_flat, torch.stack(all_weights)

# ==============================================================================
# 2. MODÈLE DE CLASSIFICATION ET FONCTION DE PERTE
# ==============================================================================

class WLMForGLUE(nn.Module):
    def __init__(self, vocab_size, embed_dim, D, H, W, num_labels):
        super().__init__()
        
        print("[*] Chargement des embeddings pré-entraînés figés (BERT)...")
        logging.info("Chargement des embeddings pré-entraînés figés (BERT)...")
        bert_temp = AutoModel.from_pretrained("bert-base-uncased")
        pretrained_weights = bert_temp.embeddings.word_embeddings.weight.data
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
        embed_dim = 768 
        del bert_temp 
        
        self.wlm = WaveletLogicModel(embed_dim, D, H, W, num_hops=3)
        
        # Le cube aplati fera 1 * 8 * 8 * 8 = 512 dimensions
        flat_dim = 1 * D * H * W
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(flat_dim, flat_dim // 2), # 512 -> 256
            nn.GELU(),
            nn.Linear(flat_dim // 2, num_labels) # 256 -> 2 [cite: 232]
        )

    def forward(self, input_ids, current_epoch=0):
        x = self.embedding(input_ids)
        wlm_out, basis_weights = self.wlm(x, current_epoch)
        logits = self.classifier(wlm_out)
        return logits, basis_weights

def wlm_loss_fn(logits, labels, basis_weights, beta=0.1, lambda_prune=0.5, tau=0.05):
    """
    Fonction de perte avec Entropy (Shannon) + Pénalité d'élagage dur (Hard Pruning)[cite: 393].
    """
    ce_loss = F.cross_entropy(logits, labels)
    
    # Correction mathématique : on SOUSTRAIT la somme (ce qui équivaut à ajouter la vraie entropie)
    entropy_loss = - torch.sum(basis_weights * torch.log(basis_weights + 1e-9))
    
    # NOUVEAU : La pénalité de Hard Pruning (Élagage dur) 
    # Si un poids passe sous le seuil tau (ex: 5%), on génère un gradient pour le tuer définitivement
    pruning_penalty = lambda_prune * torch.sum(basis_weights * (basis_weights < tau).float())
    
    return ce_loss + beta * entropy_loss + pruning_penalty

# ==============================================================================
# 3. PIPELINE DE DONNÉES ET ENTRAÎNEMENT
# ==============================================================================

def get_dataloaders(batch_size=8, max_length=512):
    print("Préparation du dataset GLUE (SST-2)...")
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=max_length)
    
    encoded_dataset = dataset.map(tokenize, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    train_dl = DataLoader(encoded_dataset["train"], batch_size=batch_size, shuffle=True)
    eval_dl = DataLoader(encoded_dataset["validation"], batch_size=batch_size)
    
    return train_dl, eval_dl, tokenizer.vocab_size

def train_and_evaluate():
    # Hyperparamètres ajustés pour 11 Go de VRAM
    EPOCHS = 6
    BATCH_SIZE = 8       
    ACCUMULATION_STEPS = 8 
    LR = 3e-4 
    D, H, W = 8, 8, 8 
    
    logging.info(f"Hyperparamètres: Epochs={EPOCHS}, Batch={BATCH_SIZE}, Accumulation={ACCUMULATION_STEPS}, LR={LR}")
    
    # max_length=512 pour donner l'espace nécessaire au text pooling [cite: 230]
    train_dl, eval_dl, vocab_size = get_dataloaders(batch_size=BATCH_SIZE, max_length=512)
    model = WLMForGLUE(vocab_size, 768, D, H, W, num_labels=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg_params = f"Paramètres entraînables (WLM + Classifieur) : {total_params / 1e6:.2f} Millions"
    print(f"[*] {msg_params}")
    logging.info(msg_params)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print("\n--- ÉVALUATION BASELINE (Pré-entraînement) ---")
    model.eval()
    all_preds_base, all_labels_base = [], []
    with torch.no_grad():
        for batch in tqdm(eval_dl, desc="[Baseline]"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, current_epoch=0)
            preds = torch.argmax(logits, dim=-1)
            all_preds_base.extend(preds.cpu().numpy())
            all_labels_base.extend(labels.cpu().numpy())
            
    acc_base = accuracy_score(all_labels_base, all_preds_base)
    print(f"Précision initiale (Aléatoire) : {acc_base:.4f}\n")
    logging.info(f"Évaluation Baseline terminée. Précision : {acc_base:.4f}")
    
    print("\nDémarrage de l'entraînement WLM avec Accumulation de Gradients et Pruning...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad() 
        
        # Phase d'entraînement
        for step, batch in enumerate(tqdm(train_dl, desc=f"Époque {epoch+1}/{EPOCHS} [Train]")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, weights = model(input_ids, current_epoch=epoch)
            # Ajout de lambda_prune et tau 
            loss = wlm_loss_fn(logits, labels, weights[-1], beta=0.1, lambda_prune=0.5, tau=0.05) / ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad() 
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            
        avg_train_loss = total_loss / len(train_dl)
        
        # Phase d'évaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(eval_dl, desc=f"Époque {epoch+1}/{EPOCHS} [Eval]"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids, current_epoch=epoch)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        final_weights = np.round(weights[-1].detach().cpu().numpy(), 3)
        bases_str = f"Bases {model.wlm.layers[0].candidates} : {final_weights}"
        
        # Sauvegarde du meilleur modèle
        saved_msg = ""
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "best_wlm_model.pt")
            saved_msg = " [Modèle sauvegardé ⭐]"
        
        log_msg = f"Époque {epoch+1} | Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f} | {bases_str}{saved_msg}"
        print(f"Bilan {log_msg}")
        logging.info(log_msg)

if __name__ == "__main__":
    train_and_evaluate()
