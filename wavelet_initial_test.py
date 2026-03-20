import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt
import pywt
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env.local"))
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env.local")

print("[1/7] HF token loaded from .env.local", flush=True)


class SpectralNonlinearity(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lambda_approx = nn.Parameter(torch.full((1, channels, 1), 0.05))
        self.lambda_detail = nn.Parameter(torch.full((1, channels, 1), 0.05))
        self.log_gamma = nn.Parameter(torch.zeros((1, channels, 1)))
        self.theta = nn.Parameter(torch.zeros((1, channels, 1)))

    def _transform(self, z, lam):
        gamma = F.softplus(self.log_gamma) + 1e-6
        return gamma * torch.sign(z) * torch.relu(torch.abs(z) - lam) * torch.cos(self.theta)

    def forward_approx(self, z):
        return self._transform(z, self.lambda_approx)

    def forward_detail(self, z):
        return self._transform(z, self.lambda_detail)


class WLMLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        wavelets=None,
        dilation_interval=2,
        max_dilation=2,
        pruning_threshold=0.02,
        ema_momentum=0.9,
    ):
        super().__init__()
        self.wavelets = wavelets or ["haar", "db4", "sym6", "bior1.3"]
        self.spectral = SpectralNonlinearity(embed_dim)
        self.basis_logits = nn.Parameter(torch.zeros(len(self.wavelets)))
        self.register_buffer("active_mask", torch.ones(len(self.wavelets), dtype=torch.bool))
        self.register_buffer("ema_basis_weights", torch.zeros(len(self.wavelets)))
        self.dilation_interval = dilation_interval
        self.max_dilation = max_dilation
        self.pruning_threshold = pruning_threshold
        self.ema_momentum = ema_momentum
        self.current_level = 1
        self._valid_cache = {}
        self._last_full_weights = None
        self.norm = nn.LayerNorm(embed_dim)

    def set_epoch(self, epoch_idx):
        dilation = min(epoch_idx // self.dilation_interval, self.max_dilation)
        self.current_level = 1 + dilation

    def _get_valid_indices(self, x):
        seq_len = x.shape[-1]
        cache_key = (seq_len, self.current_level)
        if cache_key in self._valid_cache:
            return self._valid_cache[cache_key]

        valid = []
        dummy = x[:1, :1, :]
        for idx, wavelet in enumerate(self.wavelets):
            if not self.active_mask[idx]:
                continue
            try:
                coeffs = ptwt.wavedec(dummy, wavelet, level=self.current_level)
                recon = ptwt.waverec(coeffs, wavelet)
                if recon.shape[-1] >= seq_len:
                    valid.append(idx)
            except Exception:
                continue

        if not valid:
            valid = [0]
            self.active_mask[0] = True

        self._valid_cache[cache_key] = valid
        return valid

    def maybe_prune(self):
        with torch.no_grad():
            candidates = torch.where(self.active_mask)[0]
            if len(candidates) <= 1:
                return
            to_prune = (self.ema_basis_weights < self.pruning_threshold) & self.active_mask
            if to_prune.any():
                best_idx = int(torch.argmax(self.ema_basis_weights).item())
                to_prune[best_idx] = False
                self.active_mask[to_prune] = False
                self._valid_cache.clear()

    def entropy_penalty(self):
        mask = self.active_mask
        logits = self.basis_logits[mask]
        weights = torch.softmax(logits, dim=0)
        return -(weights * (weights + 1e-8).log()).sum()

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)

        valid_indices = self._get_valid_indices(x)
        logits = self.basis_logits[valid_indices]
        weights = torch.softmax(logits, dim=0)
        recon_candidates = []

        for local_idx, basis_idx in enumerate(valid_indices):
            wavelet = self.wavelets[basis_idx]
            coeffs = ptwt.wavedec(x, wavelet, level=self.current_level)
            processed = [self.spectral.forward_approx(coeffs[0])]
            processed.extend(self.spectral.forward_detail(c) for c in coeffs[1:])
            rec = ptwt.waverec(processed, wavelet)
            rec = rec[:, :, :residual.shape[1]]
            recon_candidates.append(weights[local_idx] * rec)

        x = torch.stack(recon_candidates, dim=0).sum(dim=0)

        with torch.no_grad():
            full_weights = torch.zeros_like(self.basis_logits)
            full_weights[valid_indices] = weights.detach()
            self._last_full_weights = full_weights
            self.ema_basis_weights.mul_(self.ema_momentum).add_((1 - self.ema_momentum) * full_weights)

        x = x.transpose(1, 2)
        return self.norm(x + residual)

class WLMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, pad_token_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([WLMLayer(embed_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, 2)

    def set_epoch(self, epoch_idx):
        for layer in self.layers:
            layer.set_epoch(epoch_idx)

    def prune_wavelets(self):
        for layer in self.layers:
            layer.maybe_prune()

    def entropy_penalty(self):
        return torch.stack([layer.entropy_penalty() for layer in self.layers]).mean()

    def active_wavelets_summary(self):
        summaries = []
        for i, layer in enumerate(self.layers, start=1):
            active = [w for w, m in zip(layer.wavelets, layer.active_mask.tolist()) if m]
            summaries.append(f"L{i}:{'/'.join(active)}")
        return " | ".join(summaries)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)




# 1. Setup Data
print("[2/7] Loading tokenizer: bert-base-uncased", flush=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=hf_token)

print("[3/7] Downloading/loading dataset: GLUE/SST-2", flush=True)
dataset = load_dataset("glue", "sst2", token=hf_token)

def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)

print("[4/7] Tokenizing dataset (this can take a bit on first run)", flush=True)
tokenized_ds = dataset.map(tokenize_fn, batched=True, desc="Tokenizing SST-2")
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(tokenized_ds["train"], batch_size=32, shuffle=True,
                          collate_fn=DataCollatorWithPadding(tokenizer))
val_loader = DataLoader(tokenized_ds["validation"], batch_size=64, shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer))
print(f"[5/7] Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches", flush=True)

# 2. Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WLMClassifier(
    tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
criterion = nn.CrossEntropyLoss()
num_epochs = 6
entropy_beta = 1e-3
early_stopping_patience = 2
best_val_loss = float("inf")
epochs_without_improvement = 0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
    model.train()
    return total_loss / total_count, total_correct / total_count

# 3. Training Loop (Single Epoch for Demo)
print("[6/7] Starting training", flush=True)
model.train()
loss = None
for epoch in range(1, num_epochs + 1):
    model.set_epoch(epoch - 1)
    running_loss = 0.0
    running_ce = 0.0
    running_entropy = 0.0

    for step, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        ce_loss = criterion(outputs, labels)
        entropy_penalty = model.entropy_penalty()
        loss = ce_loss + entropy_beta * entropy_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_ce += ce_loss.item()
        running_entropy += entropy_penalty.item()

        if step == 1 or step % 200 == 0:
            print(
                (
                    f"Epoch {epoch}/{num_epochs} - step {step}/{len(train_loader)} "
                    f"- total: {loss.item():.4f} - ce: {ce_loss.item():.4f} - ent: {entropy_penalty.item():.4f}"
                ),
                flush=True,
            )

    avg_epoch_loss = running_loss / len(train_loader)
    avg_epoch_ce = running_ce / len(train_loader)
    avg_epoch_entropy = running_entropy / len(train_loader)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    model.prune_wavelets()
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        (
            f"Epoch {epoch}/{num_epochs} complete - train_total: {avg_epoch_loss:.4f} "
            f"train_ce: {avg_epoch_ce:.4f} train_ent: {avg_epoch_entropy:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} lr: {current_lr:.2e}"
        ),
        flush=True,
    )
    print(f"Active wavelets -> {model.active_wavelets_summary()}", flush=True)

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered on validation loss.", flush=True)
            break

if loss is None:
    raise RuntimeError("No training step was executed. Check dataset and dataloader setup.")

print("[7/7] Training complete", flush=True)
print(f"Final batch loss: {loss.item():.4f}", flush=True)