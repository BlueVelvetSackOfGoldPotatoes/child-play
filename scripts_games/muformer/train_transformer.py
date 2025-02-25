#!/usr/bin/env python3
"""
train_transformer.py

Loads the multi-game MuZero dataset (multigame_muzero_dataset.json) and trains a
Transformer (starting from deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
to predict (policy, value) from the ASCII state.

We handle multiple environments at once in a single dataset.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import os

class MultiGameMuZeroDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "env_name": item["env_name"],
            "ascii_state": item["ascii_state"],
            "policy": torch.tensor(item["policy"], dtype=torch.float32),
            "value": torch.tensor(item["value"], dtype=torch.float32)
        }

class TransformerPolicyValueNet(nn.Module):
    """
    (ASCII state) -> (policy, value) model, using a base Transformer
    from deepseek-ai/DeepSeek-R1-Distill-Qwen-7B plus two heads.

    We define `max_action_dim` to accommodate the largest action space
    among our combined games (e.g., 49 for a 7x7 board).
    """
    def __init__(self, hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", max_action_dim=49):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.transformer = AutoModel.from_pretrained(hf_model_name)
        hidden_size = self.transformer.config.hidden_size

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_action_dim)  # up to 49
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, ascii_batch):
        """
        ascii_batch: list of raw strings (ASCII boards)
        Returns: policy_logits (B, max_action_dim), value (B,)
        """
        enc = self.tokenizer(ascii_batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Forward pass through base model
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape (B, T, H)

        # Simple average pooling across time
        pooled = hidden_states.mean(dim=1)  # shape (B, H)

        policy_logits = self.policy_head(pooled)       # (B, max_action_dim)
        value = torch.tanh(self.value_head(pooled))    # (B, 1) => [-1,1]
        return policy_logits, value.squeeze(dim=-1)

def collate_fn(batch):
    ascii_states = [x["ascii_state"] for x in batch]
    policy = torch.stack([x["policy"] for x in batch])  # (B, max_action_dim) potentially
    value = torch.stack([x["value"] for x in batch])    # (B,)
    return ascii_states, policy, value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="multigame_muzero_dataset.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hf_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Base model for the transformer.")
    parser.add_argument("--max_action_dim", type=int, default=49, help="Max number of actions across all games.")
    parser.add_argument("--out_dir", type=str, default="transformer_policy_value")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiGameMuZeroDataset(args.dataset_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerPolicyValueNet(
        hf_model_name=args.hf_model,
        max_action_dim=args.max_action_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for ascii_batch, policy, value in train_loader:
            policy = policy.to(device)
            value = value.to(device)

            policy_logits, value_pred = model(ascii_batch)
            # policy_logits: (B, max_action_dim)
            # value_pred:   (B,)

            # We'll compute a KL divergence for policy (so that we match the MuZero distribution),
            # and MSE for the value.
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            # Ensure we handle a policy sum of zero if that occurs
            policy_targets = policy / (policy.sum(dim=-1, keepdim=True) + 1e-8)
            policy_loss = F.kl_div(policy_log_probs, policy_targets, reduction='batchmean')

            value_loss = F.mse_loss(value_pred, value)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ascii_batch, policy, value in val_loader:
                policy = policy.to(device)
                value = value.to(device)

                policy_logits, value_pred = model(ascii_batch)
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                policy_targets = policy / (policy.sum(dim=-1, keepdim=True) + 1e-8)
                policy_loss = F.kl_div(policy_log_probs, policy_targets, reduction='batchmean')
                value_loss = F.mse_loss(value_pred, value)
                val_loss += (policy_loss + value_loss).item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    model.transformer.save_pretrained(args.out_dir)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "policy_value_head.pt"))
    print(f"Model saved to {args.out_dir}")

if __name__ == "__main__":
    main()


"""
python train_transformer.py \
  --dataset_file multigame_muzero_dataset.json \
  --epochs 5 \
  --batch_size 4 \
  --hf_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --max_action_dim 49 \
  --out_dir transformer_policy_value
"""