# This is an example training file we'll use using to demonstrate docker.
# It is a simple llm that is trained on a corpus of text- it requires Pytorch, which we will bundle into its docker container.

/* 
EXAMPLE TRAINING TEXT

Machine learning is a field of computer science focused on building systems that learn from data.
Large language models predict the next token in a sequence and improve through optimization.
This is a tiny demo dataset, so the results will be rough, repetitive, and only useful for learning.

*/

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int = 128):
        if len(text) < block_size + 1:
            raise ValueError("Input text is too short for the chosen block_size.")

        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size

        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, hidden_size: int = 256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)          # (B, T, d_model)
        x, _ = self.lstm(x)            # (B, T, hidden_size)
        x = self.ln(x)
        logits = self.head(x)          # (B, T, vocab_size)
        return logits


@torch.no_grad()
def generate(model, start_tokens, max_new_tokens, block_size, device):
    model.eval()
    idx = start_tokens.clone().to(device)

    for _ in range(max_new_tokens):
        x = idx[:, -block_size:]
        logits = model(x)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


def decode(tokens, itos):
    return "".join(itos[int(t)] for t in tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--save-path", type=str, default="tiny_lm.pt")
    args = parser.parse_args()

    text_path = Path(args.input)
    if not text_path.exists():
        raise FileNotFoundError(f"Could not find input file: {text_path}")

    text = text_path.read_text(encoding="utf-8")
    dataset = CharDataset(text=text, block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset length: {len(dataset)}")

    model = TinyLM(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(
                logits.reshape(-1, dataset.vocab_size),
                y.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        print(f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
        "vocab_size": dataset.vocab_size,
        "block_size": args.block_size,
        "d_model": args.d_model,
        "hidden_size": args.hidden_size,
    }
    torch.save(checkpoint, args.save_path)
    print(f"Saved model to {args.save_path}")

    # Sample generation
    start = torch.tensor([[0]], dtype=torch.long, device=device)
    sample = generate(
        model=model,
        start_tokens=start,
        max_new_tokens=args.sample_tokens,
        block_size=args.block_size,
        device=device,
    )
    print("\n=== SAMPLE OUTPUT ===\n")
    print(decode(sample[0].tolist(), dataset.itos))


if __name__ == "__main__":
    main()
