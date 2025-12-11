import argparse
import json
import math
import os
import time
import random
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data import Calculations, ArithmeticDataset, stoi, itos
from src.model import NLM, ModelConfig
from tqdm import tqdm


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokens_to_tensor(tokens: List[str]) -> torch.Tensor:
    ids = [stoi[c] for c in tokens]
    return torch.tensor(ids, dtype=torch.long)


def collate_calculations(batch, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # batch elements are already LongTensors of ids
    fulls = []
    for seq_ids, ans_ids in batch:
        dot_id = stoi['.']
        full = torch.cat([seq_ids, ans_ids, torch.tensor([dot_id], dtype=torch.long)], dim=0)
        fulls.append(full[:max_len])

    max_t = max(int(f.numel()) for f in fulls)
    if max_t < 2:
        max_t = 2
    pad_id = stoi['.']
    batch_input = torch.full((len(fulls), max_t - 1), fill_value=pad_id, dtype=torch.long)
    batch_target = torch.full((len(fulls), max_t - 1), fill_value=-1, dtype=torch.long)
    for i, full in enumerate(fulls):
        if full.numel() < 2:
            continue
        x = full[:-1]
        y = full[1:]
        L = x.numel()
        batch_input[i, :L] = x
        batch_target[i, :L] = y
    return batch_input, batch_target


@torch.no_grad()
def generate(model: NLM, idx: torch.Tensor, max_new_tokens: int, max_len: int) -> torch.Tensor:
    # idx: (B, T)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_len:]
        logits, _ = model(idx_cond, targets=None)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
        # Early stop if all are '.'
        if torch.all(next_token.squeeze(1) == stoi['.']):
            break
    return idx


def extract_predicted_answer(seq_generated: torch.Tensor) -> List[str]:
    # Return contiguous digits until first non-digit or '.'
    digits = []
    for token_id in seq_generated.tolist():
        ch = itos[int(token_id)]
        if ch == '.':
            break
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return digits


@torch.no_grad()
def evaluate_arithmetic(model: NLM, device: torch.device, num_digits: int, op_symbol: str, batch_size: int, max_len: int):
    assert op_symbol in ['+', '*']
    import operator
    op = operator.add if op_symbol == '+' else operator.mul
    dataset = ArithmeticDataset(num_digits, op=op, symbol=op_symbol)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x)

    total = 0
    correct = 0
    # Max answer length heuristic
    max_answer_len = (num_digits + 1) if op_symbol == '+' else (num_digits * 2)
    for batch in loader:
        tokens_batch, ans_batch = zip(*batch)
        tokens_batch = list(tokens_batch)
        ans_batch = list(ans_batch)
        max_t = max(int(t.numel()) for t in tokens_batch)
        idx = torch.full((len(tokens_batch), max_t), fill_value=stoi['.'], dtype=torch.long)
        for i, t in enumerate(tokens_batch):
            idx[i, :t.numel()] = t
        idx = idx.to(device)

        gen = generate(model, idx, max_new_tokens=max_answer_len + 2, max_len=max_len)  # +2 for safety
        gen_tail = gen[:, max_t:]

        # Compare answers
        for i in range(gen_tail.size(0)):
            pred_digits = extract_predicted_answer(gen_tail[i])
            true_digits = [itos[int(x)] for x in ans_batch[i].tolist()]
            if pred_digits == true_digits:
                correct += 1
            total += 1

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_and_save(model: NLM, device: torch.device, num_digits: int, op_symbol: str, batch_size: int, max_len: int, log_dir: str, step: int, max_examples: int = 5000):
    assert op_symbol in ['+', '*']
    import operator
    op = operator.add if op_symbol == '+' else operator.mul
    dataset = ArithmeticDataset(num_digits, op=op, symbol=op_symbol)

    # Sample if too many cases
    indices = list(range(len(dataset)))
    if len(indices) > max_examples:
        indices = random.sample(indices, max_examples)
    ds = Subset(dataset, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x)

    # Prepare output paths
    out_dir = os.path.join(log_dir, 'eval', f'step_{step}')
    os.makedirs(out_dir, exist_ok=True)
    details_path = os.path.join(out_dir, f'{op_symbol}_digits{num_digits}_details.csv')

    total = 0
    correct = 0
    max_answer_len = (num_digits + 1) if op_symbol == '+' else (num_digits * 2)

    with open(details_path, 'w') as f:
        f.write('x,op,y,true,pred,correct\n')
        for batch in loader:
            tokens_batch, ans_batch = zip(*batch)
            tokens_batch = list(tokens_batch)
            ans_batch = list(ans_batch)
            max_t = max(int(t.numel()) for t in tokens_batch)
            idx = torch.full((len(tokens_batch), max_t), fill_value=stoi['.'], dtype=torch.long)
            for i, t in enumerate(tokens_batch):
                idx[i, :t.numel()] = t
            idx = idx.to(device)

            gen = generate(model, idx, max_new_tokens=max_answer_len + 2, max_len=max_len)
            gen_tail = gen[:, max_t:]

            for i in range(gen_tail.size(0)):
                # Parse x, y from tokens
                chars = [itos[int(tok)] for tok in tokens_batch[i].tolist()]
                s = ''.join(chars)
                pos_eq = s.find('=')
                left = s[:pos_eq].rstrip() if pos_eq != -1 else s.strip()
                parts = left.split(f' {op_symbol} ')
                x_str = parts[0]
                y_str = parts[1] if len(parts) > 1 else ''

                # True and predicted answers
                true_digits = [itos[int(x)] for x in ans_batch[i].tolist()]
                true_str = ''.join(true_digits)
                pred_digits = extract_predicted_answer(gen_tail[i])
                pred_str = ''.join(pred_digits)

                is_correct = int(pred_digits == true_digits)
                correct += is_correct
                total += 1
                f.write(f"{x_str},{op_symbol},{y_str},{true_str},{pred_str},{is_correct}\n")

    acc = correct / max(total, 1)
    return acc


def main():
    parser = argparse.ArgumentParser()
    # Optimization
    parser.add_argument('--total-steps', type=int, default=10_000_000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # Model/data
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--max-len', type=int, default=40)

    parser.add_argument('--calc-dataset-size', type=int, default=1_000_000)
    parser.add_argument('--calc-p', type=float, default=0.7)

    # Eval
    parser.add_argument('--eval-every', type=int, default=100_000)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--eval-max-examples', type=int, default=5000)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=str, default=os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'hparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device(args.device)

    # Model config
    cfg = ModelConfig()
    cfg.d_model = args.d_model
    cfg.vocab_size = len(stoi)
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.dropout = args.dropout
    cfg.max_len = args.max_len

    model = NLM(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Data
    train_dataset = Calculations(dataset_size=args.calc_dataset_size, p=args.calc_p)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=lambda x: x)
    train_iter = iter(train_loader)

    # Logging
    train_log_path = os.path.join(args.log_dir, 'train_log.csv')
    eval_log_path = os.path.join(args.log_dir, 'eval_log.csv')
    if not os.path.exists(train_log_path):
        with open(train_log_path, 'w') as f:
            f.write('step,loss,lr,tokens_per_sec\n')
    if not os.path.exists(eval_log_path):
        with open(eval_log_path, 'w') as f:
            f.write('step,op,num_digits,accuracy\n')

    model.train()
    step = 0
    running_tokens = 0
    t0 = time.time()
    pbar = tqdm(total=args.total_steps, dynamic_ncols=True)
    while step < args.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = collate_calculations(batch, max_len=args.max_len)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        running_tokens += x.numel()

        # Progress bar update
        elapsed = time.time() - t0
        tps = running_tokens / max(1e-9, elapsed)
        pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", tps=f"{tps:.1f}")
        pbar.update(1)

        if step % 100 == 0:
            with open(train_log_path, 'a') as f:
                f.write(f"{step},{loss.item():.6f},{scheduler.get_last_lr()[0]:.8f},{tps:.2f}\n")

        if (step % args.eval_every == 0) or (step == args.total_steps):
            model.eval()
            for op in ['+', '*']:
                for nd in [1, 2, 3]:
                    acc = evaluate_and_save(
                        model,
                        device,
                        num_digits=nd,
                        op_symbol=op,
                        batch_size=args.eval_batch_size,
                        max_len=args.max_len,
                        log_dir=args.log_dir,
                        step=step,
                        max_examples=args.eval_max_examples,
                    )
                    with open(eval_log_path, 'a') as f:
                        f.write(f"{step},{op},{nd},{acc:.6f}\n")
                    pbar.write(f"[eval] step={step} op={op} digits={nd} acc={acc:.4f}")
            # Optionally save checkpoint
            ckpt_path = os.path.join(args.log_dir, f"model_step_{step}.pt")
            torch.save({'model': model.state_dict(), 'cfg': vars(cfg), 'step': step}, ckpt_path)
            model.train()

    pbar.close()
    print('Training complete')


if __name__ == '__main__':
    main()


