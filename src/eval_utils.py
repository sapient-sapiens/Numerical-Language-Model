import os
import random
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

from src.data import ArithmeticDataset, stoi, itos
from src.model import NLM


@torch.no_grad()
def generate(model: NLM, idx: torch.Tensor, max_new_tokens: int, max_len: int) -> torch.Tensor:
    # idx: (B, T)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_len:]
        logits, _ = model(idx_cond, targets=None)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
        if torch.all(next_token.squeeze(1) == stoi['.']):
            break
    return idx


def extract_predicted_answer(seq_generated: torch.Tensor) -> List[str]:
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


def _parse_xy_from_tokens(tokens: torch.Tensor) -> tuple[str, str]:
    """Parse x, y from RPN tokens: x ' ' y ' ' op ' ' '=' ' '."""
    chars = [itos[int(tok)] for tok in tokens.tolist()]
    if '=' in chars:
        eq_idx = chars.index('=')
        prefix = chars[:eq_idx]
    else:
        prefix = chars

    j = 0
    x_chars = []
    while j < len(prefix) and prefix[j].isdigit():
        x_chars.append(prefix[j])
        j += 1
    if j < len(prefix) and prefix[j] == ' ':
        j += 1

    y_chars = []
    while j < len(prefix) and prefix[j].isdigit():
        y_chars.append(prefix[j])
        j += 1

    x_str = ''.join(x_chars)
    y_str = ''.join(y_chars)
    return x_str, y_str


@torch.no_grad()
def evaluate_arithmetic(model: NLM, device: torch.device, num_digits: int, op_symbol: str, batch_size: int, max_len: int):
    assert op_symbol in ['+', '*']
    import operator
    op = operator.add if op_symbol == '+' else operator.mul
    dataset = ArithmeticDataset(num_digits, op=op, symbol=op_symbol)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x)

    total = 0
    correct = 0

    max_answer_len = (num_digits + 1) if op_symbol == '+' else (num_digits * 2)
    for batch in loader:
        tokens_batch, ans_batch = zip(*batch)
        tokens_batch = list(tokens_batch)
        ans_batch = list(ans_batch)
        max_t = max(int(t.numel()) for t in tokens_batch)
        idx = torch.full((len(tokens_batch), max_t), fill_value=stoi['_'], dtype=torch.long)
        for i, t in enumerate(tokens_batch):
            idx[i, :t.numel()] = t
        idx = idx.to(device)

        gen = generate(model, idx, max_new_tokens=max_answer_len + 2, max_len=max_len)  # +2 for safety
        gen_tail = gen[:, max_t:]
        for i in range(gen_tail.size(0)):
            pred_digits = extract_predicted_answer(gen_tail[i])
            true_digits = [itos[int(x)] for x in ans_batch[i].tolist()]
            if pred_digits == true_digits:
                correct += 1
            total += 1

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_and_save(
    model: NLM,
    device: torch.device,
    num_digits: int,
    op_symbol: str,
    batch_size: int,
    max_len: int,
    log_dir: str,
    step: int,
    max_examples: int = 5000,
    write_details: bool = True,
):
    assert op_symbol in ['+', '*']
    import operator
    op = operator.add if op_symbol == '+' else operator.mul
    dataset = ArithmeticDataset(num_digits, op=op, symbol=op_symbol)

    # Sample without materializing all indices (supports huge datasets)
    total_len = len(dataset)
    sample_size = min(max_examples, total_len)
    # random.sample over range is efficient in Python
    sampled_indices = random.sample(range(total_len), sample_size)
    ds = Subset(dataset, sampled_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: x)

    total = 0
    correct = 0
    max_answer_len = (num_digits + 1) if op_symbol == '+' else (num_digits * 2)

    # Writing csv is optional. 
    details_f = None
    if write_details:
        out_dir = os.path.join(log_dir, 'eval', f'step_{step}')
        os.makedirs(out_dir, exist_ok=True)
        details_path = os.path.join(out_dir, f'{op_symbol}_digits{num_digits}_details.csv')
        details_f = open(details_path, 'w')
        details_f.write('x,op,y,true,pred,correct\n')

    try:
        for batch in loader:
            tokens_batch, ans_batch = zip(*batch)
            tokens_batch = list(tokens_batch)
            ans_batch = list(ans_batch)
            max_t = max(int(t.numel()) for t in tokens_batch)
            idx = torch.full((len(tokens_batch), max_t), fill_value=stoi['_'], dtype=torch.long)
            for i, t in enumerate(tokens_batch):
                idx[i, :t.numel()] = t
            idx = idx.to(device)

            gen = generate(model, idx, max_new_tokens=max_answer_len + 2, max_len=max_len)
            gen_tail = gen[:, max_t:]

            for i in range(gen_tail.size(0)):
                true_digits = [itos[int(x)] for x in ans_batch[i].tolist()]
                pred_digits = extract_predicted_answer(gen_tail[i])

                is_correct = int(pred_digits == true_digits)
                correct += is_correct
                total += 1

                if details_f is not None:
                    x_str, y_str = _parse_xy_from_tokens(tokens_batch[i])
                    true_str = ''.join(true_digits)
                    pred_str = ''.join(pred_digits)
                    details_f.write(f"{x_str},{op_symbol},{y_str},{true_str},{pred_str},{is_correct}\n")
    finally:
        if details_f is not None:
            details_f.close()

    acc = correct / max(total, 1)
    return acc


