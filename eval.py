#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re
import sys
from typing import List, Tuple

import torch

from src.model import NLM, ModelConfig
from src.eval_utils import evaluate_and_save


def find_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    ckpts = []
    for path in glob.glob(os.path.join(run_dir, 'model_step_*.pt')):
        m = re.search(r"model_step_(\d+)\.pt$", os.path.basename(path)) 
        if not m:
            continue
        step = int(m.group(1))
        ckpts.append((step, path))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[NLM, int]:
    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload.get('cfg', {})
    cfg = ModelConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = NLM(cfg).to(device)
    state = payload['model']
    needs_unwrap = any(k.startswith('_orig_mod.') for k in state.keys())
    if needs_unwrap:
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    step = int(payload.get('step', 0))
    return model, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_dir = args.run_dir
    device = torch.device(args.device)

    hp_path = os.path.join(run_dir, 'hparams.json')
    assert os.path.exists(hp_path), 'hparams.json not found in run dir'
    with open(hp_path) as f:
        hp = json.load(f)
    eval_batch_size = int(hp.get('eval_batch_size', 256))
    eval_max_examples = int(hp.get('eval_max_examples', 5000))
    max_len = int(hp.get('max_len', 40))

    # Prepare eval log fresh
    eval_log_path = os.path.join(run_dir, 'eval_log.csv')
    with open(eval_log_path, 'w') as f:
        f.write('step,op,num_digits,accuracy\n')
    eval_root = os.path.join(run_dir, 'eval')
    os.makedirs(eval_root, exist_ok=True)
    for step, ckpt_path in find_checkpoints(run_dir):
        print(f"[rerun] step={step} ckpt={ckpt_path}")
        step_dir = os.path.join(eval_root, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)
        model, saved_step = load_model_from_ckpt(ckpt_path, device)
        model.eval()
        assert saved_step == step or saved_step == 0
        for op in ['+', '*']:
            for nd in [1, 2, 3, 4, 5, 6]:
                acc = evaluate_and_save(
                    model,
                    device,
                    num_digits=nd,
                    op_symbol=op,
                    batch_size=eval_batch_size,
                    max_len=max_len,
                    log_dir=run_dir,
                    step=step,
                    max_examples=eval_max_examples,
                )
                with open(eval_log_path, 'a') as f:
                    f.write(f"{step},{op},{nd},{acc:.6f}\n")
                print(f"[eval] step={step} op={op} digits={nd} acc={acc:.4f}")


if __name__ == '__main__':
    main()


