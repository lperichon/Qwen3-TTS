# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import statistics
import subprocess
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig

from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _get_gpu_stats():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        line = out.strip().splitlines()[0]
        util_s, mem_s = [x.strip() for x in line.split(",")]
        return float(util_s), float(mem_s)
    except Exception:
        return None, None


def _pick_samples(jsonl_path, batch_size, mode):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        if mode == "first":
            return [json.loads(next(f)) for _ in range(batch_size)]
        samples = []
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
        samples.sort(key=lambda x: len(x.get("audio_codes", [])), reverse=True)
        return samples[:batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_steps", type=int, default=5)
    parser.add_argument("--sample_mode", choices=["first", "longest"], default="longest")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model on {args.device} ...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=args.device,
    )
    config = AutoConfig.from_pretrained(args.model_path)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(qwen3tts.model, lora_config)
    model.train()

    device = next(model.parameters()).device
    print(f"Using device: {device}")

    samples = _pick_samples(args.train_jsonl, args.batch_size, args.sample_mode)
    dataset = TTSDataset(samples, qwen3tts.processor, config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))

    def to_device(x):
        return x.to(device) if torch.is_tensor(x) else x

    input_ids = to_device(batch["input_ids"])
    codec_ids = to_device(batch["codec_ids"])
    ref_mels = to_device(batch["ref_mels"])
    text_embedding_mask = to_device(batch["text_embedding_mask"])
    codec_embedding_mask = to_device(batch["codec_embedding_mask"])
    attention_mask = to_device(batch["attention_mask"])
    codec_0_labels = to_device(batch["codec_0_labels"])
    codec_mask = to_device(batch["codec_mask"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_times = []
    util_samples = []
    mem_samples = []

    total_steps = args.warmup_steps + args.measure_steps
    for step in range(total_steps):
        start = time.perf_counter()

        with torch.no_grad():
            speaker_embedding = model.speaker_encoder(
                ref_mels.to(dtype=next(model.parameters()).dtype, device=device)
            ).detach()

        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = speaker_embedding

        input_embeddings = input_text_embedding + input_codec_embedding
        for i in range(1, 16):
            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids[codec_mask]

        _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
            talker_codec_ids, talker_hidden_states
        )

        loss = outputs.loss + sub_talker_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        util, mem = _get_gpu_stats()

        if step >= args.warmup_steps:
            step_times.append(elapsed)
            if util is not None:
                util_samples.append(util)
            if mem is not None:
                mem_samples.append(mem)

    tokens = codec_0_labels[:, 1:].numel()
    avg_time = statistics.mean(step_times)
    p50_time = statistics.median(step_times)
    p90_time = statistics.quantiles(step_times, n=10)[8] if len(step_times) >= 10 else max(step_times)
    tokens_per_sec = tokens / avg_time

    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    print("---- Benchmark ----")
    print(f"Batch size: {args.batch_size}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Tokens per step: {tokens}")
    print(f"Avg step time: {avg_time:.3f}s (p50 {p50_time:.3f}s, p90 {p90_time:.3f}s)")
    print(f"Tokens/sec: {tokens_per_sec:.1f}")
    print(f"Peak VRAM allocated: {max_alloc:.2f} GB")
    print(f"Peak VRAM reserved:  {max_reserved:.2f} GB")
    if util_samples:
        print(f"GPU util avg: {statistics.mean(util_samples):.1f}% (max {max(util_samples):.1f}%)")
    if mem_samples:
        print(f"GPU mem sample avg: {statistics.mean(mem_samples)/1024:.2f} GB (max {max(mem_samples)/1024:.2f} GB)")


if __name__ == "__main__":
    main()
