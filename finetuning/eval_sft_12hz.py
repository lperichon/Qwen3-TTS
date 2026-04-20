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

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def _get_fallback_speaker_id(model):
    spk_id_map = getattr(getattr(model, "config", None), "talker_config", None)
    spk_id_map = getattr(spk_id_map, "spk_id", {}) if spk_id_map is not None else {}
    if not spk_id_map:
        return None
    if 3000 in spk_id_map.values():
        return 3000
    return list(spk_id_map.values())[-1]


def compute_loss(model, batch):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    with torch.no_grad():
        if getattr(model, "speaker_encoder", None) is not None:
            speaker_embedding = model.speaker_encoder(
                ref_mels.to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
            ).detach()
        else:
            speaker_id = _get_fallback_speaker_id(model)
            if speaker_id is None:
                raise RuntimeError("No speaker_encoder and no speaker id found for evaluation.")
            weight = model.talker.model.codec_embedding.weight
            speaker_embedding = weight[speaker_id].unsqueeze(0).expand(ref_mels.shape[0], -1)

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

    return outputs.loss + sub_talker_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report_every", type=int, default=50)
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
        device_map=args.device,
    )

    model = tts.model
    if args.adapter_path:
        if PeftModel is None:
            raise SystemExit("peft is required to load LoRA adapters (pip install peft).")
        peft_model = PeftModel.from_pretrained(model, args.adapter_path)
        model = peft_model.merge_and_unload() if args.merge_lora else peft_model

    model.eval()

    config = AutoConfig.from_pretrained(args.base_model_path)
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if args.limit is not None:
        data = data[: args.limit]

    dataset = TTSDataset(data, tts.processor, config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    losses = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            loss = compute_loss(model, batch)
            losses.append(loss.item())
            if args.report_every and step % args.report_every == 0:
                print(f"step {step} | loss {loss.item():.4f}")

    if not losses:
        print("No samples to evaluate.")
        return
    avg_loss = sum(losses) / len(losses)
    print(f"Test loss avg: {avg_loss:.4f} over {len(losses)} batches")


if __name__ == "__main__":
    main()
