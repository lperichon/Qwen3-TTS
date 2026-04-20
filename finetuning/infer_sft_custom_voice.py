import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import argparse

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/checkpoint-epoch-2")
    parser.add_argument("--output_path", type=str, default="output.wav")
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--text", type=str, default="She said she would be here by noon.")
    args = parser.parse_args()

    device = "cuda:0"
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
    )
    
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker_name,
    )
    sf.write(args.output_path, wavs[0], sr)

if __name__ == "__main__":
    generate()