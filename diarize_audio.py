#!/usr/bin/env python3
"""
Interactive Speaker Diarization CLI
===================================
Segments multi-speaker audio recordings into speaker turns, identifies
who spoke when, and exports standard NIST RTTM and JSON timelines.

Usage:
    python diarize_audio.py --audio outputs/sample_meeting_conversation.wav --num-speakers 3
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
import torch

from main import (
    SpeakerEmbeddingNet,
    LogMelExtractor,
    diarize_audio_recording,
    SAMPLE_RATE
)


def load_audio(audio_path: str) -> np.ndarray:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        tensor_data = torch.from_numpy(data).unsqueeze(0)
        data = resampler(tensor_data).squeeze(0).numpy()
    return data


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS.ss string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization (Who Spoke When)")
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file (.wav, .mp3)')
    parser.add_argument('--model', type=str, default='outputs/speaker_embedding_best.pth', help='Path to trained d-vector encoder')
    parser.add_argument('--num-speakers', type=int, default=3, help='Number of unique speakers')
    parser.add_argument('--output-rttm', type=str, default='outputs/output_diarization.rttm', help='Output RTTM path')
    parser.add_argument('--output-json', type=str, default='outputs/output_timeline.json', help='Output JSON timeline path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] Model checkpoint not found: {args.model}. Run main.py first!")
        sys.exit(1)

    device = torch.device(args.device)
    ckpt = torch.load(args.model, map_location=device)

    model = SpeakerEmbeddingNet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    extractor = LogMelExtractor(sample_rate=SAMPLE_RATE)
    audio = load_audio(args.audio)
    duration_sec = len(audio) / SAMPLE_RATE

    print("\n" + "=" * 65)
    print("      SPEAKER DIARIZATION PIPELINE (WHO SPOKE WHEN)")
    print("=" * 65)
    print(f"Audio File       : {args.audio}")
    print(f"Duration         : {duration_sec:.2f} seconds ({len(audio):,} samples)")
    print(f"Target Speakers  : {args.num_speakers}")
    print(f"Embedding Model  : {args.model}")
    print("-" * 65)

    turns, _ = diarize_audio_recording(
        audio, model, extractor, num_speakers=args.num_speakers, device=device
    )

    print(f"\nDetected {len(turns)} Speaker Turns:")
    print(f"{'Turn':<6}{'Time Window':<24}{'Speaker':<15}{'Duration':<10}")
    print("-" * 65)

    rttm_lines = []
    json_turns = []

    for idx, (st, et, spk) in enumerate(turns, 1):
        dur = et - st
        spk_tag = f"SPEAKER_{spk+1:02d}"
        time_str = f"[{format_timestamp(st)} -> {format_timestamp(et)}]"
        print(f"#{idx:<5}{time_str:<24}{spk_tag:<15}{dur:>5.2f}s")

        rttm_lines.append(f"SPEAKER audio_track 1 {st:.2f} {dur:.2f} <NA> <NA> {spk_tag} <NA> <NA>\n")
        json_turns.append({
            'turn_index': idx,
            'speaker': spk_tag,
            'start_sec': st,
            'end_sec': et,
            'duration_sec': round(dur, 2)
        })

    # Save outputs
    with open(args.output_rttm, 'w') as f:
        f.writelines(rttm_lines)
    with open(args.output_json, 'w') as f:
        json.dump({'file': args.audio, 'turns': json_turns}, f, indent=2)

    print("-" * 65)
    print(f"[✓] Standard NIST RTTM saved to : {args.output_rttm}")
    print(f"[✓] Timeline JSON saved to      : {args.output_json}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()

