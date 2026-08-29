#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep codec sampling overrides against known-bad Seed-TTS samples.

For each seed value, writes /tmp/codec_override.json, re-synthesizes the
failing samples and scores them with the official ASR pipeline. Reports a
per-(seed, sample) WER matrix so a seed that fixes the deterministic codec
mispronunciations can be picked and frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minicpmo_zh_wer_selfcheck import (  # noqa: E402
    SAMPLE_RATE,
    _jiwer_wer,
    _load_meta,
    _pcm_to_wav,
    _synth_chat,
    _transcribe,
)

OVERRIDE_PATH = Path("/tmp/codec_override.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/seedtts/seedtts_testset"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--indices", type=int, nargs="+", default=[1, 8, 14])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3, 100, 123])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/codec_sweep"))
    args = parser.parse_args()

    url = args.endpoint.rstrip("/") + "/v1/chat/completions"
    rows = _load_meta(args.dataset, max(args.indices) + 1)
    samples = [(i, rows[i]) for i in args.indices]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Warm the ASR model once so per-sample scoring below is fast.
    print("loading ASR model...", flush=True)
    _transcribe(args.out_dir / "warmup.wav" if (args.out_dir / "warmup.wav").is_file() else _write_silence(args.out_dir))

    print(f"{'seed':>6} {'temp':>5} | " + " | ".join(f"idx{i:02d} wer" for i, _ in samples))
    for seed in args.seeds:
        ov: dict = {"seed": seed}
        if args.temperature is not None:
            ov["temperature"] = args.temperature
        OVERRIDE_PATH.write_text(json.dumps(ov))
        wers = []
        for i, (utt, ref_text, ref_wav, target) in samples:
            wav_name = f"seed{seed}_idx{i:02d}.wav"
            try:
                pcm, _ = _synth_chat(url, args.model, target, ref_wav, ref_text, utt, timeout=300.0)
                wav_path = args.out_dir / wav_name
                wav_path.write_bytes(_pcm_to_wav(pcm))
                hyp = _transcribe(wav_path)
                wer = _jiwer_wer(target, hyp)
            except Exception as exc:  # noqa: BLE001
                print(f"  [seed {seed} idx {i}] FAIL: {exc}", flush=True)
                wer = float("nan")
            wers.append(wer)
        t = args.temperature if args.temperature is not None else 0.8
        print(f"{seed:>6} {t:>5} | " + " | ".join(f"{w:9.4%}" for w in wers), flush=True)

    OVERRIDE_PATH.write_text("{}")
    print("restored default overrides")
    return 0


def _write_silence(out_dir: Path) -> Path:
    import wave

    p = out_dir / "warmup.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * SAMPLE_RATE)
    return p


if __name__ == "__main__":
    sys.exit(main())
