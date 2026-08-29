#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local Seed-TTS zh WER gate self-check for MiniCPM-o 4.5 perf changes.

IMPORTANT (request-shape parity): this script replicates the OFFICIAL
competition evaluation request exactly (``vllm_omni/benchmarks/patch/patch.py::
async_request_openai_chat_omni_completions`` + ``SeedTTSDataset``):

  * POST ``/v1/chat/completions`` (backend=openai-chat-omni), NOT /v1/audio/speech
  * messages = [system: SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT,
                user:   [{"type":"text","text": target_text}]]
  * top-level extra_body: ref_audio (inline data:audio/wav;base64,...),
    ref_text, task_type="Base", language="Chinese", max_new_tokens
  * stream=True + stream_options{include_usage, continuous_usage_stats}
  * modalities ["text","audio"], temperature 0.0

Audio is aggregated from per-chunk ``delta.content`` base64 PCM16 (24 kHz
mono) exactly like the official ``StreamedResponseHandler`` path, wrapped
into a RIFF wav, then scored with the official pipeline (funasr
paraformer-zh + zhconv + jiwer via ``seed_tts_eval``). Local numbers are
therefore directly comparable to the official gate.

Usage (server must already be running):

    python tools/minicpmo_zh_wer_selfcheck.py \
        --dataset /workspace/user_data/seed-tts-eval \
        --endpoint http://127.0.0.1:8000 \
        --num-prompts 32

Dataset layout (seed-tts-eval): ``zh/meta.lst`` rows
``<utt_id>|<ref_text>|<wav_rel>|<target_text>``; ref wav is resolved at
``<dataset>/zh/<wav_rel>``. Without --dataset a small built-in smoke list
(no voice clone, no ref_audio) is used for pipeline sanity checks only.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

# Kept byte-identical to vllm_omni/benchmarks/data_modules/seed_tts_dataset.py
SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech.\n"
    "For this request you act as a text-to-speech engine with zero-shot voice cloning: "
    "the API provides reference audio and its transcript (ref_audio, ref_text) and task_type Base. "
    "The user message is the exact text you must speak. "
    "Synthesize natural speech in the same language as that user text, "
    "matching the timbre, prosody, and speaking style of the reference audio while reading the new content clearly."
)

SAMPLE_RATE = 24000


def _load_meta(dataset: Path, num_prompts: int) -> list[tuple[str, str, str, str]]:
    """Return (utt_id, ref_text, ref_wav_path_or_empty, target_text) rows."""
    meta = dataset / "zh" / "meta.lst"
    rows: list[tuple[str, str, str, str]] = []
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            utt, ref_text, wav_rel, target = (p.strip() for p in parts[:4])
            if not target:
                continue
            wav_path = (dataset / "zh" / wav_rel).resolve()
            rows.append((utt, ref_text, str(wav_path) if wav_path.is_file() else "", target))
    else:
        # Smoke-test fallback: short zh utterances in Seed-TTS hard style.
        rows = [
            ("smoke-0001", "", "", "对，这就是我想要的答案，简单直接。"),
            ("smoke-0002", "", "", "今天的会议讨论了三个重点问题。"),
            ("smoke-0003", "", "", "我觉得这个方案在实际落地时会有困难。"),
            ("smoke-0004", "", "", "请把这份报告在下班前发给我。"),
            ("smoke-0005", "", "", "演员的表演非常到位，剧情也很紧凑。"),
        ]
    return rows[:num_prompts]


def _ref_audio_data_url(wav_path: str) -> str:
    b64 = base64.b64encode(Path(wav_path).read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _synth_chat(
    endpoint: str,
    model: str,
    target_text: str,
    ref_wav: str,
    ref_text: str,
    utt: str,
    timeout: float,
    max_new_tokens: int = 2048,
) -> tuple[bytes, float]:
    """Official openai-chat-omni request shape; returns (pcm16 bytes, wall s)."""
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": target_text}]},
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "modalities": ["text", "audio"],
        "language": "Chinese",
    }
    if ref_wav:
        payload["ref_audio"] = _ref_audio_data_url(ref_wav)
        payload["ref_text"] = ref_text
        payload["task_type"] = "Base"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    pcm = bytearray()
    wav_params: tuple[int, int, int] | None = None
    stray_text: list[str] = []
    n_bad_wav = 0
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        done = False
        while not done:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                sline = line.decode("utf-8", "replace").strip()
                if not sline.startswith("data: "):
                    continue
                data = sline[len("data: "):]
                if data == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("modality") != "audio":
                    # capture stray text tokens for diagnostics
                    for ch in obj.get("choices", []):
                        content = (ch.get("delta") or {}).get("content")
                        if isinstance(content, str) and content and content[:1] == "{":
                            stray_text.append(content)
                    continue
                for ch in obj.get("choices", []):
                    delta = ch.get("delta") or {}
                    content = delta.get("content")
                    if not content and isinstance(delta.get("audio"), dict):
                        content = delta["audio"].get("data")
                    if not isinstance(content, str) or not content:
                        continue
                    # Each audio chunk is a COMPLETE standalone wav (RIFF header
                    # + frames), exactly like the official benchmark handler:
                    # parse with wave, drop chunks with inconsistent params.
                    try:
                        audio_bytes = base64.b64decode(content, validate=True)
                        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                            params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
                            if wav_params is None:
                                wav_params = params
                            elif wav_params != params:
                                n_bad_wav += 1
                                continue
                            pcm.extend(wf.readframes(wf.getnframes()))
                    except Exception:  # noqa: BLE001 - text token, not audio
                        stray_text.append(content)
    elapsed = time.perf_counter() - start
    if n_bad_wav:
        print(f"    [warn] dropped {n_bad_wav} wav chunks with inconsistent params")
    if not pcm:
        raise RuntimeError(f"no audio in stream (text={''.join(stray_text)[:120]!r})")
    if wav_params is not None and wav_params[2] != SAMPLE_RATE:
        # resample to 24 kHz mono PCM16 like the official WER capture path
        import numpy as np

        from vllm.multimodal.audio import AudioResampler

        f32 = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32767.0
        f32 = AudioResampler(target_sr=SAMPLE_RATE).resample(f32, orig_sr=wav_params[2])
        pcm = bytearray((np.clip(f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    print(f"    synth {utt[:24]:24s} {elapsed:6.2f}s {len(pcm)/2/SAMPLE_RATE:5.2f}s audio")
    return bytes(pcm), elapsed


def _pcm_to_wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return out.getvalue()


def _transcribe(wav_path: Path) -> str:
    """Official zh transcription: paraformer-zh @16k + zhconv (seed-tts-eval)."""
    import numpy as np

    from vllm_omni.benchmarks.data_modules import seed_tts_eval as ste

    if hasattr(ste, "transcribe_zh"):
        return ste.transcribe_zh(str(wav_path))
    # Official path: resample 24k->16k f32, paraformer, zh-cn convert.
    data, sr = _read_wav_f32(wav_path)
    if sr != 16000:
        from scipy.signal import resample_poly

        data = resample_poly(data, 16000, sr).astype(np.float32)
    ste._ensure_zh_asr()
    from zhconv import convert as zh_convert

    raw = ste._zh_paraformer.generate(input=[np.ascontiguousarray(data)])
    text = raw[0]["text"] if raw else ""
    return zh_convert(text, "zh-cn")


def _read_wav_f32(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def _jiwer_wer(reference: str, hypothesis: str) -> float:
    """Official zh WER: strip punctuation + per-char split (run_wer.process_one).

    Calling ``jiwer.wer`` directly compares the raw strings: paraformer zh
    output is space-separated (``'简 单 的'``) while references are not, so
    jiwer sees 1 ref word vs N hyp words and reports N*100%. Normalize
    exactly like the official evaluation instead.
    """
    from vllm_omni.benchmarks.data_modules import seed_tts_eval as ste

    if hasattr(ste, "process_one_official"):
        return float(ste.process_one_official(hypothesis, reference, "zh")[0])
    if hasattr(ste, "_jiwer_wer"):
        return ste._jiwer_wer(reference, hypothesis)
    import jiwer

    return jiwer.wer(reference, hypothesis)


def _eval_only(args: argparse.Namespace) -> int:
    """Score wavs already synthesized by a --synth-only run (no server needed)."""
    targets_json = args.out_dir / "targets.json"
    if not targets_json.is_file():
        print(f"FAIL: {targets_json} not found; run --synth-only first")
        return 2
    results = [(Path(p), t, u) for p, t, u in json.loads(targets_json.read_text(encoding="utf-8"))]
    print(f"eval-only: {len(results)} wavs in {args.out_dir}, gate <= {args.gate:.4f}")
    wers: list[float] = []
    for wav_path, target, utt in results:
        hyp = _transcribe(wav_path)
        wer = _jiwer_wer(target, hyp)
        wers.append(wer)
        flag = "  <-- high" if wer > args.gate else ""
        print(f"    WER {utt[:24]:24s} {wer:8.4%}  ref={target[:18]!r} hyp={hyp[:18]!r}{flag}")
    mean_wer = sum(wers) / max(len(wers), 1)
    verdict = "PASS" if mean_wer <= args.gate else "FAIL"
    print(f"\nZH WER (mean): {mean_wer:.4%}  gate {args.gate:.4%}  -> {verdict}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/user_data/seed-tts-eval"))
    parser.add_argument(
        "--endpoint",
        required="--eval-only" not in sys.argv,
        help="base URL or full chat URL; /v1/audio/speech is auto-redirected to /v1/chat/completions",
    )
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5", help="served model name")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--gate", type=float, default=0.0156, help="default 1.56%% (0.0156)")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/wer_selfcheck"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--synth-only",
        action="store_true",
        help="only synthesize wavs into --out-dir (no ASR model loaded); pair with --eval-only",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="only run ASR+WER on wavs already in --out-dir (requires targets.json from --synth-only)",
    )
    args = parser.parse_args()

    url = (args.endpoint or "").rstrip("/")
    if url.endswith("/v1/audio/speech"):
        print("[warn] /v1/audio/speech shape is NOT the official eval path; redirecting to /v1/chat/completions")
        url = url[: -len("/v1/audio/speech")] + "/v1/chat/completions"
    elif url and not url.endswith("/v1/chat/completions"):
        url = url + "/v1/chat/completions"
    if url:
        print(f"endpoint (official openai-chat-omni shape): {url}")

    if args.eval_only:
        return _eval_only(args)

    rows = _load_meta(args.dataset, args.num_prompts)
    n_clone = sum(1 for r in rows if r[2])
    print(f"WER self-check: {len(rows)} prompts ({n_clone} voice-clone), gate <= {args.gate:.4f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, str, str]] = []
    wall_audio = 0.0
    wall_synth = 0.0
    for i, (utt, ref_text, ref_wav, target) in enumerate(rows):
        try:
            pcm, elapsed = _synth_chat(url, args.model, target, ref_wav, ref_text, utt, timeout=args.timeout)
            wall_synth += elapsed
            wall_audio += len(pcm) / 2 / SAMPLE_RATE
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {utt}: {exc}")
            return 2
        wav_path = args.out_dir / f"{i:05d}.wav"
        wav_path.write_bytes(_pcm_to_wav(pcm))
        results.append((wav_path, target, utt))
    if wall_audio > 0:
        print(f"synth wall={wall_synth:.2f}s audio={wall_audio:.2f}s RTF={wall_synth/wall_audio:.4f}")
    # Persist targets so a later --eval-only run (possibly after the server is
    # shut down to free its ~34 GB cgroup footprint) can score these wavs.
    (args.out_dir / "targets.json").write_text(
        json.dumps([[str(p), t, u] for p, t, u in results], ensure_ascii=False),
        encoding="utf-8",
    )
    if args.synth_only:
        print(f"[synth-only] wrote {len(results)} wavs + targets.json to {args.out_dir}; run --eval-only next")
        return 0

    wers: list[float] = []
    for wav_path, target, utt in results:
        hyp = _transcribe(wav_path)
        wer = _jiwer_wer(target, hyp)
        wers.append(wer)
        flag = "  <-- high" if wer > args.gate else ""
        print(f"    WER {utt[:24]:24s} {wer:8.4%}  ref={target[:18]!r} hyp={hyp[:18]!r}{flag}")

    mean_wer = sum(wers) / max(len(wers), 1)
    verdict = "PASS" if mean_wer <= args.gate else "FAIL"
    print(f"\nZH WER (mean): {mean_wer:.4%}  gate {args.gate:.4%}  -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
