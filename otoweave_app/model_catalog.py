from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelDisclosure:
    key: str
    name: str
    purpose: str
    license_name: str
    source_url: str
    required: bool
    available: bool

    @property
    def status(self) -> str:
        if self.available:
            return "利用可能"
        return "未配置" if self.required else "未配置（任意）"


def _reazon_model_available() -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    root = Path(os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE))
    repository = root / "models--reazon-research--reazonspeech-k2-v2" / "snapshots"
    return any(repository.glob("*/encoder-epoch-99-avg-1.int8.onnx"))


def model_disclosures(project_root: Path) -> list[ModelDisclosure]:
    root = Path(project_root)
    parakeet = root / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
    speechbrain = root / "models" / "speechbrain-lang-id-voxlingua107-ecapa-onnx"
    qwen = root / "models" / "qwen3-asr-gguf"
    diarization = root / "models" / "diarization"
    return [
        ModelDisclosure(
            key="qwen3.5-4b",
            name="Qwen3.5 4B Q4_K_M",
            purpose="文字起こしの要約",
            license_name="Apache-2.0",
            source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
            required=False,
            available=(root / "models" / "Qwen3.5-4B-Q4_K_M.gguf").is_file(),
        ),
        ModelDisclosure(
            key="qwen3.5-9b",
            name="Qwen3.5 9B Q4_K_M",
            purpose="文字起こしの要約（メモリ16GB級PC向けの上位オプション）",
            license_name="Apache-2.0",
            source_url="https://huggingface.co/Qwen/Qwen3.5-9B",
            required=False,
            available=(root / "models" / "Qwen3.5-9B-Q4_K_M.gguf").is_file(),
        ),
        ModelDisclosure(
            key="qwen3.5-2b",
            name="Qwen3.5 2B Q4_K_M",
            purpose="AIチューター",
            license_name="Apache-2.0",
            source_url="https://huggingface.co/Qwen/Qwen3.5-2B",
            required=False,
            available=(root / "models" / "Qwen3.5-2B-Q4_K_M.gguf").is_file(),
        ),
        ModelDisclosure(
            key="reazonspeech-k2-v2",
            name="ReazonSpeech K2 v2",
            purpose="日本語音声認識",
            license_name="Apache-2.0",
            source_url="https://huggingface.co/reazon-research/reazonspeech-k2-v2",
            required=True,
            available=_reazon_model_available(),
        ),
        ModelDisclosure(
            key="parakeet-tdt-v2-int8",
            name="NVIDIA Parakeet TDT 0.6B v2 int8",
            purpose="英語音声認識",
            license_name="CC-BY-4.0",
            source_url=(
                "https://huggingface.co/csukuangfj/"
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
            ),
            required=True,
            available=all(
                (parakeet / filename).is_file()
                for filename in (
                    "encoder.int8.onnx",
                    "decoder.int8.onnx",
                    "joiner.int8.onnx",
                    "tokens.txt",
                )
            ),
        ),
        ModelDisclosure(
            key="speechbrain-voxlingua107",
            name="SpeechBrain ECAPA-TDNN VoxLingua107",
            purpose="日英音声言語判定",
            license_name="Apache-2.0",
            source_url="https://huggingface.co/drakulavich/SpeechBrain-coreml",
            required=True,
            available=all(
                (speechbrain / filename).is_file()
                for filename in (
                    "lang-id-ecapa.onnx",
                    "lang-id-ecapa.onnx.data",
                    "labels.json",
                )
            ),
        ),
        ModelDisclosure(
            key="qwen3-asr-1.7b",
            name="Qwen3-ASR-1.7B Q8",
            purpose="日英混在音声認識（実験）",
            license_name="Apache-2.0",
            source_url="https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
            required=False,
            available=all(
                (qwen / filename).is_file()
                for filename in (
                    "Qwen3-ASR-1.7B-Q8_0.gguf",
                    "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
                )
            ),
        ),
        ModelDisclosure(
            key="sherpa-onnx",
            name="sherpa-onnx",
            purpose="ローカル音声認識エンジン",
            license_name="Apache-2.0",
            source_url="https://github.com/k2-fsa/sherpa-onnx",
            required=True,
            available=importlib.util.find_spec("sherpa_onnx") is not None,
        ),
        ModelDisclosure(
            key="llama-cpp-python",
            name="llama-cpp-python",
            purpose="ローカルLLM実行エンジン（要約・AIチューター）",
            license_name="MIT",
            source_url="https://github.com/abetlen/llama-cpp-python",
            required=False,
            available=importlib.util.find_spec("llama_cpp") is not None,
        ),
        ModelDisclosure(
            key="onnxruntime",
            name="ONNX Runtime",
            purpose="ONNXモデル実行エンジン（言語判定・話者分離）",
            license_name="MIT",
            source_url="https://github.com/microsoft/onnxruntime",
            required=True,
            available=importlib.util.find_spec("onnxruntime") is not None,
        ),
        ModelDisclosure(
            key="windows-tts-haruka",
            name="Microsoft Haruka（System.Speech）",
            purpose="読み上げ（TTS）",
            license_name="Windows標準機能（OSに同梱）",
            source_url="https://learn.microsoft.com/dotnet/api/system.speech.synthesis",
            required=False,
            available=os.name == "nt",
        ),
        ModelDisclosure(
            key="macos-tts-kyoko",
            name="macOS標準音声 Kyoko（say）",
            purpose="読み上げ（TTS）",
            license_name="macOS標準機能（OSに同梱）",
            source_url="https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer",
            required=False,
            # TODO(platform_support): platform_support.py 導入後は共通の
            # is_macos() 判定に置き換える。
            available=sys.platform == "darwin",
        ),
        ModelDisclosure(
            key="pyannote-segmentation-3.0",
            name="pyannote segmentation-3.0（sherpa-onnx ONNX変換版）",
            purpose="話者分離（発話区間の検出）",
            license_name="MIT",
            source_url="https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models",
            required=False,
            available=(
                diarization / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
            ).is_file(),
        ),
        ModelDisclosure(
            key="3dspeaker-eres2net",
            name="3D-Speaker ERes2Net（sv_zh-cn_3dspeaker, 16kHz）",
            purpose="話者分離（話者埋め込み・照合）",
            license_name="Apache-2.0",
            source_url="https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models",
            required=False,
            available=(
                diarization / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
            ).is_file(),
        ),
    ]
