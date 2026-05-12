from __future__ import annotations

import itertools
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio as ta
from dotenv import load_dotenv

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

load_dotenv(API_ROOT / ".env")

from app.config import Config, detect_device  # noqa: E402
from app.core.engine import load_t3_state_dict, resolve_t3_checkpoint  # noqa: E402
from app.core.text_processing import concatenate_audio_chunks, split_text_into_chunks  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

DEFAULT_OUTPUT_DIR = API_ROOT / "output" / "checkpoint_comparison"
VOICE_DIR = API_ROOT / "voices"
VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# Edit these values, then run:
#   python scripts\compare_checkpoints.py
CHECKPOINTS = [
    {
        "name": "t3-ar-1ep",
        "path": "../runs/t3-ar-1ep/t3_mtl23ls_v2.safetensors",
    },
    {
        "name": "t3-ar-50ep-checkpoint-90000",
        "path": "../runs/t3-ar-50ep/checkpoint-90000",
    },
    {
        "name": "t3-ar-50ep",
        "path": "../runs/t3-ar-50ep/t3_mtl23ls_v2.safetensors",
    },
    {
        "name": "t3-ar-eg-50ep",
        "path": "../runs/t3-ar-eg-50ep/pretrained_model_download/t3_mtl23ls_v2.safetensors",
    },
    {
        "name": "t3-sa-male-eg-15ep",
        "path": "../runs/t3-sa-male-eg-15ep/pretrained_model_download/t3_mtl23ls_v2.safetensors",
    },
]

TEXTS = [
    {
        "name": "egyptian",
        "language_id": "ar",
        "text": """
        مساء الفل يا فندم مع حضرتك احمد الجندي مستشار عقاري في شركة متخصصة في المشروعات السكنية والاستثمارية في القاهرة الجديدة والعاصمة الادارية

حبيت اعرف حضرتك بنفسي بسرعة وافهم احتياجك بشكل دقيق علشان اقدر ارشح لك انسب اختيار سواء للسكن او للاستثمار

المشروع اللي بكلمك عنه موجود في موقع مميز جدا قريب من الطرق الرئيسية والمحاور المهمة وده بيسهل الحركة والانتقال بشكل كبير

الكمباوند فيه مساحات متنوعة شقق دوبلكسات وبنتهاوس والتشطيب فيه مستوى عالي جدا وكمان فيه انظمة سداد مرنة تبدأ بمقدم بسيط وتقسيط على عدة سنين

من الحاجات اللي مميزة المشروع فعلا ان فيه مناطق خضراء واسعة جيم كلوب هاوس تراك للجري واماكن مخصصة للاطفال

ولو حضرتك بتفكر في الاستثمار فالمشروع عليه طلب عالي جدا خصوصا ان المنطقة سعر المتر فيها بيزيد بشكل ملحوظ كل فترة

على فكرة عندنا وحدات باطلالات مختلفة لاجون جاردن وكورنر وفيه كمان وحدات جاهزة للاستلام الفوري

احب اعرف من حضرتك بتدور على كام غرفة وايه الميزانية المناسبة ليك

وانا هشرح لك كل التفاصيل بكل وضوح من غير اي ضغط او لف ودوران علشان تكون واخد قرار مرتاح ومقتنع مية في المية

متشرف جدا بالتعامل مع حضرتك وان شاء الله نلاقي الوحدة المناسبة اللي تناسب ذوقك واحتياجات اسرتك
""",
    },
    {
        "name": "saudi",
        "language_id": "ar",
        "text": """
        السلام عليكم ورحمة الله وبركاته حياك الله يا استاذ معك خالد العتيبي مستشار عقاري مختص بالمشاريع السكنية الحديثة في الرياض والخبر وجدة

ابغى في البداية اعرفك بنفسي بشكل سريع وافهم احتياجك بالتفصيل عشان اقدر اوفر لك الخيار المناسب سواء للسكن العائلي او للاستثمار طويل المدى

المشروع اللي اتكلم عنه اليوم يعتبر من المشاريع المميزة جدا من ناحية الموقع وجودة البناء والخدمات المتوفرة داخله

المشروع قريب من الطرق الرئيسية والمدارس والمستشفيات والمولات وهذا الشي يعطيه قيمة عالية وراحة كبيرة للسكان

عندنا خيارات متعددة شقق تاون هاوس فلل مستقلة ومساحات مختلفة تناسب العوايل الصغيرة والكبيرة

كذلك فيه انظمة دفع مرنة جدا تقدر تبدأ بدفعة اولى مناسبة والباقي على اقساط شهرية او ربع سنوية حسب الخطة اللي تناسبك

من الاشياء الجميلة بالمشروع وجود مساحات خضراء نادي رياضي مسارات للمشي جلسات خارجية ومناطق العاب للاطفال

واذا كنت مهتم بالاستثمار فالمشروع عليه طلب ممتاز خصوصا مع التطور الكبير اللي تشهده المنطقة وارتفاع الاسعار بشكل مستمر

بعض الوحدات تتميز باطلالة مباشرة على الحديقة او المسبح وفيه وحدات جاهزة للتسليم الفوري ووحدات تحت الانشاء باسعار منافسة

اذا تسمح لي ابغى اعرف كم عدد الغرف اللي تحتاجها وهل تفضل السكن داخل مجمع هادي او قريب من المناطق الحيوية

وانا باذن الله اوضح لك كل التفاصيل بكل شفافية من المساحات الى الضمانات والخدمات عشان تكون الصورة واضحة بالكامل قبل اتخاذ القرار

وتشرفنا بخدمتك وان شاء الله نساعدك تحصل على العقار المناسب اللي يليق فيك وفي عايلتك""",
    },
]

# Use all audio files in chatterbox-tts-api\voices when this is empty.
# To test only some voices, use names or paths, for example:
# VOICES = ["nadia_speaker", "fahd_speaker.wav"]
VOICES: list[str] = []

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MANIFEST_NAME = "manifest.json"
DIAGNOSTICS_NAME = "diagnostics.jsonl"
SEED = None
DEVICE = None
ALLOW_PARTIAL_CHECKPOINT_LOAD = True

# Baseline: every checkpoint x every voice x every text uses the values in .env.
ENV_PARAMETERS = {
    "temperature": Config.TEMPERATURE,
    "exaggeration": Config.EXAGGERATION,
    "cfg_weight": Config.CFG_WEIGHT,
}

# Extra sweep: use this to tune parameters for one checkpoint/voice/text.
# Leave PARAMETER_SWEEPS empty if you only want the .env values.
PARAMETER_SWEEPS = [
    # {
    #     "checkpoint_name": "t3-ar-eg-50ep",
    #     "voice_names": ["nadia_speaker"],
    #     "text_names": ["egyptian", "saudi"],
    #     "temperature_values": [0.6, 0.8, 1.0],
    #     "exaggeration_values": [0.45, 0.6],
    #     "cfg_weight_values": [0.35, 0.5],
    # },
]

REPETITION_PENALTY = 2.0
MIN_P = 0.05
TOP_P = 1.0


def resolve_path(path: str | Path, base_dir: Path = API_ROOT) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "item"


def validate_parameter(name: str, value: float, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")


def validate_generation_args(args: dict[str, float]) -> None:
    validate_parameter("TEMPERATURE", args["temperature"], 0.05, 5.0)
    validate_parameter("EXAGGERATION", args["exaggeration"], 0.25, 2.0)
    validate_parameter("CFG_WEIGHT", args["cfg_weight"], 0.0, 1.0)


def discover_voices() -> list[dict[str, str]]:
    voices = []
    for path in sorted(VOICE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS:
            voices.append({"name": path.stem, "path": str(path)})
    return voices


def resolve_voice(value: str) -> dict[str, str]:
    path = resolve_path(value)
    if not path.exists():
        path = VOICE_DIR / value
    if not path.exists() and path.suffix == "":
        matches = [
            candidate
            for candidate in VOICE_DIR.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in VOICE_EXTENSIONS
            and candidate.stem == value
        ]
        if matches:
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"Voice not found: {value}")
    return {"name": path.stem, "path": str(path)}


def selected_voices() -> list[dict[str, str]]:
    return [resolve_voice(voice) for voice in VOICES] if VOICES else discover_voices()


def static_generation_args(params: dict[str, float]) -> dict[str, float]:
    return {
        "exaggeration": params["exaggeration"],
        "cfg_weight": params["cfg_weight"],
        "temperature": params["temperature"],
        "repetition_penalty": REPETITION_PENALTY,
        "min_p": MIN_P,
        "top_p": TOP_P,
    }


def build_sweep_jobs(
    checkpoint: dict[str, str],
    voices_by_name: dict[str, dict[str, str]],
    texts_by_name: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    jobs = []
    for sweep in PARAMETER_SWEEPS:
        if sweep["checkpoint_name"] != checkpoint["name"]:
            continue

        for (
            voice_name,
            text_name,
            temperature,
            exaggeration,
            cfg_weight,
        ) in itertools.product(
            sweep["voice_names"],
            sweep["text_names"],
            sweep["temperature_values"],
            sweep["exaggeration_values"],
            sweep["cfg_weight_values"],
        ):
            if voice_name not in voices_by_name:
                raise ValueError(f"Sweep voice is not selected/found: {voice_name}")
            if text_name not in texts_by_name:
                raise ValueError(f"Sweep text name is not configured: {text_name}")
            params = {
                "temperature": temperature,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
            validate_generation_args(params)
            jobs.append(
                {
                    "kind": "sweep",
                    "checkpoint": checkpoint,
                    "voice": voices_by_name[voice_name],
                    "text": texts_by_name[text_name],
                    "params": params,
                }
            )
    return jobs


def build_jobs() -> list[dict[str, Any]]:
    if not CHECKPOINTS:
        raise ValueError("Set CHECKPOINTS at the top of this file before running it.")
    if not TEXTS:
        raise ValueError("Set TEXTS at the top of this file before running it.")
    validate_generation_args(ENV_PARAMETERS)
    voices = selected_voices()
    if not voices:
        raise ValueError(f"No audio voices found in {VOICE_DIR}")
    texts_by_name = {text["name"]: text for text in TEXTS}
    voices_by_name = {voice["name"]: voice for voice in voices}
    jobs = []
    for checkpoint in CHECKPOINTS:
        for voice in voices:
            for text in TEXTS:
                jobs.append(
                    {
                        "kind": "env",
                        "checkpoint": checkpoint,
                        "voice": voice,
                        "text": text,
                        "params": ENV_PARAMETERS,
                    }
                )
        jobs.extend(build_sweep_jobs(checkpoint, voices_by_name, texts_by_name))
    return jobs


def unique_wav_path(output_dir: Path, parts: list[str], used_names: set[str]) -> Path:
    base_name = "__".join(slug(part) for part in parts) + ".wav"
    name = base_name
    index = 2
    key = str((output_dir / name).resolve()).lower()
    while key in used_names:
        stem = Path(base_name).stem
        name = f"{stem}_{index}.wav"
        key = str((output_dir / name).resolve()).lower()
        index += 1
    used_names.add(key)
    return output_dir / name


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_audio(
    model: ChatterboxMultilingualTTS,
    *,
    text: str,
    voice_sample_path: Path,
    language_id: str,
    generation_args: dict[str, float],
) -> torch.Tensor:
    chunks = split_text_into_chunks(text, Config.MAX_CHUNK_LENGTH)
    audio_chunks = []
    for chunk in chunks:
        wav = model.generate(
            text=chunk,
            language_id=language_id,
            audio_prompt_path=str(voice_sample_path),
            **generation_args,
        )
        audio_chunks.append(wav.detach() if hasattr(wav, "detach") else wav)
    if len(audio_chunks) == 1:
        return audio_chunks[0]
    return concatenate_audio_chunks(audio_chunks, model.sr)


def load_checkpoint_into_model(
    model: ChatterboxMultilingualTTS,
    checkpoint_path: Path,
) -> dict[str, Any]:
    state_dict = load_t3_state_dict(checkpoint_path)
    missing, unexpected = model.t3.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected}")
    if missing and not ALLOW_PARTIAL_CHECKPOINT_LOAD:
        raise RuntimeError(f"Checkpoint is missing keys: {missing}")
    model.t3.eval()
    return {
        "loaded_key_count": len(state_dict),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_key_examples": missing[:20],
    }


def write_diagnostic(
    diagnostics_path: Path,
    *,
    level: str,
    event: str,
    details: dict[str, Any],
) -> None:
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **details,
    }
    with diagnostics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / DIAGNOSTICS_NAME
    if diagnostics_path.exists():
        diagnostics_path.unlink()

    device = DEVICE or detect_device()
    rng = random.Random(SEED)
    jobs = build_jobs()
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / MANIFEST_NAME),
        "diagnostics_path": str(diagnostics_path),
        "device": device,
        "seed": SEED,
        "env_parameters": ENV_PARAMETERS,
        "parameter_sweeps": PARAMETER_SWEEPS,
        "static_generation_args": {
            "repetition_penalty": REPETITION_PENALTY,
            "min_p": MIN_P,
            "top_p": TOP_P,
        },
        "results": [],
    }
    print(f"Loading multilingual base model on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    used_names: set[str] = set()
    loaded_checkpoint_path: Path | None = None
    for index, job in enumerate(jobs, start=1):
        checkpoint = job["checkpoint"]
        voice = job["voice"]
        text_item = job["text"]
        params = job["params"]
        checkpoint_path = resolve_t3_checkpoint(checkpoint["path"])
        voice_path = resolve_path(voice["path"])
        generation_args = static_generation_args(params)
        generation_seed = rng.randrange(0, 2**31)
        checkpoint_dir = output_dir / slug(checkpoint["name"])
        voice_dir = checkpoint_dir / slug(voice["name"])

        wav_path = unique_wav_path(
            voice_dir,
            [
                f"{index:04d}",
                job["kind"],
                text_item["name"],
                f"temp-{params['temperature']}",
                f"exag-{params['exaggeration']}",
                f"cfg-{params['cfg_weight']}",
            ],
            used_names,
        )
        result: dict[str, Any] = {
            "kind": job["kind"],
            "checkpoint_name": checkpoint["name"],
            "checkpoint_input": checkpoint["path"],
            "checkpoint_path": str(checkpoint_path),
            "voice_name": voice["name"],
            "voice_path": str(voice_path),
            "text_name": text_item["name"],
            "text": text_item["text"],
            "language_id": text_item["language_id"],
            "wav_path": str(wav_path),
            "generation_seed": generation_seed,
            "generation_args": generation_args,
            "status": "pending",
        }
        manifest["results"].append(result)
        try:
            if loaded_checkpoint_path != checkpoint_path:
                print(f"Loading checkpoint: {checkpoint['name']} -> {checkpoint_path}")
                checkpoint_load_info = load_checkpoint_into_model(model, checkpoint_path)
                if checkpoint_load_info["missing_key_count"]:
                    warning_message = (
                        "Partial checkpoint load: "
                        f"loaded={checkpoint_load_info['loaded_key_count']} "
                        f"missing={checkpoint_load_info['missing_key_count']} "
                        "unexpected=0"
                    )
                    print(warning_message)
                    write_diagnostic(
                        diagnostics_path,
                        level="warning",
                        event="partial_checkpoint_load",
                        details={
                            "checkpoint_name": checkpoint["name"],
                            "checkpoint_path": str(checkpoint_path),
                            "load_info": checkpoint_load_info,
                            "message": warning_message,
                        },
                    )
                loaded_checkpoint_path = checkpoint_path
            else:
                checkpoint_load_info = {
                    "loaded_key_count": None,
                    "missing_key_count": None,
                    "unexpected_key_count": None,
                    "missing_key_examples": [],
                }

            print(
                f"[{index}/{len(jobs)}] {checkpoint['name']} | {voice['name']} | "
                f"{text_item['name']} | {job['kind']} -> {wav_path.name}"
            )
            set_generation_seed(generation_seed)
            with torch.no_grad():
                wav = generate_audio(
                    model,
                    text=text_item["text"],
                    voice_sample_path=voice_path,
                    language_id=text_item["language_id"],
                    generation_args=generation_args,
                )
            voice_dir.mkdir(parents=True, exist_ok=True)
            ta.save(str(wav_path), wav.cpu(), model.sr, format="wav")
            result.update(
                {
                    "status": "ok",
                    "checkpoint_load": checkpoint_load_info,
                    "sample_rate": model.sr,
                    "chunks": len(
                        split_text_into_chunks(
                            text_item["text"], Config.MAX_CHUNK_LENGTH
                        )
                    ),
                }
            )
        except Exception as exc:
            result.update({"status": "error", "error": str(exc)})
            print(
                f"Failed {checkpoint['name']} / {voice['name']} / {text_item['name']}: {exc}"
            )
            write_diagnostic(
                diagnostics_path,
                level="error",
                event="generation_failed",
                details={
                    "checkpoint_name": checkpoint["name"],
                    "checkpoint_input": checkpoint["path"],
                    "checkpoint_path": str(checkpoint_path),
                    "voice_name": voice["name"],
                    "voice_path": str(voice_path),
                    "text_name": text_item["name"],
                    "language_id": text_item["language_id"],
                    "wav_path": str(wav_path),
                    "generation_seed": generation_seed,
                    "generation_args": generation_args,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        with (output_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved manifest: {output_dir / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
