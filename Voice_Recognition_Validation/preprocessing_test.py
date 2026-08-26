"""
Voice Preprocessing Validation Experiment

Isolated experiment to evaluate whether VAD, Normalization, or VAD + Normalization
improves speaker separation and identification performance using the existing
ECAPA-TDNN encoder and voice profiles.

Production Safety:
- Does NOT modify any production code in Voice_Recognition/, Voice_Cloning/, Voice_Input/, main.py, data/
- Does NOT duplicate audio files or save preprocessed audio to disk.
- All audio processing is in-memory.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from Voice_Recognition.speaker_encoder import SpeakerEncoder
from Voice_Recognition.speaker_preprocessing import SpeakerAudioPreprocessor
from Voice_Recognition.voice_profile_manager import VoiceProfileManager
from Voice_Recognition.speaker_identifier import SpeakerIdentifier


def apply_energy_vad(
    waveform: torch.Tensor,
    frame_len: int = 512,
    hop_len: int = 256,
    threshold_db: float = -25.0,
    sample_rate: int = 16000,
    min_duration_sec: float = 1.0,
) -> torch.Tensor:
    """
    Apply simple frame-level energy-based VAD to extract speech regions in-memory.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    samples = waveform.shape[1]
    if samples < frame_len:
        return waveform

    frames = waveform.squeeze(0).unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
    max_rms = rms.max()
    if max_rms < 1e-6:
        return waveform

    db = 20 * torch.log10(rms / max_rms)
    active_frames = db >= threshold_db

    # Create sample-level active mask
    mask = torch.zeros(samples, dtype=torch.bool)
    num_frames = active_frames.shape[0]
    for i in range(num_frames):
        if active_frames[i]:
            start = i * hop_len
            end = min(start + frame_len, samples)
            mask[start:end] = True

    # Fallback to original waveform if active duration is too short (< min_duration_sec)
    if not torch.any(mask) or mask.sum().item() < sample_rate * min_duration_sec:
        return waveform

    return waveform[:, mask]


def safe_peak_normalize(waveform: torch.Tensor, peak_target: float = 0.95) -> torch.Tensor:
    """
    Apply safe peak amplitude normalization in-memory.
    """
    max_amplitude = torch.max(torch.abs(waveform))
    if max_amplitude > 1e-6:
        waveform = (waveform / max_amplitude) * peak_target
    return waveform


def discover_audio_and_profiles(
    audio_dir: Path, profile_manager: VoiceProfileManager
) -> Tuple[List[Path], Dict[str, Optional[str]], Dict[str, List[Tuple[str, torch.Tensor]]], Dict[str, Optional[str]]]:
    """
    Dynamically discover all audio recordings and map them to known speaker profiles.

    Returns:
        audio_files: List of audio file Paths in data/audio
        file_to_speaker: Mapping from filename -> true speaker user_id or None (UNKNOWN)
        speaker_references: Mapping from user_id -> list of (ref_identifier, embedding_tensor)
        file_to_ref_id: Mapping from filename -> ref_identifier if this file was used to create that reference
    """
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    profiles = profile_manager.list_profiles()

    file_to_speaker: Dict[str, Optional[str]] = {f.name: None for f in audio_files}
    file_to_ref_id: Dict[str, Optional[str]] = {f.name: None for f in audio_files}
    speaker_references: Dict[str, List[Tuple[str, torch.Tensor]]] = {}

    for profile in profiles:
        user_id = profile["user_id"]
        metadata = profile_manager.load_metadata(user_id)
        active_records = profile_manager.get_active_reference_records(metadata)
        embeddings_dir = profile_manager._get_embeddings_directory(user_id)
        audio_sub_dir = profile_manager._get_audio_directory(user_id)

        # Check audio in profile's audio directory
        if audio_sub_dir.exists():
            for prof_wav in audio_sub_dir.glob("*.wav"):
                if prof_wav.name in file_to_speaker:
                    file_to_speaker[prof_wav.name] = user_id

        # Load reference embeddings and track associated audio files
        refs = []
        for record in active_records:
            emb_file = record.get("embedding_file")
            rec_id = record.get("recording_id", emb_file)
            audio_path_str = record.get("audio_path")
            
            if audio_path_str:
                src_name = Path(audio_path_str).name
                if src_name in file_to_speaker:
                    file_to_speaker[src_name] = user_id
                    file_to_ref_id[src_name] = emb_file

            if emb_file:
                emb_path = embeddings_dir / emb_file
                if emb_path.exists():
                    emb = torch.load(emb_path, map_location="cpu")
                    emb = profile_manager._validate_embedding(emb)
                    refs.append((emb_file, emb))

        speaker_references[user_id] = refs

    return audio_files, file_to_speaker, speaker_references, file_to_ref_id


def extract_embedding_for_config(
    audio_path: Path,
    config_name: str,
    preprocessor: SpeakerAudioPreprocessor,
    encoder: SpeakerEncoder,
) -> torch.Tensor:
    """
    Extract 192-dimensional ECAPA-TDNN speaker embedding for the specified configuration.
    """
    # Baseline preprocessing: Load -> Mono -> Resample 16kHz -> Silence Trim
    signal = preprocessor.process(audio_path, trim_silence=True, normalize=False)

    if config_name == "Baseline":
        pass
    elif config_name == "VAD":
        signal = apply_energy_vad(signal)
    elif config_name == "Normalization":
        signal = safe_peak_normalize(signal)
    elif config_name == "VAD + Normalization":
        signal = apply_energy_vad(signal)
        signal = safe_peak_normalize(signal)
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

    signal = signal.to(encoder.device)
    with torch.no_grad():
        embedding = encoder.model.encode_batch(signal).squeeze()

    embedding = embedding.detach().cpu().float()
    if embedding.ndim != 1 or embedding.shape[0] != 192:
        raise ValueError(f"Expected 192-dim embedding, got shape {embedding.shape}")

    return embedding


import contextlib
import io


def run_experiment():
    data_dir = PROJECT_ROOT / "data"
    audio_dir = data_dir / "audio"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = results_dir / "preprocessing_comparison.csv"

    # Suppress verbose loading messages during initialization
    with contextlib.redirect_stdout(io.StringIO()):
        profile_manager = VoiceProfileManager(str(data_dir / "voice_profiles"))
        preprocessor = SpeakerAudioPreprocessor()
        encoder = SpeakerEncoder()
        identifier = SpeakerIdentifier(profile_manager=profile_manager, similarity_threshold=0.70)

    audio_files, file_to_speaker, speaker_references, file_to_ref_id = discover_audio_and_profiles(
        audio_dir, profile_manager
    )

    known_speakers = {s for s in file_to_speaker.values() if s is not None}
    known_files = [f for f in audio_files if file_to_speaker[f.name] is not None]
    unknown_files = [f for f in audio_files if file_to_speaker[f.name] is None]

    print("=" * 60)
    print("VOICE PREPROCESSING QUICK VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Audio files discovered: {len(audio_files)}")
    print(f"Speaker profiles discovered: {len(speaker_references)}")
    print(f"Known speaker files: {len(known_files)}")
    print(f"Unknown/unassigned files: {len(unknown_files)}")
    print("=" * 60)

    configurations = [
        "Baseline",
        "VAD",
        "Normalization",
        "VAD + Normalization",
    ]

    config_results = []

    for config in configurations:
        print(f"Evaluating Configuration: {config} ...")
        
        all_same_similarities: List[float] = []
        all_diff_similarities: List[float] = []
        
        correct_known = 0
        incorrect_known = 0
        rejected_known = 0
        
        correct_unknown = 0
        incorrect_unknown = 0

        for audio_file in audio_files:
            true_speaker = file_to_speaker[audio_file.name]
            own_ref_id = file_to_ref_id.get(audio_file.name)

            with contextlib.redirect_stdout(io.StringIO()):
                emb = extract_embedding_for_config(
                    audio_file, config, preprocessor, encoder
                )
                id_result = identifier.identify(emb)

            predicted_user = id_result.get("user_id") if id_result.get("identified") else None

            if true_speaker is not None:
                if id_result.get("identified"):
                    if predicted_user == true_speaker:
                        correct_known += 1
                    else:
                        incorrect_known += 1
                else:
                    rejected_known += 1

                # Same-speaker similarities (excluding self-reference if applicable to avoid leakage)
                user_refs = speaker_references.get(true_speaker, [])
                for ref_id, ref_emb in user_refs:
                    if own_ref_id and ref_id == own_ref_id:
                        continue  # Skip self-reference comparison
                    sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                    all_same_similarities.append(sim)

                # Different-speaker similarities
                for other_user, other_refs in speaker_references.items():
                    if other_user != true_speaker:
                        for _, ref_emb in other_refs:
                            sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                            all_diff_similarities.append(sim)
            else:
                # Unknown recording
                if not id_result.get("identified"):
                    correct_unknown += 1
                else:
                    incorrect_unknown += 1

                # Different-speaker similarities against all known profiles
                for _, other_refs in speaker_references.items():
                    for _, ref_emb in other_refs:
                        sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                        all_diff_similarities.append(sim)

        avg_same = sum(all_same_similarities) / len(all_same_similarities) if all_same_similarities else 0.0
        avg_diff = sum(all_diff_similarities) / len(all_diff_similarities) if all_diff_similarities else 0.0
        margin = avg_same - avg_diff

        total_tested = len(audio_files)
        total_correct = correct_known + correct_unknown
        overall_accuracy = (total_correct / total_tested * 100.0) if total_tested > 0 else 0.0
        known_accuracy = (correct_known / len(known_files) * 100.0) if len(known_files) > 0 else 0.0

        config_results.append({
            "configuration": config,
            "audio_files_tested": total_tested,
            "known_samples": len(known_files),
            "unknown_samples": len(unknown_files),
            "average_same_speaker_similarity": avg_same,
            "average_different_speaker_similarity": avg_diff,
            "separation_margin": margin,
            "identification_accuracy": overall_accuracy,
            "known_accuracy": known_accuracy,
            "correct_known": correct_known,
            "incorrect_known": incorrect_known,
            "rejected_known": rejected_known,
            "correct_unknown_rejected": correct_unknown,
            "incorrect_unknown_accepted": incorrect_unknown,
        })

    # Find baseline margin and best configuration
    baseline_res = next(r for r in config_results if r["configuration"] == "Baseline")
    baseline_margin = baseline_res["separation_margin"]

    best_config_res = max(config_results, key=lambda x: (x["separation_margin"], x["identification_accuracy"]))
    best_config_name = best_config_res["configuration"]
    margin_improvement = (
        ((best_config_res["separation_margin"] - baseline_margin) / baseline_margin * 100.0)
        if baseline_margin > 0
        else 0.0
    )

    # Print Summary Table
    print("\n" + "=" * 60)
    print("VOICE PREPROCESSING QUICK VALIDATION RESULTS")
    print("=" * 60)
    print(f"Audio files tested: {len(audio_files)}")
    print(f"Known speakers: {len(known_speakers)}")
    print(f"Unknown/unassigned: {len(unknown_files)}")
    print()
    print(f"{'Configuration':<22} {'Same':<10} {'Different':<12} {'Margin':<10}")
    print("-" * 60)
    for r in config_results:
        print(
            f"{r['configuration']:<22} "
            f"{r['average_same_speaker_similarity']:<10.4f} "
            f"{r['average_different_speaker_similarity']:<12.4f} "
            f"{r['separation_margin']:<10.4f}"
        )

    print("\nIdentification Accuracy (Threshold = 0.70)")
    print("-" * 60)
    for r in config_results:
        print(
            f"{r['configuration']:<22} "
            f"Overall: {r['identification_accuracy']:>5.1f}%  "
            f"(Known: {r['correct_known']}/{r['known_samples']} correct, "
            f"Unknown: {r['correct_unknown_rejected']}/{r['unknown_samples']} rejected)"
        )

    print("-" * 60)
    print(f"Best Configuration: {best_config_name}")
    print(f"Improvement over Baseline: {margin_improvement:+.2f}%")
    print("=" * 60)

    # Save to CSV
    fieldnames = [
        "configuration",
        "audio_files_tested",
        "known_samples",
        "unknown_samples",
        "average_same_speaker_similarity",
        "average_different_speaker_similarity",
        "separation_margin",
        "identification_accuracy",
        "known_accuracy",
        "correct_known",
        "incorrect_known",
        "rejected_known",
        "correct_unknown_rejected",
        "incorrect_unknown_accepted",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in config_results:
            writer.writerow(r)

    print(f"\n[SUCCESS] Experiment results saved to:\n  {csv_output_path.resolve()}\n")


if __name__ == "__main__":
    run_experiment()
