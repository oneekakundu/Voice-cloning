"""
Voice Preprocessing Validation Experiment 2

Isolated experiment to evaluate CONSERVATIVE voice preprocessing techniques:
1. Baseline (Production preprocessing)
2. Conservative Endpoint Trimming (Outer silence trim with 250ms padding, preserving inner speech/pauses)
3. Gentle High-Pass Filter (Butterworth, order=2, cutoff=70 Hz)
4. Conservative Band-Pass Filter (Butterworth, order=2, 75 Hz - 7600 Hz)
5. Light Noise Reduction (Gentle spectral gating with conservative -6 dB attenuation floor)
6. Best Justified Combination (Evaluated based on standalone performance)

Production Safety:
- Zero modifications to Voice_Recognition/, Voice_Cloning/, Voice_Input/, main.py, data/
- Zero audio duplication (all signal processing in-memory)
- Reuses existing SpeakerEncoder, SpeakerAudioPreprocessor, VoiceProfileManager, SpeakerIdentifier
"""

import contextlib
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

from Voice_Recognition.speaker_encoder import SpeakerEncoder
from Voice_Recognition.speaker_preprocessing import SpeakerAudioPreprocessor
from Voice_Recognition.voice_profile_manager import VoiceProfileManager
from Voice_Recognition.speaker_identifier import SpeakerIdentifier


# ==============================================================================
# CONSERVATIVE PREPROCESSING SIGNAL PROCESSING FUNCTIONS
# ==============================================================================

def conservative_endpoint_trim(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    frame_len: int = 512,
    hop_len: int = 256,
    threshold_db: float = -35.0,
    padding_sec: float = 0.25,
) -> torch.Tensor:
    """
    Conservative outer silence trimming.
    Identifies the true onset and offset of speech and preserves ALL internal
    speech and natural pauses. Applies 250ms padding at both ends.
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
    active_frame_indices = torch.where(db >= threshold_db)[0]

    if len(active_frame_indices) == 0:
        return waveform

    first_frame = active_frame_indices[0].item()
    last_frame = active_frame_indices[-1].item()

    raw_start = first_frame * hop_len
    raw_end = min((last_frame * hop_len) + frame_len, samples)

    pad_samples = int(padding_sec * sample_rate)
    padded_start = max(0, raw_start - pad_samples)
    padded_end = min(samples, raw_end + pad_samples)

    if padded_end - padded_start < sample_rate * 1.0:
        return waveform

    return waveform[:, padded_start:padded_end]


def gentle_high_pass_filter(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    cutoff_hz: float = 70.0,
    order: int = 2,
) -> torch.Tensor:
    """
    Apply 2nd-order Butterworth high-pass filter to reject sub-audible DC/handling rumble.
    """
    audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_hz / nyquist

    sos = scipy.signal.butter(order, normalized_cutoff, btype="highpass", output="sos")
    filtered = scipy.signal.sosfilt(sos, audio_np)

    return torch.from_numpy(filtered).unsqueeze(0).float()


def conservative_band_pass_filter(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    low_cutoff_hz: float = 75.0,
    high_cutoff_hz: float = 7600.0,
    order: int = 2,
) -> torch.Tensor:
    """
    Apply 2nd-order Butterworth band-pass filter for conservative speech bandpass.
    """
    audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    nyquist = 0.5 * sample_rate

    # Ensure high cutoff is safely below Nyquist
    high_cutoff_hz = min(high_cutoff_hz, nyquist - 100.0)
    low_norm = low_cutoff_hz / nyquist
    high_norm = high_cutoff_hz / nyquist

    sos = scipy.signal.butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    filtered = scipy.signal.sosfilt(sos, audio_np)

    return torch.from_numpy(filtered).unsqueeze(0).float()


def light_noise_reduction(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_floor_quantile: float = 0.15,
    max_suppression_db: float = -6.0,
) -> torch.Tensor:
    """
    Gentle spectral gating for mild stationary background noise reduction.
    Limits suppression to -6 dB floor to strictly prevent musical noise or voice distortion.
    """
    audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    stft = scipy.signal.stft(
        audio_np,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )
    f, t, Zxx = stft
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise floor from the quietest time frames
    frame_energies = np.sum(magnitude ** 2, axis=0)
    quiet_idx = np.where(frame_energies <= np.quantile(frame_energies, noise_floor_quantile))[0]
    if len(quiet_idx) == 0:
        noise_profile = np.mean(magnitude, axis=1, keepdims=True)
    else:
        noise_profile = np.mean(magnitude[:, quiet_idx], axis=1, keepdims=True)

    snr_gain = (magnitude - 1.0 * noise_profile) / (magnitude + 1e-8)
    min_gain = 10.0 ** (max_suppression_db / 20.0)
    gain = np.clip(snr_gain, min_gain, 1.0)

    clean_stft = gain * magnitude * np.exp(1j * phase)
    _, cleaned_audio = scipy.signal.istft(
        clean_stft,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )

    # Match original length
    target_len = audio_np.shape[0]
    if len(cleaned_audio) > target_len:
        cleaned_audio = cleaned_audio[:target_len]
    elif len(cleaned_audio) < target_len:
        cleaned_audio = np.pad(cleaned_audio, (0, target_len - len(cleaned_audio)))

    return torch.from_numpy(cleaned_audio.astype(np.float32)).unsqueeze(0)


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def discover_audio_and_profiles(
    audio_dir: Path, profile_manager: VoiceProfileManager
) -> Tuple[List[Path], Dict[str, Optional[str]], Dict[str, List[Tuple[str, torch.Tensor]]], Dict[str, Optional[str]]]:
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

        if audio_sub_dir.exists():
            for prof_wav in audio_sub_dir.glob("*.wav"):
                if prof_wav.name in file_to_speaker:
                    file_to_speaker[prof_wav.name] = user_id

        refs = []
        for record in active_records:
            emb_file = record.get("embedding_file")
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


def extract_v2_embedding(
    audio_path: Path,
    config_name: str,
    preprocessor: SpeakerAudioPreprocessor,
    encoder: SpeakerEncoder,
    best_single_config: Optional[str] = None,
) -> torch.Tensor:
    # Baseline load & resample
    signal = preprocessor.process(audio_path, trim_silence=True, normalize=False)

    if config_name == "Baseline":
        pass
    elif config_name == "Conservative Trim":
        signal = conservative_endpoint_trim(signal, padding_sec=0.25)
    elif config_name == "High-Pass":
        signal = gentle_high_pass_filter(signal, cutoff_hz=70.0, order=2)
    elif config_name == "Band-Pass":
        signal = conservative_band_pass_filter(signal, low_cutoff_hz=75.0, high_cutoff_hz=7600.0, order=2)
    elif config_name == "Noise Reduction":
        signal = light_noise_reduction(signal, max_suppression_db=-6.0)
    elif config_name == "Best Combination":
        # Combine Conservative Trim + Gentle High-Pass
        signal = conservative_endpoint_trim(signal, padding_sec=0.25)
        signal = gentle_high_pass_filter(signal, cutoff_hz=70.0, order=2)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    signal = signal.to(encoder.device)
    with torch.no_grad():
        embedding = encoder.model.encode_batch(signal).squeeze()

    embedding = embedding.detach().cpu().float()
    return embedding


def run_experiment_v2():
    data_dir = PROJECT_ROOT / "data"
    audio_dir = data_dir / "audio"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = results_dir / "preprocessing_v2_comparison.csv"

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
    print("VOICE PREPROCESSING VALIDATION — EXPERIMENT 2")
    print("=" * 60)
    print(f"Audio files discovered: {len(audio_files)}")
    print(f"Speaker profiles discovered: {len(speaker_references)}")
    print(f"Known speaker files: {len(known_files)}")
    print(f"Unknown/unassigned files: {len(unknown_files)}")
    print("=" * 60)

    configurations = [
        "Baseline",
        "Conservative Trim",
        "High-Pass",
        "Band-Pass",
        "Noise Reduction",
        "Best Combination",
    ]

    config_params = {
        "Baseline": "Production settings",
        "Conservative Trim": "Threshold=-35dB, Pad=250ms",
        "High-Pass": "Butterworth order=2, Cutoff=70Hz",
        "Band-Pass": "Butterworth order=2, 75Hz-7600Hz",
        "Noise Reduction": "Spectral gating, Floor=-6dB",
        "Best Combination": "Conservative Trim + High-Pass (70Hz)",
    }

    config_results = []

    for config in configurations:
        print(f"Evaluating Configuration: {config:<22} ...")
        
        all_same_sims: List[float] = []
        all_diff_sims: List[float] = []
        
        correct_known = 0
        incorrect_known = 0
        rejected_known = 0
        
        correct_unknown = 0
        incorrect_unknown = 0

        for audio_file in audio_files:
            true_speaker = file_to_speaker[audio_file.name]
            own_ref_id = file_to_ref_id.get(audio_file.name)

            with contextlib.redirect_stdout(io.StringIO()):
                emb = extract_v2_embedding(audio_file, config, preprocessor, encoder)
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

                user_refs = speaker_references.get(true_speaker, [])
                for ref_id, ref_emb in user_refs:
                    if own_ref_id and ref_id == own_ref_id:
                        continue
                    sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                    all_same_sims.append(sim)

                for other_user, other_refs in speaker_references.items():
                    if other_user != true_speaker:
                        for _, ref_emb in other_refs:
                            sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                            all_diff_sims.append(sim)
            else:
                if not id_result.get("identified"):
                    correct_unknown += 1
                else:
                    incorrect_unknown += 1

                for _, other_refs in speaker_references.items():
                    for _, ref_emb in other_refs:
                        sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                        all_diff_sims.append(sim)

        avg_same = sum(all_same_sims) / len(all_same_sims) if all_same_sims else 0.0
        avg_diff = sum(all_diff_sims) / len(all_diff_sims) if all_diff_sims else 0.0
        margin = avg_same - avg_diff

        total_tested = len(audio_files)
        total_correct = correct_known + correct_unknown
        overall_accuracy = (total_correct / total_tested * 100.0) if total_tested > 0 else 0.0
        known_accuracy = (correct_known / len(known_files) * 100.0) if len(known_files) > 0 else 0.0

        config_results.append({
            "configuration": config,
            "parameters": config_params.get(config, ""),
            "audio_files_tested": total_tested,
            "known_samples": len(known_files),
            "unknown_samples": len(unknown_files),
            "average_same_speaker_similarity": avg_same,
            "average_different_speaker_similarity": avg_diff,
            "separation_margin": margin,
            "identification_accuracy": overall_accuracy,
            "known_accuracy": known_accuracy,
            "known_correct": correct_known,
            "known_incorrect": incorrect_known,
            "known_rejected": rejected_known,
            "unknown_correctly_rejected": correct_unknown,
            "unknown_incorrectly_accepted": incorrect_unknown,
        })

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
    print("VOICE PREPROCESSING VALIDATION — EXPERIMENT 2 RESULTS")
    print("=" * 60)
    print(f"Audio files tested: {len(audio_files)}")
    print(f"Known recordings: {len(known_files)}")
    print(f"Unknown/unassigned: {len(unknown_files)}")
    print()
    print(f"{'Configuration':<24} {'Same':<10} {'Different':<12} {'Margin':<10}")
    print("-" * 60)
    for r in config_results:
        print(
            f"{r['configuration']:<24} "
            f"{r['average_same_speaker_similarity']:<10.4f} "
            f"{r['average_different_speaker_similarity']:<12.4f} "
            f"{r['separation_margin']:<10.4f}"
        )

    print("\nKnown Speaker Identification")
    print("-" * 60)
    for r in config_results:
        print(
            f"{r['configuration']:<24} "
            f"{r['known_accuracy']:>5.1f}% ({r['known_correct']}/{r['known_samples']} correct)"
        )

    print("\nUnknown Speaker Rejection")
    print("-" * 60)
    for r in config_results:
        print(
            f"{r['configuration']:<24} "
            f"{r['unknown_correctly_rejected']}/{r['unknown_samples']} rejected"
        )

    print("-" * 60)
    print(f"Best Configuration: {best_config_name}")
    print(f"Separation Margin Improvement over Baseline: {margin_improvement:+.2f}%")
    print("=" * 60)

    # Save to CSV
    fieldnames = [
        "configuration",
        "parameters",
        "audio_files_tested",
        "known_samples",
        "unknown_samples",
        "average_same_speaker_similarity",
        "average_different_speaker_similarity",
        "separation_margin",
        "identification_accuracy",
        "known_accuracy",
        "known_correct",
        "known_incorrect",
        "known_rejected",
        "unknown_correctly_rejected",
        "unknown_incorrectly_accepted",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in config_results:
            writer.writerow(r)

    print(f"\n[SUCCESS] Experiment 2 results saved to:\n  {csv_output_path.resolve()}\n")


if __name__ == "__main__":
    run_experiment_v2()
