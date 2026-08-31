"""
Speaker Diarization Feasibility Experiment

Isolated experiment to test whether recordings containing multiple speakers or turns
can be segmented and clustered into distinct anonymous speaker turns, and then
matched against existing voice profiles using the ECAPA-TDNN encoder and SpeakerIdentifier.

Production Safety:
- Zero modifications to Voice_Recognition/, Voice_Cloning/, Voice_Input/, main.py, data/
- Zero audio duplication (all windowing and segment embeddings extracted in-memory)
- Results saved to Voice_Recognition_Validation/results/diarization_results/
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
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F

from Voice_Recognition.speaker_encoder import SpeakerEncoder
from Voice_Recognition.speaker_preprocessing import SpeakerAudioPreprocessor
from Voice_Recognition.voice_profile_manager import VoiceProfileManager
from Voice_Recognition.speaker_identifier import SpeakerIdentifier


def detect_speech_regions(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    frame_len: int = 512,
    hop_len: int = 256,
    threshold_db: float = -30.0,
    min_speech_sec: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Detect coarse speech intervals in audio based on RMS energy.
    Returns list of (start_sec, end_sec) tuples.
    """
    samples = waveform.shape[1]
    if samples < frame_len:
        return [(0.0, float(samples / sample_rate))]

    frames = waveform.squeeze(0).unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
    max_rms = rms.max()
    if max_rms < 1e-6:
        return [(0.0, float(samples / sample_rate))]

    db = 20 * torch.log10(rms / max_rms)
    active_frames = (db >= threshold_db).cpu().numpy()

    # Find contiguous active regions
    regions: List[Tuple[float, float]] = []
    in_region = False
    start_frame = 0

    for i, active in enumerate(active_frames):
        if active and not in_region:
            in_region = True
            start_frame = i
        elif not active and in_region:
            in_region = False
            end_frame = i
            start_t = (start_frame * hop_len) / sample_rate
            end_t = min(samples, (end_frame * hop_len) + frame_len) / sample_rate
            if end_t - start_t >= min_speech_sec:
                regions.append((start_t, end_t))

    if in_region:
        start_t = (start_frame * hop_len) / sample_rate
        end_t = samples / sample_rate
        if end_t - start_t >= min_speech_sec:
            regions.append((start_t, end_t))

    if not regions:
        regions = [(0.0, float(samples / sample_rate))]

    return regions


def extract_window_embeddings(
    waveform: torch.Tensor,
    speech_regions: List[Tuple[float, float]],
    encoder: SpeakerEncoder,
    sample_rate: int = 16000,
    window_sec: float = 1.5,
    step_sec: float = 0.75,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Extract sliding window segments and corresponding 192-dim ECAPA embeddings.
    """
    window_samples = int(window_sec * sample_rate)
    step_samples = int(step_sec * sample_rate)
    total_samples = waveform.shape[1]

    segments: List[Tuple[float, float]] = []
    embeddings: List[np.ndarray] = []

    for reg_start, reg_end in speech_regions:
        start_sample = int(reg_start * sample_rate)
        end_sample = min(total_samples, int(reg_end * sample_rate))
        region_len = end_sample - start_sample

        if region_len < window_samples:
            # Pad short region to min window length for robust ECAPA extraction
            chunk = waveform[:, start_sample:end_sample]
            pad_needed = window_samples - chunk.shape[1]
            padded_chunk = F.pad(chunk, (0, pad_needed), mode="constant", value=0.0)
            
            with torch.no_grad():
                emb = encoder.model.encode_batch(padded_chunk.to(encoder.device)).squeeze()
            emb_norm = F.normalize(emb, p=2, dim=0).detach().cpu().numpy()
            
            segments.append((reg_start, reg_end))
            embeddings.append(emb_norm)
            continue

        for w_start in range(start_sample, end_sample - window_samples + 1, step_samples):
            w_end = w_start + window_samples
            chunk = waveform[:, w_start:w_end]

            with torch.no_grad():
                emb = encoder.model.encode_batch(chunk.to(encoder.device)).squeeze()
            emb_norm = F.normalize(emb, p=2, dim=0).detach().cpu().numpy()

            t_start = float(w_start / sample_rate)
            t_end = float(w_end / sample_rate)

            segments.append((t_start, t_end))
            embeddings.append(emb_norm)

    if not embeddings:
        # Fallback to whole file
        with torch.no_grad():
            emb = encoder.model.encode_batch(waveform.to(encoder.device)).squeeze()
        emb_norm = F.normalize(emb, p=2, dim=0).detach().cpu().numpy()
        segments.append((0.0, float(total_samples / sample_rate)))
        embeddings.append(emb_norm)

    return segments, np.vstack(embeddings)


def cluster_speaker_segments(
    embeddings: np.ndarray,
    distance_threshold: float = 0.48,
) -> np.ndarray:
    """
    Cluster segment embeddings into anonymous speaker IDs using Agglomerative Clustering (cosine metric).
    """
    n_samples = embeddings.shape[0]
    if n_samples == 1:
        return np.array([0])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return labels


def merge_contiguous_segments(
    segments: List[Tuple[float, float]],
    labels: np.ndarray,
) -> List[Dict]:
    """
    Merge sequential overlapping windows with the same speaker label into distinct speaker turns.
    """
    if not segments:
        return []

    merged = []
    current_label = labels[0]
    current_start = segments[0][0]
    current_end = segments[0][1]

    for (s_start, s_end), lbl in zip(segments[1:], labels[1:]):
        if lbl == current_label and s_start <= current_end + 0.5:
            current_end = max(current_end, s_end)
        else:
            merged.append({
                "speaker_label": f"SPEAKER_{current_label + 1:02d}",
                "cluster_id": int(current_label),
                "start_time": round(current_start, 2),
                "end_time": round(current_end, 2),
                "duration": round(current_end - current_start, 2),
            })
            current_label = lbl
            current_start = s_start
            current_end = s_end

    merged.append({
        "speaker_label": f"SPEAKER_{current_label + 1:02d}",
        "cluster_id": int(current_label),
        "start_time": round(current_start, 2),
        "end_time": round(current_end, 2),
        "duration": round(current_end - current_start, 2),
    })

    return merged


def run_diarization_experiment():
    data_dir = PROJECT_ROOT / "data"
    audio_dir = data_dir / "audio"
    results_dir = Path(__file__).resolve().parent / "results" / "diarization_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = results_dir / "diarization_summary.csv"

    with contextlib.redirect_stdout(io.StringIO()):
        profile_manager = VoiceProfileManager(str(data_dir / "voice_profiles"))
        preprocessor = SpeakerAudioPreprocessor()
        encoder = SpeakerEncoder()
        identifier = SpeakerIdentifier(profile_manager=profile_manager, similarity_threshold=0.70)

    audio_files = sorted(list(audio_dir.glob("*.wav")))

    print("=" * 65)
    print("SPEAKER DIARIZATION FEASIBILITY EXPERIMENT")
    print("=" * 65)
    print(f"Audio files discovered: {len(audio_files)}")
    print(f"Profiles available for matching: {len(profile_manager.list_profiles())}")
    print("Windowing parameters: window=1.5s, step=0.75s, cosine distance threshold=0.48")
    print("=" * 65)

    summary_rows = []

    for audio_file in audio_files:
        with contextlib.redirect_stdout(io.StringIO()):
            # Load and resample to 16kHz mono
            waveform, sr = preprocessor.load_audio(audio_file)
            waveform = preprocessor.convert_to_mono(waveform)
            waveform = preprocessor.resample_audio(waveform, sr)
            waveform = preprocessor.remove_silence(waveform)

        total_dur = float(waveform.shape[1] / 16000)

        # 1. Speech regions
        speech_regions = detect_speech_regions(waveform)

        # 2. Sliding window embeddings
        segments, embeddings = extract_window_embeddings(
            waveform, speech_regions, encoder, window_sec=1.5, step_sec=0.75
        )

        # 3. Agglomerative Clustering
        labels = cluster_speaker_segments(embeddings, distance_threshold=0.32)
        distinct_clusters = sorted(list(set(labels)))
        num_speakers = len(distinct_clusters)

        # 4. Merge contiguous turns
        turns = merge_contiguous_segments(segments, labels)

        # 5. Cluster-level speaker identification
        cluster_identification = {}
        for c_id in distinct_clusters:
            c_indices = np.where(labels == c_id)[0]
            c_embs = embeddings[c_indices]
            mean_emb = np.mean(c_embs, axis=0)
            mean_emb_tensor = torch.from_numpy(mean_emb).float()
            mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=0)

            with contextlib.redirect_stdout(io.StringIO()):
                id_result = identifier.identify(mean_emb_tensor)

            cluster_identification[c_id] = {
                "identified": id_result.get("identified", False),
                "user_id": id_result.get("user_id", "Unknown"),
                "confidence": id_result.get("confidence", 0.0),
            }

        # Build turn descriptions
        turn_strings = []
        for t in turns:
            c_id = t["cluster_id"]
            id_info = cluster_identification[c_id]
            matched_user = id_info["user_id"] if id_info["identified"] else "Unknown"
            conf = id_info["confidence"]
            turn_str = f"{t['start_time']:.1f}s-{t['end_time']:.1f}s: {t['speaker_label']} ({matched_user}, {conf:.2f})"
            turn_strings.append(turn_str)

        timeline_summary = " | ".join(turn_strings)

        print(f"\nRecording: {audio_file.name} ({total_dur:.2f}s)")
        print(f"  Estimated Speakers: {num_speakers}")
        print(f"  Segment Turns ({len(turns)}):")
        for ts in turn_strings:
            print(f"    - {ts}")

        for t in turns:
            c_id = t["cluster_id"]
            id_info = cluster_identification[c_id]
            summary_rows.append({
                "audio_file": audio_file.name,
                "total_duration_sec": f"{total_dur:.2f}",
                "estimated_speaker_count": num_speakers,
                "segment_index": len(summary_rows) + 1,
                "speaker_label": t["speaker_label"],
                "start_time": t["start_time"],
                "end_time": t["end_time"],
                "segment_duration": t["duration"],
                "identified_user": id_info["user_id"] if id_info["identified"] else "Unknown",
                "confidence": f"{id_info['confidence']:.4f}",
                "is_identified": id_info["identified"],
            })

    # Save summary CSV
    fieldnames = [
        "audio_file",
        "total_duration_sec",
        "estimated_speaker_count",
        "segment_index",
        "speaker_label",
        "start_time",
        "end_time",
        "segment_duration",
        "identified_user",
        "confidence",
        "is_identified",
    ]

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\n" + "=" * 65)
    print("DIARIZATION EXPERIMENT COMPLETED")
    print(f"Total audio files diarized: {len(audio_files)}")
    print(f"Total speaker turns segmented: {len(summary_rows)}")
    print(f"[SUCCESS] Diarization summary saved to:\n  {summary_csv_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    run_diarization_experiment()
