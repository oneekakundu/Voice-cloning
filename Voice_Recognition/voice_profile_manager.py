"""
Voice Profile Manager

Responsibilities:
    - Create a new voice profile
    - Add reference embeddings incrementally
    - Store a maximum of 5 reference embeddings
    - Load reference embeddings
    - Manage profile metadata
    - Delete profiles

This module does NOT:
    - Record audio
    - Preprocess audio
    - Generate embeddings
    - Perform speaker identification
    - Calculate similarity scores
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import json
import torch


class VoiceProfileManager:

    # ECAPA-TDNN embedding dimension
    EXPECTED_EMBEDDING_DIMENSION = 192

    # Maximum number of reference embeddings per profile
    MAX_REFERENCE_EMBEDDINGS = 5

    def __init__(
        self,
        profiles_directory: str = "data/voice_profiles"
    ):
        """
        Initialize Voice Profile Manager.
        """

        self.profiles_directory = Path(
            profiles_directory
        )

        self.profiles_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # PATH HELPERS
    # =========================================================

    def _get_profile_directory(
        self,
        user_id: str
    ) -> Path:

        return (
            self.profiles_directory
            / user_id
        )

    def _get_embeddings_directory(
        self,
        user_id: str
    ) -> Path:

        return (
            self._get_profile_directory(user_id)
            / "embeddings"
        )

    def _get_metadata_path(
        self,
        user_id: str
    ) -> Path:

        return (
            self._get_profile_directory(user_id)
            / "profile.json"
        )

    def _get_audio_directory(
        self,
        user_id: str
    ) -> Path:

        return (
            self._get_profile_directory(user_id)
            / "audio"
        )

    def _get_embedding_path(
        self,
        user_id: str,
        reference_number: int
    ) -> Path:

        return (
            self._get_embeddings_directory(user_id)
            / f"reference_{reference_number}.pt"
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_user_id(
        self,
        user_id: str
    ) -> None:

        if not isinstance(
            user_id,
            str
        ):

            raise TypeError(
                "user_id must be a string."
            )

        if not user_id.strip():

            raise ValueError(
                "user_id cannot be empty."
            )

    def _validate_embedding(
        self,
        embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Validate and normalize an embedding.

        The returned tensor is:
            - detached
            - moved to CPU
            - float32
            - shape [192]
        """

        if not isinstance(
            embedding,
            torch.Tensor
        ):

            raise TypeError(
                "Embedding must be a torch.Tensor."
            )

        # Remove dimensions such as [1, 192]
        embedding = embedding.squeeze()

        if embedding.ndim != 1:

            raise ValueError(
                "Embedding must have shape [192]."
            )

        if embedding.shape[0] != (
            self.EXPECTED_EMBEDDING_DIMENSION
        ):

            raise ValueError(
                "Invalid embedding dimension. "
                f"Expected "
                f"{self.EXPECTED_EMBEDDING_DIMENSION}, "
                f"received "
                f"{embedding.shape[0]}."
            )

        return (
            embedding
            .detach()
            .cpu()
            .float()
        )

    def _get_timestamp(self) -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    # =========================================================
    # PROFILE EXISTENCE
    # =========================================================

    def profile_exists(
        self,
        user_id: str
    ) -> bool:

        self._validate_user_id(
            user_id
        )

        profile_directory = (
            self._get_profile_directory(
                user_id
            )
        )

        metadata_path = (
            self._get_metadata_path(
                user_id
            )
        )

        embeddings_directory = (
            self._get_embeddings_directory(
                user_id
            )
        )

        return (
            profile_directory.exists()
            and metadata_path.exists()
            and embeddings_directory.exists()
        )

    # =========================================================
    # CREATE EMPTY PROFILE
    # =========================================================

    def create_profile(
        self,
        user_id: str,
        profile_name: str = "Primary User"
    ) -> bool:
        """
        Create an empty voice profile.

        The first reference embedding is added later using:
            add_reference_embedding()
        """

        self._validate_user_id(
            user_id
        )

        if self.profile_exists(
            user_id
        ):

            raise ValueError(
                f"Profile already exists: {user_id}"
            )

        profile_directory = (
            self._get_profile_directory(
                user_id
            )
        )

        embeddings_directory = (
            self._get_embeddings_directory(
                user_id
            )
        )

        profile_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        embeddings_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        current_time = (
            self._get_timestamp()
        )

        metadata = {

            "user_id": user_id,

            "profile_name": profile_name,

            "embedding_dimension": (
                self.EXPECTED_EMBEDDING_DIMENSION
            ),

            "reference_embedding_count": 0,

            "total_enrollment_records": 0,

            "maximum_reference_embeddings": (
                self.MAX_REFERENCE_EMBEDDINGS
            ),

            "enrollment_complete": False,

            "created_at": current_time,

            "updated_at": current_time,

            "status": "active",

            "enrollment_records": []

        }

        metadata_path = (
            self._get_metadata_path(
                user_id
            )
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        return True

    # =========================================================
    # GENERATE NEXT USER ID
    # =========================================================

    def generate_next_user_id(self) -> str:
        """
        Generate the next available user ID (e.g., user_001, user_002, user_003).

        Determines the next available ID by inspecting existing profile directories
        to prevent overwriting existing users.
        """
        existing_ids = set()
        if self.profiles_directory.exists():
            for path in self.profiles_directory.iterdir():
                if path.is_dir():
                    existing_ids.add(path.name)

        number = 1
        while True:
            user_id = f"user_{number:03d}"
            if user_id not in existing_ids:
                return user_id
            number += 1


    # =========================================================
    # ACTIVE REFERENCE SELECTION ALGORITHM
    # =========================================================

    def get_active_reference_records(
        self,
        metadata_or_user_id: dict | str
    ) -> List[dict]:
        """
        Select up to 5 enrollment records with the longest usable speech duration.
        
        Sorting Key:
            1. Usable speech duration (descending)
            2. Earlier enrollment order (ascending) as deterministic tie-breaker
            
        Legacy records with unknown duration (None) sort after positive durations,
        preserving their original enrollment order as fallback.
        """
        if isinstance(metadata_or_user_id, str):
            metadata = self.load_metadata(metadata_or_user_id)
        else:
            metadata = metadata_or_user_id

        records = metadata.get("enrollment_records", [])
        if not records:
            return []

        def sorting_key(record):
            dur = record.get("usable_speech_duration")
            # Primary: descending usable speech duration (-dur)
            # Fallback for None duration: 1.0 (so legacy records sort after positive durations)
            dur_val = -dur if dur is not None else 1.0
            # Tie-breaker: earlier enrollment_order (ascending)
            order_val = record.get("enrollment_order", 0)
            return (dur_val, order_val)

        sorted_records = sorted(records, key=sorting_key)
        return sorted_records[:self.MAX_REFERENCE_EMBEDDINGS]

    # =========================================================
    # GET ACTIVE REFERENCE AUDIO PATHS
    # =========================================================

    def get_active_reference_audio_paths(
        self,
        user_id: str
    ) -> List[Path]:
        """
        Retrieve the list of active Top-5 reference audio file paths for a user.

        Reuses get_active_reference_records(...) as the single source of truth for Top-5 selection.
        Prefers profile-managed audio files in data/voice_profiles/{user_id}/audio/.
        Falls back to metadata audio_path if valid and present on disk.
        Raises an exception if any active reference cannot be resolved to an existing audio file.
        """
        self._validate_user_id(user_id)

        if not self.profile_exists(user_id):
            raise FileNotFoundError(
                f"Voice profile does not exist for user_id: '{user_id}'"
            )

        metadata = self.load_metadata(user_id)
        active_records = self.get_active_reference_records(metadata)

        if not active_records:
            raise ValueError(
                f"Voice profile for user '{user_id}' exists but has no active reference records."
            )

        audio_dir = self._get_audio_directory(user_id)

        # Get all wav files in audio_dir sorted deterministically by name
        audio_files = []
        if audio_dir.exists() and audio_dir.is_dir():
            audio_files = sorted(
                [f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"],
                key=lambda f: f.name.lower()
            )

        resolved_paths: List[Path] = []
        seen_paths = set()

        for record in active_records:
            rec_id = record.get("recording_id", "unknown")
            meta_audio_path_str = record.get("audio_path")

            resolved_path: Optional[Path] = None
            attempted_profile_location: Optional[Path] = None

            # 1. Prefer profile-managed audio copy based on save_audio_recording behavior (source_path.name)
            if meta_audio_path_str:
                source_name = Path(meta_audio_path_str).name
                candidate_path = audio_dir / source_name
                attempted_profile_location = candidate_path
                if candidate_path.exists() and candidate_path.is_file():
                    resolved_path = candidate_path

            # 2. Check profile audio directory by recording_id filename
            if resolved_path is None and rec_id:
                candidate_by_id = audio_dir / f"{rec_id}.wav"
                if attempted_profile_location is None:
                    attempted_profile_location = candidate_by_id
                if candidate_by_id.exists() and candidate_by_id.is_file():
                    resolved_path = candidate_by_id

            # 3. Check fallback in profile audio directory by enrollment_order index if legacy profile
            if resolved_path is None and audio_files:
                order = record.get("enrollment_order", 0)
                if 1 <= order <= len(audio_files):
                    candidate_by_order = audio_files[order - 1]
                    if attempted_profile_location is None:
                        attempted_profile_location = candidate_by_order
                    if candidate_by_order.exists() and candidate_by_order.is_file():
                        resolved_path = candidate_by_order

            # 4. Fall back to metadata audio_path if it exists on disk
            if resolved_path is None and meta_audio_path_str:
                meta_path = Path(meta_audio_path_str)
                if meta_path.exists() and meta_path.is_file():
                    resolved_path = meta_path

            # 5. If any active reference record cannot be resolved, raise clear exception
            if resolved_path is None:
                raise FileNotFoundError(
                    f"Could not resolve audio file for active reference record.\n"
                    f"User ID: {user_id}\n"
                    f"Recording ID: {rec_id}\n"
                    f"Attempted profile audio location: {attempted_profile_location}\n"
                    f"Metadata audio_path: {meta_audio_path_str}"
                )

            resolved_abs = resolved_path.resolve()
            if resolved_abs in seen_paths:
                continue
            seen_paths.add(resolved_abs)
            resolved_paths.append(resolved_path)

        return resolved_paths

    # =========================================================
    # GET NUMBER OF REFERENCES
    # =========================================================

    def get_reference_count(
        self,
        user_id: str
    ) -> int:
        """
        Return the count of active reference embeddings (min(5, total_valid_recordings)).
        """
        metadata = self.load_metadata(user_id)
        active_records = self.get_active_reference_records(metadata)
        return len(active_records)

    def get_total_enrollment_count(
        self,
        user_id: str
    ) -> int:
        """
        Return total number of valid enrollment recordings in speaker history.
        """
        metadata = self.load_metadata(user_id)
        return len(metadata.get("enrollment_records", []))

    # =========================================================
    # CHECK ENROLLMENT STATUS
    # =========================================================

    def is_enrollment_complete(
        self,
        user_id: str
    ) -> bool:
        """
        Returns True if profile contains at least MAX_REFERENCE_EMBEDDINGS (5) recordings.
        """
        return self.get_reference_count(user_id) >= self.MAX_REFERENCE_EMBEDDINGS

    # =========================================================
    # ADD REFERENCE EMBEDDING
    # =========================================================

    def add_reference_embedding(
        self,
        user_id: str,
        embedding: torch.Tensor,
        usable_speech_duration: Optional[float] = None,
        audio_path: Optional[str | Path] = None,
        recording_id: Optional[str] = None
    ) -> int:
        """
        Add a valid enrollment recording and embedding to speaker history.
        
        Recalculates active top-five reference recordings by usable speech duration.
        """
        self._validate_user_id(user_id)

        if not self.profile_exists(user_id):
            raise FileNotFoundError(
                f"Profile does not exist: {user_id}"
            )

        validated_embedding = self._validate_embedding(embedding)
        metadata = self.load_metadata(user_id)
        records = metadata.get("enrollment_records", [])

        next_order = len(records) + 1
        rec_id = recording_id or f"recording_{next_order:04d}"
        embedding_filename = f"{rec_id}.pt"

        embeddings_directory = self._get_embeddings_directory(user_id)
        embeddings_directory.mkdir(parents=True, exist_ok=True)
        embedding_path = embeddings_directory / embedding_filename

        torch.save(validated_embedding, embedding_path)

        new_record = {
            "recording_id": rec_id,
            "audio_path": str(audio_path) if audio_path else None,
            "usable_speech_duration": float(usable_speech_duration) if usable_speech_duration is not None else None,
            "embedding_file": embedding_filename,
            "enrollment_order": next_order,
            "created_at": self._get_timestamp()
        }

        records.append(new_record)
        metadata["enrollment_records"] = records
        metadata["total_enrollment_records"] = len(records)

        active_records = self.get_active_reference_records(metadata)
        metadata["reference_embedding_count"] = len(active_records)
        metadata["enrollment_complete"] = (len(records) >= self.MAX_REFERENCE_EMBEDDINGS)
        metadata["updated_at"] = self._get_timestamp()

        metadata_path = self._get_metadata_path(user_id)
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        # Concise enrollment log
        active_durations = [
            f"{r['usable_speech_duration']:.2f}" if r.get('usable_speech_duration') is not None else "N/A"
            for r in active_records
        ]
        print(f"Speaker ID: {user_id}")
        if usable_speech_duration is not None:
            print(f"New usable speech duration: {usable_speech_duration:.2f} seconds")
        print(f"Total valid enrollment recordings: {len(records)}")
        print(f"Active reference recordings: {len(active_records)}")
        print(f"Active reference durations: [{', '.join(active_durations)}]")

        return next_order

    # =========================================================
    # SAVE AUDIO RECORDING TO PROFILE
    # =========================================================

    def save_audio_recording(
        self,
        user_id: str,
        audio_path: str | Path
    ) -> Path:
        """
        Copy the original recorded WAV file into the user's profile audio directory.

        Example destination:
            data/voice_profiles/user_001/audio/recording_20260727_092147.wav
        """

        self._validate_user_id(
            user_id
        )

        source_path = Path(
            audio_path
        )

        if not source_path.exists():

            raise FileNotFoundError(
                f"Source audio file not found: {source_path}"
            )

        audio_dir = (
            self._get_audio_directory(
                user_id
            )
        )

        audio_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_path = (
            audio_dir
            / source_path.name
        )

        import shutil

        if source_path.resolve() != destination_path.resolve():

            shutil.copy2(
                source_path,
                destination_path
            )

        if not destination_path.exists():

            raise RuntimeError(
                f"Failed to copy audio recording to: {destination_path}"
            )

        print(
            f"✓ Original audio recording saved to profile: {destination_path}"
        )

        return destination_path

    # =========================================================
    # LOAD METADATA
    # =========================================================

    def load_metadata(
        self,
        user_id: str
    ) -> dict:

        self._validate_user_id(
            user_id
        )

        metadata_path = (
            self._get_metadata_path(
                user_id
            )
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Profile metadata not found for "
                f"user: {user_id}"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )

        # Backward compatibility / legacy profile migration
        if "enrollment_records" not in metadata:
            records = []
            ref_count = metadata.get("reference_embedding_count", 0)
            embeddings_dir = self._get_embeddings_directory(user_id)

            for i in range(1, ref_count + 1):
                emb_path = embeddings_dir / f"reference_{i}.pt"
                if emb_path.exists():
                    records.append({
                        "recording_id": f"reference_{i}",
                        "audio_path": None,
                        "usable_speech_duration": None,
                        "embedding_file": f"reference_{i}.pt",
                        "enrollment_order": i,
                        "created_at": metadata.get("created_at")
                    })
            metadata["enrollment_records"] = records
            metadata["total_enrollment_records"] = len(records)
            metadata["reference_embedding_count"] = min(self.MAX_REFERENCE_EMBEDDINGS, len(records))

        return metadata

    # =========================================================
    # LOAD ONE EMBEDDING
    # =========================================================

    def load_embedding(
        self,
        user_id: str,
        reference_number: int
    ) -> torch.Tensor:

        self._validate_user_id(
            user_id
        )

        metadata = self.load_metadata(user_id)
        active_records = self.get_active_reference_records(metadata)

        if 1 <= reference_number <= len(active_records):
            record = active_records[reference_number - 1]
            embedding_path = self._get_embeddings_directory(user_id) / record["embedding_file"]
            if embedding_path.exists():
                embedding = torch.load(
                    embedding_path,
                    map_location="cpu"
                )
                return self._validate_embedding(embedding)

        fallback_path = self._get_embedding_path(user_id, reference_number)
        if fallback_path.exists():
            embedding = torch.load(
                fallback_path,
                map_location="cpu"
            )
            return self._validate_embedding(embedding)

        raise FileNotFoundError(
            f"Reference embedding not found: user={user_id}, ref={reference_number}"
        )

    # =========================================================
    # LOAD ALL EXISTING EMBEDDINGS
    # =========================================================

    def load_all_embeddings(
        self,
        user_id: str
    ) -> List[torch.Tensor]:
        """
        Load only the active top-five reference embeddings.
        """

        metadata = self.load_metadata(user_id)
        active_records = self.get_active_reference_records(metadata)
        embeddings_dir = self._get_embeddings_directory(user_id)

        embeddings = []

        for record in active_records:
            emb_file = record.get("embedding_file")
            if emb_file:
                emb_path = embeddings_dir / emb_file
                if emb_path.exists():
                    embedding = torch.load(
                        emb_path,
                        map_location="cpu"
                    )
                    embeddings.append(
                        self._validate_embedding(embedding)
                    )

        return embeddings

    # =========================================================
    # LOAD COMPLETE PROFILE
    # =========================================================

    def load_profile(
        self,
        user_id: str
    ) -> dict:

        return {

            "metadata": (
                self.load_metadata(
                    user_id
                )
            ),

            "embeddings": (
                self.load_all_embeddings(
                    user_id
                )
            )

        }

    # =========================================================
    # LIST PROFILES
    # =========================================================

    def list_profiles(self) -> list[dict]:

        profiles = []

        for profile_directory in (
            self.profiles_directory.iterdir()
        ):

            if not profile_directory.is_dir():

                continue

            metadata_path = (
                profile_directory
                / "profile.json"
            )

            if not metadata_path.exists():

                continue

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:

                profiles.append(
                    json.load(
                        file
                    )
                )

        return profiles

    # =========================================================
    # DELETE PROFILE
    # =========================================================

    def delete_profile(
        self,
        user_id: str
    ) -> bool:

        self._validate_user_id(
            user_id
        )

        profile_directory = (
            self._get_profile_directory(
                user_id
            )
        )

        if not profile_directory.exists():

            raise FileNotFoundError(
                f"Profile does not exist: "
                f"{user_id}"
            )

        for path in profile_directory.rglob(
            "*"
        ):

            if path.is_file():

                path.unlink()

        directories = sorted(

            [
                path
                for path in profile_directory.rglob(
                    "*"
                )
                if path.is_dir()
            ],

            key=lambda path: len(
                path.parts
            ),

            reverse=True

        )

        for directory in directories:

            directory.rmdir()

        profile_directory.rmdir()

        return True