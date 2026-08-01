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

            "maximum_reference_embeddings": (
                self.MAX_REFERENCE_EMBEDDINGS
            ),

            "enrollment_complete": False,

            "created_at": current_time,

            "updated_at": current_time,

            "status": "active"

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
    # GET NUMBER OF REFERENCES
    # =========================================================

    def get_reference_count(
        self,
        user_id: str
    ) -> int:

        metadata = (
            self.load_metadata(
                user_id
            )
        )

        return int(
            metadata[
                "reference_embedding_count"
            ]
        )

    # =========================================================
    # CHECK ENROLLMENT STATUS
    # =========================================================

    def is_enrollment_complete(
        self,
        user_id: str
    ) -> bool:

        return (
            self.get_reference_count(
                user_id
            )
            >= self.MAX_REFERENCE_EMBEDDINGS
        )

    # =========================================================
    # ADD REFERENCE EMBEDDING
    # =========================================================

    def add_reference_embedding(
        self,
        user_id: str,
        embedding: torch.Tensor
    ) -> int:
        """
        Add one reference embedding.

        The embedding is saved in the next available slot.

        Returns:
            Reference number assigned to the embedding.

        Example:

            First embedding:
                reference_1.pt

            Second embedding:
                reference_2.pt

            ...

            Fifth embedding:
                reference_5.pt
        """

        self._validate_user_id(
            user_id
        )

        if not self.profile_exists(
            user_id
        ):

            raise FileNotFoundError(
                f"Profile does not exist: {user_id}"
            )

        current_count = (
            self.get_reference_count(
                user_id
            )
        )

        if current_count >= (
            self.MAX_REFERENCE_EMBEDDINGS
        ):

            raise RuntimeError(
                "Voice profile already contains "
                f"{self.MAX_REFERENCE_EMBEDDINGS} "
                "reference embeddings."
            )

        validated_embedding = (
            self._validate_embedding(
                embedding
            )
        )

        next_reference_number = (
            current_count + 1
        )

        embedding_path = (
            self._get_embedding_path(
                user_id,
                next_reference_number
            )
        )

        torch.save(
            validated_embedding,
            embedding_path
        )

        metadata = (
            self.load_metadata(
                user_id
            )
        )

        metadata[
            "reference_embedding_count"
        ] = next_reference_number

        metadata[
            "enrollment_complete"
        ] = (
            next_reference_number
            >= self.MAX_REFERENCE_EMBEDDINGS
        )

        metadata[
            "updated_at"
        ] = self._get_timestamp()

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

        return next_reference_number

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

            return json.load(
                file
            )

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

        if not (
            1
            <= reference_number
            <= self.MAX_REFERENCE_EMBEDDINGS
        ):

            raise ValueError(
                "reference_number must be between "
                "1 and 5."
            )

        embedding_path = (
            self._get_embedding_path(
                user_id,
                reference_number
            )
        )

        if not embedding_path.exists():

            raise FileNotFoundError(
                f"Reference embedding not found: "
                f"{embedding_path}"
            )

        embedding = torch.load(
            embedding_path,
            map_location="cpu"
        )

        return self._validate_embedding(
            embedding
        )

    # =========================================================
    # LOAD ALL EXISTING EMBEDDINGS
    # =========================================================

    def load_all_embeddings(
        self,
        user_id: str
    ) -> List[torch.Tensor]:
        """
        Load only the embeddings that currently exist.

        If a profile has 2 references:
            returns 2 embeddings

        If a profile has 5 references:
            returns 5 embeddings
        """

        count = (
            self.get_reference_count(
                user_id
            )
        )

        embeddings = []

        for reference_number in range(
            1,
            count + 1
        ):

            embedding = (
                self.load_embedding(
                    user_id,
                    reference_number
                )
            )

            embeddings.append(
                embedding
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