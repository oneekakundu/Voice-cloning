"""
Speaker Identifier

Responsibilities:
- Receive a new speaker embedding
- Load stored reference embeddings
- Compare the new embedding with profile references
- Count matching references
- Perform profile-level voting
- Return the best matching user
- Return UNKNOWN if no reliable match exists

This module does NOT:
- Record audio
- Preprocess audio
- Generate embeddings
"""

from typing import Optional

import torch
import torch.nn.functional as F

from Voice_Recognition.voice_profile_manager import (
VoiceProfileManager
)

class SpeakerIdentifier:

    def __init__(
        self,
        profile_manager: VoiceProfileManager,
        similarity_threshold: float = 0.70,
        minimum_matches: int = 1
    ):
        """
        Initialize Speaker Identifier.

        Args:
            profile_manager:
                VoiceProfileManager instance.

            similarity_threshold:
                Minimum cosine similarity required for
                one reference embedding to count as a match.

            minimum_matches:
                Optional minimum number of matching references.

                The profile-level rule is determined by
                _get_required_matches().

                This parameter is retained for compatibility.
        """

        self.profile_manager = (
            profile_manager
        )

        self.similarity_threshold = (
            similarity_threshold
        )

        self.minimum_matches = (
            minimum_matches
        )

    # =========================================================
    # EMBEDDING VALIDATION
    # =========================================================

    def _validate_embedding(
        self,
        embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Validate and standardize an embedding.

        Expected shape:

            [192]

        The embedding is:

            - Verified as a torch.Tensor
            - Squeezed
            - Detached from autograd
            - Converted to float32

        The original device is preserved.
        """

        if not isinstance(
            embedding,
            torch.Tensor
        ):

            raise TypeError(
                "Embedding must be a torch.Tensor."
            )

        embedding = (

            embedding
            .squeeze()
            .detach()
            .float()

        )

        if embedding.ndim != 1:

            raise ValueError(
                "Embedding must have shape [192]."
            )

        if embedding.shape[0] != 192:

            raise ValueError(
                "Embedding must have 192 dimensions."
            )

        return embedding

    # =========================================================
    # COSINE SIMILARITY
    # =========================================================

    def calculate_similarity(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Both embeddings are moved to the same device before
        cosine similarity is calculated.

        Typical pipeline:

            New embedding:
                CUDA:0

            Stored reference:
                CPU

        The stored reference is temporarily moved to the
        new embedding's device for comparison.
        """

        embedding_a = (

            self
            ._validate_embedding(

                embedding_a

            )

        )

        embedding_b = (

            self
            ._validate_embedding(

                embedding_b

            )

        )

        # Move the second embedding to the same device
        # as the first embedding.

        embedding_b = (

            embedding_b
            .to(

                embedding_a.device

            )

        )

        similarity = (

            F.cosine_similarity(

                embedding_a.unsqueeze(0),

                embedding_b.unsqueeze(0),

                dim=1

            )

        )

        return float(

            similarity.item()

        )

    # =========================================================
    # MATCHING RULE
    # =========================================================

    def _get_required_matches(
        self,
        reference_count: int
    ) -> int:
        """
        Determine how many stored references must match.

        Current architecture:

            1 reference → 1 match required

            2 references → 2 matches required

            3 references → 3 matches required

            4 references → 4 matches required

            5 references → 4 matches required

        This implements the proposed rule:

            With five reference embeddings,
            at least four must exceed the
            similarity threshold.
        """

        if reference_count <= 0:

            return 0

        if reference_count < 5:

            return reference_count

        return 4

    # =========================================================
    # COMPARE AGAINST ONE PROFILE
    # =========================================================

    def compare_with_profile(
        self,
        new_embedding: torch.Tensor,
        user_id: str
    ) -> dict:
        """
        Compare a new embedding against all references
        belonging to one user profile.

        Returns:

            {
                "user_id": "...",
                "match_count": 4,
                "reference_count": 5,
                "required_matches": 4,
                "match_ratio": 0.8,
                "average_similarity": 0.91,
                "maximum_similarity": 0.96,
                "is_match": True,
                "similarities": [...]
            }
        """

        new_embedding = (

            self
            ._validate_embedding(

                new_embedding

            )

        )

        reference_embeddings = (

            self
            .profile_manager
            .load_all_embeddings(

                user_id

            )

        )

        if not reference_embeddings:

            return {

                "user_id": user_id,

                "match_count": 0,

                "reference_count": 0,

                "required_matches": 0,

                "match_ratio": 0.0,

                "average_similarity": 0.0,

                "maximum_similarity": 0.0,

                "is_match": False,

                "similarities": []

            }

        similarities = []

        match_count = 0

        for reference_embedding in (

            reference_embeddings

        ):

            similarity = (

                self
                .calculate_similarity(

                    new_embedding,

                    reference_embedding

                )

            )

            similarities.append(

                similarity

            )

            if similarity >= (

                self
                .similarity_threshold

            ):

                match_count += 1

        reference_count = (

            len(

                reference_embeddings

            )

        )

        required_matches = (

            self
            ._get_required_matches(

                reference_count

            )

        )

        average_similarity = (

            sum(

                similarities

            )

            /

            len(

                similarities

            )

        )

        maximum_similarity = (

            max(

                similarities

            )

        )

        match_ratio = (

            match_count

            /

            reference_count

        )

        is_match = (

            match_count

            >=

            required_matches

        )

        return {

            "user_id": user_id,

            "match_count": match_count,

            "reference_count": reference_count,

            "required_matches": required_matches,

            "match_ratio": match_ratio,

            "average_similarity": (

                average_similarity

            ),

            "maximum_similarity": (

                maximum_similarity

            ),

            "is_match": is_match,

            "similarities": similarities

        }

    # =========================================================
    # IDENTIFY SPEAKER
    # =========================================================

    def identify(
        self,
        new_embedding: torch.Tensor
    ) -> dict:
        """
        Identify the speaker using all available profiles.

        The process is:

            1. Load all profiles
            2. Compare the new embedding against
               every reference in every profile
            3. Count matching references
            4. Rank profiles
            5. Return the best reliable match

        Returns:

            {
                "identified": True,
                "user_id": "...",
                "confidence": ...,
                "match_count": ...,
                "profile_results": [...]
            }
        """

        new_embedding = (

            self
            ._validate_embedding(

                new_embedding

            )

        )

        profiles = (

            self
            .profile_manager
            .list_profiles()

        )

        if not profiles:

            return {

                "identified": False,

                "user_id": None,

                "confidence": 0.0,

                "match_count": 0,

                "message": (

                    "No voice profiles available."

                ),

                "profile_results": []

            }

        profile_results = []

        for profile in (

            profiles

        ):

            user_id = (

                profile[

                    "user_id"

                ]

            )

            result = (

                self
                .compare_with_profile(

                    new_embedding,

                    user_id

                )

            )

            profile_results.append(

                result

            )

        # Rank profiles by:
        #
        # 1. Number of matching references
        # 2. Match ratio
        # 3. Average similarity
        # 4. Maximum similarity

        profile_results.sort(

            key=lambda result: (

                result[

                    "match_count"

                ],

                result[

                    "match_ratio"

                ],

                result[

                    "average_similarity"

                ],

                result[

                    "maximum_similarity"

                ]

            ),

            reverse=True

        )

        best_result = (

            profile_results[0]

        )

        identified = (

            best_result[

                "is_match"

            ]

        )

        if identified:

            return {

                "identified": True,

                "user_id": (

                    best_result[

                        "user_id"

                    ]

                ),

                "confidence": (

                    best_result[

                        "match_ratio"

                    ]

                ),

                "match_count": (

                    best_result[

                        "match_count"

                    ]

                ),

                "reference_count": (

                    best_result[

                        "reference_count"

                    ]

                ),

                "required_matches": (

                    best_result[

                        "required_matches"

                    ]

                ),

                "average_similarity": (

                    best_result[

                        "average_similarity"

                    ]

                ),

                "maximum_similarity": (

                    best_result[

                        "maximum_similarity"

                    ]

                ),

                "profile_results": (

                    profile_results

                )

            }

        return {

            "identified": False,

            "user_id": None,

            "confidence": 0.0,

            "match_count": (

                best_result[

                    "match_count"

                ]

            ),

            "reference_count": (

                best_result[

                    "reference_count"

                ]

            ),

            "required_matches": (

                best_result[

                    "required_matches"

                ]

            ),

            "average_similarity": (

                best_result[

                    "average_similarity"

                ]

            ),

            "maximum_similarity": (

                best_result[

                    "maximum_similarity"

                ]

            ),

            "message": (

                "Unknown speaker."

            ),

            "profile_results": (

                profile_results

            )
        }
