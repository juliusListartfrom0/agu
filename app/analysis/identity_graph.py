from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class IdentityTracklet:
    tracklet_id: str
    team_id: str
    start_frame: int
    end_frame: int
    embedding: tuple[float, ...]
    source_video_id: str = "video_001"
    spatial_samples: tuple[tuple[int, float, float, float], ...] = ()
    jersey_number: str | None = None
    jersey_confidence: float = 0.0
    gallery_person_id: str | None = None
    gallery_confidence: float = 0.0
    face_embedding: tuple[float, ...] = ()
    face_embedding_model: str | None = None
    face_quality: float = 0.0


@dataclass(frozen=True)
class CanonicalPlayerIdentity:
    player_id: str
    team_id: str
    tracklet_ids: tuple[str, ...]
    jersey_number: str | None
    confidence: float
    evidence: tuple[str, ...]


class IdentityGraph:
    """Conservative raw-video tracklet stitching without a reference roster."""

    def __init__(
        self,
        *,
        embedding_threshold: float = 0.88,
        trusted_jersey_confidence: float = 0.85,
        overlap_tolerance_frames: int = 2,
        face_match_threshold: float = 0.45,
        face_conflict_threshold: float = 0.30,
        enrolled_minimum_face_quality: float = 0.65,
    ) -> None:
        self.embedding_threshold = embedding_threshold
        self.trusted_jersey_confidence = trusted_jersey_confidence
        self.overlap_tolerance_frames = overlap_tolerance_frames
        self.face_match_threshold = face_match_threshold
        self.face_conflict_threshold = face_conflict_threshold
        self.enrolled_minimum_face_quality = enrolled_minimum_face_quality

    def resolve(self, tracklets: list[IdentityTracklet]) -> list[CanonicalPlayerIdentity]:
        parents = list(range(len(tracklets)))
        edge_scores: dict[tuple[int, int], float] = {}

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        candidates: list[tuple[float, int, int]] = []
        for left in range(len(tracklets)):
            for right in range(left + 1, len(tracklets)):
                score = self._merge_score(tracklets[left], tracklets[right])
                if score is not None:
                    candidates.append((score, left, right))
        for score, left, right in sorted(candidates, reverse=True):
            left_members = [index for index in range(len(tracklets)) if find(index) == find(left)]
            right_members = [index for index in range(len(tracklets)) if find(index) == find(right)]
            if any(self._hard_conflict(tracklets[a], tracklets[b]) for a in left_members for b in right_members):
                continue
            if not self._anchored_group_compatible(
                [tracklets[index] for index in left_members],
                [tracklets[index] for index in right_members],
            ):
                continue
            # Complete-link compatibility prevents a chain of individually
            # plausible edges from collapsing visually different players.
            if any(self._merge_score(tracklets[a], tracklets[b]) is None for a in left_members for b in right_members):
                continue
            union(left, right)
            edge_scores[(min(left, right), max(left, right))] = score

        groups: dict[int, list[int]] = {}
        for index in range(len(tracklets)):
            groups.setdefault(find(index), []).append(index)
        identities = [
            self._identity([tracklets[index] for index in indices], edge_scores) for indices in groups.values()
        ]
        identities = _ensure_unique_player_ids(identities)
        return sorted(identities, key=lambda item: (item.team_id, item.jersey_number or "", item.player_id))

    def _anchored_group_compatible(
        self,
        left: list[IdentityTracklet],
        right: list[IdentityTracklet],
    ) -> bool:
        """Do not propagate an enrolled identity through body appearance alone."""

        anchors = [item for item in [*left, *right] if item.gallery_person_id]
        if not anchors:
            return True
        gallery_ids = {item.gallery_person_id for item in anchors}
        if len(gallery_ids) != 1:
            return False
        for member in [*left, *right]:
            if member.gallery_person_id:
                continue
            if any(self._has_direct_anchor_evidence(member, anchor) for anchor in anchors):
                continue
            return False
        return True

    def _has_direct_anchor_evidence(
        self,
        member: IdentityTracklet,
        anchor: IdentityTracklet,
    ) -> bool:
        if min(member.face_quality, anchor.face_quality) < self.enrolled_minimum_face_quality:
            return False
        face_cosine = self._face_cosine(member, anchor)
        if face_cosine is not None and face_cosine >= self.face_match_threshold:
            return True
        return bool(
            _trusted_jersey(member, self.trusted_jersey_confidence)
            and _trusted_jersey(anchor, self.trusted_jersey_confidence)
            and member.jersey_number == anchor.jersey_number
        )

    def _merge_score(self, left: IdentityTracklet, right: IdentityTracklet) -> float | None:
        if self._hard_conflict(left, right):
            return None
        if left.gallery_person_id and left.gallery_person_id == right.gallery_person_id:
            return 3.0 + min(left.gallery_confidence, right.gallery_confidence)
        face_cosine = self._face_cosine(left, right)
        if face_cosine is not None and face_cosine >= self.face_match_threshold:
            return 1.0 + face_cosine
        jersey_match = _trusted_jersey(left, self.trusted_jersey_confidence) and _trusted_jersey(
            right, self.trusted_jersey_confidence
        )
        cosine = _cosine(left.embedding, right.embedding)
        if jersey_match and left.jersey_number == right.jersey_number:
            return 2.0 + min(left.jersey_confidence, right.jersey_confidence)
        if cosine >= self.embedding_threshold:
            return cosine
        return None

    def _hard_conflict(self, left: IdentityTracklet, right: IdentityTracklet) -> bool:
        if left.team_id != right.team_id:
            return True
        if left.gallery_person_id and right.gallery_person_id and left.gallery_person_id != right.gallery_person_id:
            return True
        face_cosine = self._face_cosine(left, right)
        if face_cosine is not None and face_cosine < self.face_conflict_threshold:
            return True
        overlap = min(left.end_frame, right.end_frame) - max(left.start_frame, right.start_frame)
        if left.source_video_id == right.source_video_id and overlap > self.overlap_tolerance_frames:
            if not left.spatial_samples or not right.spatial_samples:
                return True
            aligned_distances = _aligned_spatial_distances(
                left.spatial_samples,
                right.spatial_samples,
                frame_tolerance=self.overlap_tolerance_frames,
            )
            if not aligned_distances:
                return True
            if float(sorted(aligned_distances)[len(aligned_distances) // 2]) > 0.45:
                return True
        if (
            _trusted_jersey(left, self.trusted_jersey_confidence)
            and _trusted_jersey(right, self.trusted_jersey_confidence)
            and left.jersey_number != right.jersey_number
        ):
            return True
        return False

    @staticmethod
    def _face_cosine(left: IdentityTracklet, right: IdentityTracklet) -> float | None:
        if (
            not left.face_embedding
            or not right.face_embedding
            or not left.face_embedding_model
            or left.face_embedding_model != right.face_embedding_model
            or min(left.face_quality, right.face_quality) < 0.50
        ):
            return None
        return _cosine(left.face_embedding, right.face_embedding)

    def _identity(
        self,
        members: list[IdentityTracklet],
        edge_scores: dict[tuple[int, int], float],
    ) -> CanonicalPlayerIdentity:
        trusted_numbers = [
            member.jersey_number
            for member in members
            if _trusted_jersey(member, self.trusted_jersey_confidence) and member.jersey_number is not None
        ]
        jersey_number = max(set(trusted_numbers), key=trusted_numbers.count) if trusted_numbers else None
        digest = hashlib.sha256("|".join(sorted(member.tracklet_id for member in members)).encode("utf-8")).hexdigest()[
            :10
        ]
        gallery_ids = [member.gallery_person_id for member in members if member.gallery_person_id]
        gallery_person_id = gallery_ids[0] if gallery_ids and len(set(gallery_ids)) == 1 else None
        if gallery_person_id:
            player_id = gallery_person_id
        elif jersey_number:
            player_id = f"{members[0].team_id}-jersey-{jersey_number}"
        else:
            player_id = f"{members[0].team_id}-player-{digest}"
        similarities = [
            _cosine(left.embedding, right.embedding)
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        ]
        confidence = min(similarities) if similarities else max(0.5, members[0].jersey_confidence)
        evidence = [f"tracklets={len(members)}"]
        if jersey_number:
            evidence.append(f"trusted_jersey={jersey_number}")
        if gallery_person_id:
            evidence.append(f"face_gallery={gallery_person_id}")
            confidence = min(member.gallery_confidence for member in members if member.gallery_person_id)
        if similarities:
            evidence.append(f"minimum_embedding_cosine={min(similarities):.4f}")
        face_similarities = [
            self._face_cosine(left, right) for index, left in enumerate(members) for right in members[index + 1 :]
        ]
        face_similarities = [value for value in face_similarities if value is not None]
        if face_similarities:
            evidence.append(f"minimum_face_cosine={min(face_similarities):.4f}")
        return CanonicalPlayerIdentity(
            player_id=player_id,
            team_id=members[0].team_id,
            tracklet_ids=tuple(sorted(member.tracklet_id for member in members)),
            jersey_number=jersey_number,
            confidence=min(1.0, confidence),
            evidence=tuple(evidence),
        )


def _trusted_jersey(tracklet: IdentityTracklet, threshold: float) -> bool:
    return tracklet.jersey_number is not None and tracklet.jersey_confidence >= threshold


def _ensure_unique_player_ids(
    identities: list[CanonicalPlayerIdentity],
) -> list[CanonicalPlayerIdentity]:
    """Prevent conflicting same-number/gallery groups from sharing one ID."""

    counts: dict[str, int] = {}
    for identity in identities:
        counts[identity.player_id] = counts.get(identity.player_id, 0) + 1
    output = []
    for identity in identities:
        if counts[identity.player_id] == 1:
            output.append(identity)
            continue
        digest = hashlib.sha256("|".join(identity.tracklet_ids).encode("utf-8")).hexdigest()[:8]
        output.append(
            replace(
                identity,
                player_id=f"{identity.player_id}-ambiguous-{digest}",
                evidence=(*identity.evidence, "ambiguous_duplicate_anchor"),
            )
        )
    return output


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _aligned_spatial_distances(
    left: tuple[tuple[int, float, float, float], ...],
    right: tuple[tuple[int, float, float, float], ...],
    *,
    frame_tolerance: int,
) -> list[float]:
    distances: list[float] = []
    for left_frame, left_x, left_y, left_height in left:
        candidates = [item for item in right if abs(item[0] - left_frame) <= frame_tolerance]
        if not candidates:
            continue
        right_frame, right_x, right_y, right_height = min(
            candidates, key=lambda item: (abs(item[0] - left_frame), item[0])
        )
        del right_frame
        scale = max(1.0, (left_height + right_height) / 2.0)
        distances.append(math.hypot(left_x - right_x, left_y - right_y) / scale)
    return distances
