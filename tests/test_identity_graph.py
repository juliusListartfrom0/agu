from app.analysis.identity_graph import IdentityGraph, IdentityTracklet


def test_identity_graph_merges_disjoint_same_jersey_tracklets() -> None:
    graph = IdentityGraph()
    identities = graph.resolve(
        [
            IdentityTracklet("t1", "dark", 0, 20, (1.0, 0.0), jersey_number="15", jersey_confidence=0.95),
            IdentityTracklet("t2", "dark", 30, 50, (0.7, 0.1), jersey_number="15", jersey_confidence=0.93),
        ]
    )

    assert len(identities) == 1
    assert identities[0].player_id == "dark-jersey-15"
    assert identities[0].tracklet_ids == ("t1", "t2")


def test_identity_graph_never_reuses_player_id_for_conflicting_same_jersey_groups() -> None:
    identities = IdentityGraph(overlap_tolerance_frames=0).resolve(
        [
            IdentityTracklet(
                "t1",
                "light",
                0,
                20,
                (1.0, 0.0),
                jersey_number="30",
                jersey_confidence=0.95,
                spatial_samples=((10, 0.0, 0.0, 100.0),),
            ),
            IdentityTracklet(
                "t2",
                "light",
                0,
                20,
                (0.0, 1.0),
                jersey_number="30",
                jersey_confidence=0.95,
                spatial_samples=((10, 300.0, 0.0, 100.0),),
            ),
        ]
    )

    assert len({identity.player_id for identity in identities}) == 2
    assert all("-ambiguous-" in identity.player_id for identity in identities)
    assert all("ambiguous_duplicate_anchor" in identity.evidence for identity in identities)


def test_trusted_jersey_anchor_precedes_conflicting_body_reid_edge() -> None:
    identities = IdentityGraph(embedding_threshold=0.9, overlap_tolerance_frames=0).resolve(
        [
            IdentityTracklet(
                "jersey-early",
                "light",
                0,
                10,
                (1.0, 0.0),
                jersey_number="6",
                jersey_confidence=0.96,
            ),
            IdentityTracklet(
                "jersey-late",
                "light",
                30,
                40,
                (0.0, 1.0),
                jersey_number="6",
                jersey_confidence=0.95,
                spatial_samples=((35, 300.0, 0.0, 100.0),),
            ),
            IdentityTracklet(
                "body-lookalike",
                "light",
                35,
                45,
                (1.0, 0.0),
                spatial_samples=((35, 0.0, 0.0, 100.0),),
            ),
        ]
    )

    jersey_identity = next(identity for identity in identities if identity.jersey_number == "6")
    assert jersey_identity.tracklet_ids == ("jersey-early", "jersey-late")
    assert len(identities) == 2


def test_identity_graph_never_merges_simultaneous_players_or_conflicting_jerseys() -> None:
    graph = IdentityGraph(embedding_threshold=0.8)
    identities = graph.resolve(
        [
            IdentityTracklet("t1", "light", 0, 30, (1.0, 0.0), jersey_number="7", jersey_confidence=0.95),
            IdentityTracklet("t2", "light", 10, 35, (1.0, 0.0), jersey_number="7", jersey_confidence=0.95),
            IdentityTracklet("t3", "light", 40, 60, (1.0, 0.0), jersey_number="24", jersey_confidence=0.95),
        ]
    )

    assert len(identities) == 3
    assert {identity.jersey_number for identity in identities} == {"7", "24"}


def test_identity_graph_assigns_stable_anonymous_id_without_reference_roster() -> None:
    tracklet = IdentityTracklet("track-a", "dark", 0, 20, (0.2, 0.4))
    first = IdentityGraph().resolve([tracklet])[0]
    second = IdentityGraph().resolve([tracklet])[0]

    assert first.player_id == second.player_id
    assert first.player_id.startswith("dark-player-")


def test_identity_graph_uses_complete_link_not_similarity_chaining() -> None:
    graph = IdentityGraph(embedding_threshold=0.75)
    identities = graph.resolve(
        [
            IdentityTracklet("a", "dark", 0, 10, (1.0, 0.0)),
            IdentityTracklet("b", "dark", 20, 30, (0.8, 0.6)),
            IdentityTracklet("c", "dark", 40, 50, (0.28, 0.96)),
        ]
    )

    assert len(identities) == 2


def test_identity_graph_allows_same_player_across_overlapping_video_timelines() -> None:
    graph = IdentityGraph(embedding_threshold=0.9)
    identities = graph.resolve(
        [
            IdentityTracklet("period-1:t1", "light", 0, 30, (1.0, 0.0), source_video_id="period-1"),
            IdentityTracklet("period-2:t1", "light", 0, 30, (1.0, 0.0), source_video_id="period-2"),
        ]
    )

    assert len(identities) == 1


def test_identity_graph_allows_duplicate_overlapping_boxes_for_same_player() -> None:
    graph = IdentityGraph(embedding_threshold=0.9)
    shared_position = ((10, 100.0, 200.0, 160.0), (12, 102.0, 201.0, 160.0))
    identities = graph.resolve(
        [
            IdentityTracklet("t1", "dark", 0, 20, (1.0, 0.0), spatial_samples=shared_position),
            IdentityTracklet("t2", "dark", 5, 25, (1.0, 0.0), spatial_samples=shared_position),
        ]
    )

    assert len(identities) == 1


def test_identity_graph_rejects_overlapping_spatially_distinct_players() -> None:
    graph = IdentityGraph(embedding_threshold=0.9)
    identities = graph.resolve(
        [
            IdentityTracklet("t1", "dark", 0, 20, (1.0, 0.0), spatial_samples=((10, 100.0, 200.0, 160.0),)),
            IdentityTracklet("t2", "dark", 5, 25, (1.0, 0.0), spatial_samples=((10, 400.0, 200.0, 160.0),)),
        ]
    )

    assert len(identities) == 2


def test_identity_graph_rejects_overlapping_tracks_without_aligned_samples() -> None:
    graph = IdentityGraph(embedding_threshold=0.9, overlap_tolerance_frames=2)
    identities = graph.resolve(
        [
            IdentityTracklet(
                "t1",
                "dark",
                0,
                40,
                (1.0, 0.0),
                spatial_samples=((5, 100.0, 200.0, 160.0),),
            ),
            IdentityTracklet(
                "t2",
                "dark",
                20,
                60,
                (1.0, 0.0),
                spatial_samples=((35, 100.0, 200.0, 160.0),),
            ),
        ]
    )

    assert len(identities) == 2


def test_identity_graph_uses_gallery_identity_and_rejects_cross_person_merge() -> None:
    graph = IdentityGraph(embedding_threshold=0.80)
    identities = graph.resolve(
        [
            IdentityTracklet("a", "dark", 0, 10, (1.0, 0.0), gallery_person_id="player-01", gallery_confidence=0.93),
            IdentityTracklet("b", "dark", 20, 30, (0.0, 1.0), gallery_person_id="player-01", gallery_confidence=0.91),
            IdentityTracklet("c", "dark", 40, 50, (1.0, 0.0), gallery_person_id="player-02", gallery_confidence=0.95),
        ]
    )

    assert len(identities) == 2
    player_01 = next(item for item in identities if item.player_id == "player-01")
    assert player_01.tracklet_ids == ("a", "b")
    assert "face_gallery=player-01" in player_01.evidence


def test_identity_graph_does_not_propagate_gallery_identity_by_body_only() -> None:
    graph = IdentityGraph(embedding_threshold=0.80)
    identities = graph.resolve(
        [
            IdentityTracklet(
                "anchor",
                "dark",
                0,
                10,
                (1.0, 0.0),
                gallery_person_id="player-01",
                gallery_confidence=0.93,
            ),
            IdentityTracklet("body-only", "dark", 20, 30, (1.0, 0.0)),
        ]
    )

    assert len(identities) == 2
    enrolled = next(item for item in identities if item.player_id == "player-01")
    assert enrolled.tracklet_ids == ("anchor",)


def test_identity_graph_propagates_gallery_identity_with_direct_face_evidence() -> None:
    graph = IdentityGraph(embedding_threshold=0.80)
    identities = graph.resolve(
        [
            IdentityTracklet(
                "anchor",
                "dark",
                0,
                10,
                (1.0, 0.0),
                gallery_person_id="player-01",
                gallery_confidence=0.93,
                face_embedding=(1.0, 0.0),
                face_embedding_model="sface-test",
                face_quality=0.9,
            ),
            IdentityTracklet(
                "face-confirmed",
                "dark",
                20,
                30,
                (1.0, 0.0),
                face_embedding=(0.99, 0.05),
                face_embedding_model="sface-test",
                face_quality=0.9,
            ),
        ]
    )

    assert len(identities) == 1
    assert identities[0].player_id == "player-01"
    assert identities[0].tracklet_ids == ("anchor", "face-confirmed")


def test_identity_graph_uses_quality_gated_face_for_merge_and_conflict() -> None:
    graph = IdentityGraph(embedding_threshold=0.80)
    identities = graph.resolve(
        [
            IdentityTracklet(
                "a",
                "dark",
                0,
                10,
                (1.0, 0.0),
                face_embedding=(1.0, 0.0),
                face_embedding_model="sface-test",
                face_quality=0.9,
            ),
            IdentityTracklet(
                "b",
                "dark",
                20,
                30,
                (0.0, 1.0),
                face_embedding=(0.99, 0.05),
                face_embedding_model="sface-test",
                face_quality=0.9,
            ),
            IdentityTracklet(
                "c",
                "dark",
                40,
                50,
                (1.0, 0.0),
                face_embedding=(0.0, 1.0),
                face_embedding_model="sface-test",
                face_quality=0.9,
            ),
        ]
    )

    assert len(identities) == 2
    merged = next(item for item in identities if item.tracklet_ids == ("a", "b"))
    assert any(value.startswith("minimum_face_cosine=") for value in merged.evidence)
