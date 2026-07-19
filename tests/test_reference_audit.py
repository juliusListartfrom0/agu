from app.analysis.reference_audit import (
    aggregate_player_stats,
    group_reference_match_samples,
    localize_reference_scenes,
    normalize_detail_rows,
    parse_origin_time,
    reconcile_player_stats,
    suppress_short_scene_boundaries,
)


def test_counting_field_contract_is_stable() -> None:
    from app.analysis.reference_audit import COUNTING_FIELDS

    assert len(COUNTING_FIELDS) == 16
    assert len(set(COUNTING_FIELDS)) == len(COUNTING_FIELDS)


def test_sparse_reference_matches_merge_across_short_rejected_gap() -> None:
    samples = [
        {"reference_time": 0.0, "raw_time": 10.0, "raw_video": "p1.mov", "distance": 2.0, "margin": 8.0, "accepted": True},
        {"reference_time": 1.0, "raw_time": 99.0, "raw_video": "p2.mov", "distance": 90.0, "margin": 1.0, "accepted": False},
        {"reference_time": 2.0, "raw_time": 12.0, "raw_video": "p1.mov", "distance": 3.0, "margin": 7.0, "accepted": True},
    ]

    segments = group_reference_match_samples(samples)

    assert len(segments) == 1
    assert segments[0]["matched_samples"] == 2
    assert segments[0]["raw_start"] == 10.0
    assert segments[0]["raw_end"] == 12.0


def test_scene_localization_uses_dominant_stable_offset() -> None:
    samples = [
        {"reference_time": 1.0, "raw_time": 21.0, "raw_video": "p1.mov", "distance": 2.0, "margin": 8.0, "accepted": True},
        {"reference_time": 2.0, "raw_time": 22.0, "raw_video": "p1.mov", "distance": 3.0, "margin": 7.0, "accepted": True},
        {"reference_time": 3.0, "raw_time": 99.0, "raw_video": "p2.mov", "distance": 4.0, "margin": 6.0, "accepted": True},
    ]

    segments = localize_reference_scenes(samples, [0.0, 5.0])

    assert segments[0]["status"] == "localized"
    assert segments[0]["raw_video"] == "p1.mov"
    assert segments[0]["raw_start"] == 20.0
    assert segments[0]["raw_end"] == 25.0


def test_scene_localization_allows_one_near_exact_short_scene_match() -> None:
    samples = [
        {"reference_time": 1.0, "raw_time": 21.0, "raw_video": "p1.mov", "distance": 6.0, "margin": 3.0, "accepted": True},
    ]

    segments = localize_reference_scenes(samples, [0.0, 2.0])

    assert segments[0]["status"] == "localized"
    assert segments[0]["matched_samples"] == 1


def test_transition_frame_boundaries_are_collapsed() -> None:
    boundaries = [0.0, 10.0, 10.033, 10.2, 20.0]

    assert suppress_short_scene_boundaries(boundaries) == [0.0, 10.0, 20.0]


def test_parse_origin_time() -> None:
    assert parse_origin_time("6 - 24 : 28") == (6, 1468)


def test_relation_rows_expand_into_atomic_events() -> None:
    rows = [
        {"Team": "白队", "Player": "A", "OriginTime": "1 - 1 : 00", "Event": "助攻", "Info": "3分", "Object": "B"},
        {"Team": "黑队", "Player": "C", "OriginTime": "1 - 1 : 10", "Event": "盖帽", "Info": "2分", "Object": "B"},
        {"Team": "黑队", "Player": "D", "OriginTime": "1 - 1 : 20", "Event": "抢断", "Info": "", "Object": "B"},
        {"Team": "白队", "Player": "B", "OriginTime": "1 - 1 : 30", "Event": "犯规", "Info": "进攻犯规", "Object": "D"},
    ]

    events = normalize_detail_rows(rows)
    stats = aggregate_player_stats(events)

    assert len(events) == 8
    assert stats[("白队", "A")]["assists"] == 1
    assert stats[("白队", "B")]["three_pt_made"] == 1
    assert stats[("白队", "B")]["two_pt_attempted"] == 1
    assert stats[("白队", "B")]["blocked_shots"] == 1
    assert stats[("白队", "B")]["turnovers"] == 2
    assert stats[("黑队", "C")]["blocks"] == 1
    assert stats[("黑队", "D")]["steals"] == 1


def test_reconciliation_reports_exact_denominator() -> None:
    expected = {("黑队", "A"): {"points": 2}}
    actual = {("黑队", "A"): {"points": 2}}
    result = reconcile_player_stats(actual, expected)
    assert result["field_accuracy"] == 1.0
    assert result["matched_fields"] == result["total_fields"] == 16
    assert result["mismatches"] == []
