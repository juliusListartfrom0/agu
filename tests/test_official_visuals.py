import numpy as np

from app.analysis.official_visuals import build_temporal_contact_sheet


def test_temporal_contact_sheet_preserves_chronological_grid() -> None:
    frames = [np.full((50, 100, 3), value, dtype=np.uint8) for value in (20, 60, 100, 140, 180)]

    sheet = build_temporal_contact_sheet(frames, duration_sec=4, columns=3, cell_width=100)

    assert sheet is not None
    assert sheet.shape == (100, 300, 3)
    # Check unlabelled lower portions of the first and last cells preserve order.
    assert int(sheet[40, 50].mean()) == 20
    assert int(sheet[90, 150].mean()) == 180
