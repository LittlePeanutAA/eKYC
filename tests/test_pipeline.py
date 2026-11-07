"""End-to-end tests on the fictitious specimen cards. They need the released weights (downloaded on first run)."""
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "specimen_1.jpg": {"id": "001099012345", "name": "NGUYỄN VĂN AN", "dob": "01/01/1999",
                       "hometown": "Láng Thượng, Đống Đa, Hà Nội"},
    "specimen_2.jpg": {"id": "079301005678", "name": "TRẦN THỊ BÌNH", "dob": "15/06/2001",
                       "hometown": "Tân Định, Quận 1, TP Hồ Chí Minh"},
}


@pytest.fixture(scope="module")
def ekyc():
    from ekyc import EKYC

    try:
        return EKYC.from_pretrained("corners")
    except OSError as e:  # offline and no cached weights
        pytest.skip(f"weights unavailable: {e}")


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_specimen(ekyc, name):
    result = ekyc(cv2.imread(str(ROOT / "examples" / name)))
    assert {k: v.text for k, v in result.fields.items()} == EXPECTED[name]
    assert result.face_box is not None
    assert not result.warnings


def test_no_card(ekyc):
    import numpy as np

    from ekyc import CardNotFoundError

    with pytest.raises(CardNotFoundError):
        ekyc(np.full((480, 640, 3), 255, np.uint8))


def test_api(ekyc, monkeypatch):
    from fastapi.testclient import TestClient

    from ekyc import api

    monkeypatch.setattr(api, "get_model", lambda: ekyc)
    client = TestClient(api.app)
    data = (ROOT / "examples" / "specimen_1.jpg").read_bytes()
    r = client.post("/v1/id-card/extract", files={"file": ("card.jpg", data, "image/jpeg")})
    assert r.status_code == 200 and r.json()["fields"]["id"]["text"] == "001099012345"
    assert client.post("/v1/id-card/extract", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 415
