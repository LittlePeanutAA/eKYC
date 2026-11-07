import pytest

from ekyc.postprocess import normalize


@pytest.mark.parametrize("raw, text, ok", [
    ("001099012345", "001099012345", True),
    ("0010 9901 2345", "001099012345", True),
    ("O01099012345", "001099012345", True),
    ("123456789", "123456789", True),
    ("12345", "12345", False),
])
def test_id(raw, text, ok):
    out, warnings = normalize("id", raw)
    assert out == text and (not warnings) == ok


@pytest.mark.parametrize("raw, text, ok", [
    ("01/01/1999", "01/01/1999", True),
    ("1/1/1999", "01/01/1999", True),
    ("01-01-1999", "01/01/1999", True),
    ("31/02/1999", "31/02/1999", False),
    ("01/01/1850", "01/01/1850", False),
    ("abc", "abc", False),
])
def test_dob(raw, text, ok):
    out, warnings = normalize("dob", raw)
    assert out == text and (not warnings) == ok


def test_name_and_place():
    assert normalize("name", " nguyễn  văn an. ")[0] == "NGUYỄN VĂN AN"
    assert normalize("hometown", "Láng Thượng ,Đống Đa,  Hà Nội.")[0] == "Láng Thượng, Đống Đa, Hà Nội"
