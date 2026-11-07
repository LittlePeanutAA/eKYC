"""Normalise and validate the raw OCR text of each field."""
from __future__ import annotations

import re
from datetime import date

ID_LENGTHS = {12: "CCCD", 9: "CMND"}  # 12-digit citizen ID card, 9-digit legacy ID card


def normalize_id(text: str) -> tuple[str, list[str]]:
    digits = re.sub(r"\D", "", text.replace("O", "0").replace("o", "0"))
    warnings = [] if len(digits) in ID_LENGTHS else [f"id: expected 9 or 12 digits, got {len(digits)}"]
    return digits, warnings


def normalize_dob(text: str) -> tuple[str, list[str]]:
    m = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", text)
    if not m:
        return text.strip(), [f"dob: not a dd/mm/yyyy date: {text!r}"]
    d, mth, y = map(int, m.groups())
    try:
        born = date(y, mth, d)
    except ValueError:
        return text.strip(), [f"dob: invalid date {d:02d}/{mth:02d}/{y}"]
    warnings = [] if 1900 <= y <= date.today().year and born <= date.today() else [f"dob: implausible year {y}"]
    return f"{d:02d}/{mth:02d}/{y}", warnings


def normalize_name(text: str) -> tuple[str, list[str]]:
    name = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip().upper()
    return name, [] if name else ["name: empty"]


def normalize_place(text: str) -> tuple[str, list[str]]:
    place = re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", " ", text)).strip(" ,.")
    return place, [] if place else ["place: empty"]


NORMALIZERS = {"id": normalize_id, "dob": normalize_dob, "name": normalize_name, "hometown": normalize_place}


def normalize(field: str, text: str) -> tuple[str, list[str]]:
    fn = NORMALIZERS.get(field)
    return fn(text) if fn else (text.strip(), [])
