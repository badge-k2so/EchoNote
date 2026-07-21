from __future__ import annotations


def decode_windows_process_output(value: bytes) -> str:
    if not value:
        return ""
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return value.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace").strip()
