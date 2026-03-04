from __future__ import annotations

from typing import Optional


def compose_edit_prompt(
    scene_prompt: str,
    reference_prompt: str,
    explicit_edit_prompt: Optional[str] = None,
    placeholder: str = "a person",
) -> str:
    if explicit_edit_prompt is not None and explicit_edit_prompt.strip():
        return explicit_edit_prompt.strip()

    scene = scene_prompt.strip()
    reference = reference_prompt.strip()
    marker = placeholder.strip()

    if marker and marker in scene:
        return scene.replace(marker, reference, 1)

    # Fallback if prompt does not contain the expected placeholder.
    # Keeps pipeline usable while still prioritizing paper-like behavior.
    if "person" in scene:
        return scene.replace("person", reference, 1)

    return f"{scene}, with {reference}"
