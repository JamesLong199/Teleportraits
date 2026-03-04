from teleportraits.prompts import compose_edit_prompt


def test_compose_from_placeholder() -> None:
    scene = "a wide city street with a person near a crosswalk"
    ref = "a woman in a red coat"
    out = compose_edit_prompt(scene, ref)
    assert out == "a wide city street with a woman in a red coat near a crosswalk"


def test_compose_with_override() -> None:
    scene = "a person in a park"
    ref = "a man with sunglasses"
    out = compose_edit_prompt(scene, ref, explicit_edit_prompt="custom edit prompt")
    assert out == "custom edit prompt"
