import numpy as np
from PIL import Image

from sdlora.metrics import l2n
from sdlora.safety import RED_TEAM, check_prompt, concept_flags, evaluate_prompt_filter, normalize, save_with_provenance


def test_normalize_undoes_obfuscation():
    assert "nude" in normalize("n.u.d.e")
    assert "nude" in normalize("NUD3")


def test_red_team_balanced_and_passes():
    assert sum(b for _, b in RED_TEAM) == sum(not b for _, b in RED_TEAM) == 30
    r = evaluate_prompt_filter()
    assert r["recall"] == 1.0 and r["false_positive_rate"] == 0.0, (r["missed"], r["false_blocks"])


def test_species_names_are_not_blocked():
    for p in ["a museum specimen photo of a Essex Skipper, Hesperiidae family",
              "a museum specimen photo of a Dead Leaf, Nymphalidae family",
              "blood-red wings visible to the naked eye"]:
        assert check_prompt(p).allowed, p


def test_minors_category():
    v = check_prompt("a nude teen")
    assert not v.allowed and "minors" in v.categories


def test_concept_flags_by_nearest_concept():
    unsafe, safe = l2n(np.eye(4)[:2]), l2n(np.eye(4)[2:])
    imgs = l2n(np.array([[1.0, 0, 0.1, 0], [0, 0, 1.0, 0.1]]))
    flags, mass = concept_flags(imgs, unsafe, safe)
    assert flags.tolist() == [True, False] and mass.min() >= 0 and mass.max() <= 1


def test_provenance_round_trip(tmp_path):
    p = tmp_path / "sub" / "x.png"
    save_with_provenance(Image.new("RGB", (8, 8)), p, prompt="a moth", seed=3)
    info = Image.open(p).info
    assert info["ai_generated"] == "true" and info["prompt"] == "a moth" and info["seed"] == "3"
