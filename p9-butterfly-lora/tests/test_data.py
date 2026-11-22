import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sdlora.data import caption_for, clean_name, family_of, records_from_frame, split_indices, to_square, write_images


def _png(w: int, h: int, colour=(200, 10, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_family_and_name_fallbacks():
    assert family_of("Animalia, Arthropoda, Insecta, Lepidoptera, Nymphalidae, Morpho") == "Nymphalidae"
    assert family_of(None) == "Lepidoptera"
    assert family_of("lowercaseidae, Insecta") == "Lepidoptera"
    assert clean_name("  Blue   Morpho ", "Morpho menelaus") == "Blue Morpho"
    assert clean_name(None, "Morpho menelaus") == "Morpho menelaus"
    assert clean_name(float("nan"), "") == "unidentified butterfly"
    assert caption_for("Monarch", None, "Lepidoptera, Nymphalidae") == \
        "a museum specimen photo of a Monarch, Nymphalidae family"


def test_split_is_fixed_and_sized():
    a, b = split_indices(100, 20, 0), split_indices(100, 20, 0)
    assert a.sum() == 20 and (a == b).all()
    assert not (a == split_indices(100, 20, 1)).all()
    with pytest.raises(ValueError):
        split_indices(10, 10, 0)


def test_to_square_pads_with_border_colour():
    img = Image.new("RGB", (40, 20), (0, 0, 255))
    out = to_square(img, 64)
    assert out.size == (64, 64)
    assert np.asarray(out)[0, 0].tolist() == [0, 0, 255]  # padding uses the border colour, not black


def test_records_and_images(tmp_path):
    df = pd.DataFrame({
        "id": [f"x{i}" for i in range(6)], "name": ["Monarch", None, "Skipper", "A", "B", "C"],
        "scientific_name": ["Danaus plexippus", "Papilio sp.", None, "a", "b", "c"],
        "taxonomy": ["Nymphalidae", "Papilionidae", "Hesperiidae", None, None, None], "source": ["s"] * 6,
        "image": [{"bytes": _png(30, 20)}] * 3 + [_png(20, 30)] * 3,
    })
    recs = records_from_frame(df, n_test=2, split_seed=0)
    assert [r["idx"] for r in recs] == list(range(6))
    assert sum(r["split"] == "test" for r in recs) == 2
    assert recs[1]["name"] == "Papilio sp." and recs[2]["scientific_name"] == ""
    write_images(df, recs, tmp_path, 64)
    assert Image.open(tmp_path / "0005.png").size == (64, 64)
