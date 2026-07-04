"""Run with: python3 tests/test_text_objects.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from text_objects import extract_candidate_nouns


def test_basic_extraction():
    caption = "A man is standing near a car and a dog on a city street."
    nouns = extract_candidate_nouns(caption)
    for expected in ["man", "car", "dog", "street"]:
        assert expected in nouns, (expected, nouns)
    for stop in ["a", "is", "near", "and", "on", "the"]:
        assert stop not in nouns, (stop, nouns)
    print("test_basic_extraction OK")


def test_double_word_merge():
    caption = "There is a teddy bear next to a traffic light and a cell phone."
    nouns = extract_candidate_nouns(caption)
    assert "teddy bear" in nouns, nouns
    assert "traffic light" in nouns, nouns
    assert "cell phone" in nouns, nouns
    # the raw halves should not also appear standalone
    assert "teddy" not in nouns, nouns
    assert "bear" not in nouns, nouns
    print("test_double_word_merge OK")


def test_baby_animal_double_word():
    caption = "A baby bird is sitting on a passenger train."
    nouns = extract_candidate_nouns(caption)
    assert "bird" in nouns, nouns
    assert "train" in nouns, nouns
    assert "baby bird" not in nouns
    assert "passenger train" not in nouns
    print("test_baby_animal_double_word OK")


def test_toilet_seat_special_case():
    caption = "A white toilet with the seat up in a small bathroom."
    nouns = extract_candidate_nouns(caption)
    assert "toilet" in nouns
    assert "seat" not in nouns, nouns
    print("test_toilet_seat_special_case OK")


def test_bus_not_corrupted_to_bu():
    # Real caption from the 500-image audit (COCO_val2014_000000061959.jpg):
    # TextBlob's singularize() turned "bus" -> "bu", silently dropping a
    # real MSCOCO category whenever only the VLM (not RAM/DETR) mentions it.
    caption = "A white and orange bus is driving down the street."
    nouns = extract_candidate_nouns(caption)
    assert "bus" in nouns, nouns
    assert "bu" not in nouns, nouns
    print("test_bus_not_corrupted_to_bu OK")


def test_tennis_not_corrupted_to_tenni():
    # Real caption from the audit (COCO_val2014_000000062692.jpg): "tennis"
    # -> "tenni" under the old TextBlob singularization.
    caption = "A woman in a black shirt and white skirt is playing tennis."
    nouns = extract_candidate_nouns(caption)
    assert "tenni" not in nouns, nouns
    print("test_tennis_not_corrupted_to_tenni OK")


def test_empty_caption():
    assert extract_candidate_nouns("") == []
    assert extract_candidate_nouns("   ") == []
    print("test_empty_caption OK")


def test_singularization():
    caption = "Two dogs and several cats are playing with balls."
    nouns = extract_candidate_nouns(caption)
    assert "dog" in nouns, nouns
    assert "cat" in nouns, nouns
    assert "ball" in nouns, nouns
    assert "dogs" not in nouns
    print("test_singularization OK")


if __name__ == "__main__":
    test_basic_extraction()
    test_double_word_merge()
    test_baby_animal_double_word()
    test_toilet_seat_special_case()
    test_bus_not_corrupted_to_bu()
    test_tennis_not_corrupted_to_tenni()
    test_empty_caption()
    test_singularization()
    print("\nALL text_objects.py TESTS PASSED")
