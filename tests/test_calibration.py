"""The confidence measures of rarag_eval/calibration.py on small known cases."""
from rarag_eval import calibration


def test_ece_is_zero_when_confidence_matches_accuracy_and_large_when_it_does_not():
    assert calibration.ece([0.75] * 4, [1, 1, 1, 0]) == 0.0
    assert abs(calibration.ece([1.0] * 4, [1, 0, 0, 0]) - 0.75) < 1e-9


def test_aurc_rewards_ranking_the_errors_last():
    good = calibration.aurc([0.9, 0.8, 0.2], [1, 1, 0])     # the wrong answer is the least confident
    bad = calibration.aurc([0.2, 0.8, 0.9], [1, 1, 0])      # the wrong answer is the most confident
    assert good < bad and abs(good - (0 + 0 + 1 / 3) / 3) < 1e-9
