"""Tiny filesystem and RNG checks for PRD_v3 reproducibility."""

from datetime import datetime

import numpy as np
import pytest

from synthraw.run_layout import RUN_DIRECTORIES, RandomStreams, RunLayout


def test_exact_output_tree(tmp_path):
    layout = RunLayout.create(tmp_path / "outputs", timestamp=datetime(2026, 10, 26, 9, 8, 7))
    assert layout.root.name == "run_20261026_090807"
    assert {p.name for p in layout.root.iterdir()} == set(RUN_DIRECTORIES)
    for name in RUN_DIRECTORIES:
        assert layout.directory(name).is_dir()
        assert list(layout.directory(name).iterdir()) == []


def test_collision_preserves_existing_run(tmp_path):
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    layout = RunLayout.create(tmp_path, timestamp=timestamp)
    marker = layout.directory("source") / "original.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        RunLayout.create(tmp_path, timestamp=timestamp)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_default_timestamp(tmp_path):
    before = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    layout = RunLayout.create(tmp_path)
    after = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    assert layout.root.name in (before, after)


@pytest.mark.parametrize("name", ["../source", "other", "", "/tmp"])
def test_unknown_directory(tmp_path, name):
    with pytest.raises(ValueError, match="Unknown run directory"):
        RunLayout(tmp_path).directory(name)


def test_defaults_reproduce_each_stream():
    first, second = RandomStreams.from_seeds(), RandomStreams.from_seeds()
    for name, seed in [("scene", 42), ("sensor_noise", 42001), ("sensor_body", 42002)]:
        samples = getattr(first, name).random(16)
        np.testing.assert_array_equal(samples, getattr(second, name).random(16))
        np.testing.assert_array_equal(samples, np.random.default_rng(seed).random(16))


@pytest.mark.parametrize("changed", ["scene", "sensor_noise", "sensor_body"])
def test_seed_changes_are_isolated(changed):
    seeds = {"scene": 42, "sensor_noise": 42001, "sensor_body": 42002}
    baseline = RandomStreams.from_seeds(**{f"{key}_seed": value for key, value in seeds.items()})
    seeds[changed] += 1
    modified = RandomStreams.from_seeds(**{f"{key}_seed": value for key, value in seeds.items()})
    for name in seeds:
        equal = np.array_equal(getattr(baseline, name).random(16), getattr(modified, name).random(16))
        assert equal == (name != changed)


def test_consumption_is_isolated_and_global_rng_untouched():
    before = np.random.get_state()
    streams = RandomStreams.from_seeds()
    streams.sensor_noise.random(32)
    np.testing.assert_array_equal(streams.scene.random(16), RandomStreams.from_seeds().scene.random(16))
    np.testing.assert_array_equal(streams.sensor_body.random(16), RandomStreams.from_seeds().sensor_body.random(16))
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42", None])
@pytest.mark.parametrize("argument", ["scene_seed", "sensor_noise_seed", "sensor_body_seed"])
def test_invalid_seeds(seed, argument):
    with pytest.raises(ValueError, match="Seeds"):
        RandomStreams.from_seeds(**{argument: seed})
