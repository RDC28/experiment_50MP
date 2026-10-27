"""Tiny, mocked runtime checks for PRD_v3 compute budgets."""

from io import StringIO
from types import SimpleNamespace

import pytest

from synthraw.environment import EnvironmentInspector


@pytest.mark.parametrize(
    ("vram", "tile", "overlap", "offload", "heavy"),
    [
        (None, 192, 32, True, False),
        (0, 192, 32, True, False),
        (8, 192, 32, True, True),
        (11.99, 192, 32, True, True),
        (12, 256, 64, False, True),
        (19.99, 256, 64, False, True),
        (20, 512, 64, False, True),
        (39.99, 512, 64, False, True),
        (40, 768, 96, False, True),
        (80, 768, 96, False, True),
    ],
)
def test_tile_bands(vram, tile, overlap, offload, heavy):
    parameters = EnvironmentInspector.derive_tile_parameters(vram)
    assert (parameters.tile_size, parameters.overlap) == (tile, overlap)
    assert parameters.aggressive_offload is offload
    assert parameters.heavy_reconstruction_supported is heavy


@pytest.mark.parametrize("vram", [-1, float("nan"), float("inf")])
def test_invalid_vram(vram):
    with pytest.raises(ValueError, match="VRAM"):
        EnvironmentInspector.derive_tile_parameters(vram)


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setattr("synthraw.environment.psutil.virtual_memory", lambda: SimpleNamespace(total=32 * 1024**3))
    monkeypatch.setattr("synthraw.environment.shutil.disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr("synthraw.environment.os.cpu_count", lambda: 8)


def test_missing_torch_and_printed_startup(resources, monkeypatch, tmp_path):
    def missing(name):
        assert name == "torch"
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr("synthraw.environment.importlib.import_module", missing)
    inspector = EnvironmentInspector()
    report = inspector.inspect(tmp_path)
    assert report.gpu_model is None
    assert report.pytorch_version is None
    assert report.system_ram_bytes == 32 * 1024**3
    assert report.free_disk_bytes == 100 * 1024**3
    assert report.cpu_cores == 8
    assert report.to_dict()["tiles"]["heavy_reconstruction_supported"] is False
    output = StringIO()
    inspector.print_report(report, file=output)
    for label in ["GPU model", "GPU VRAM", "CUDA version", "PyTorch version", "System RAM", "Free disk", "CPU cores"]:
        assert label in output.getvalue()
    assert "sensor/metadata" in output.getvalue()


@pytest.mark.parametrize("cuda_available", [True, False])
def test_torch_with_and_without_gpu(resources, monkeypatch, cuda_available):
    cuda = SimpleNamespace(is_available=lambda: cuda_available)
    if cuda_available:
        cuda.current_device = lambda: 2

        def properties(device):
            assert device == 2
            return SimpleNamespace(name="Mock A100", total_memory=40 * 1024**3)

        cuda.get_device_properties = properties
    torch = SimpleNamespace(__version__="test-version", version=SimpleNamespace(cuda="test-cuda"), cuda=cuda)
    monkeypatch.setattr("synthraw.environment.importlib.import_module", lambda name: torch)
    report = EnvironmentInspector().inspect()
    assert report.pytorch_version == "test-version"
    assert report.cuda_version == "test-cuda"
    assert report.gpu_model == ("Mock A100" if cuda_available else None)
    assert report.tiles.tile_size == (768 if cuda_available else 192)


def test_unknown_core_count_uses_one(resources, monkeypatch):
    monkeypatch.setattr("synthraw.environment.os.cpu_count", lambda: None)
    torch = SimpleNamespace(__version__="cpu", version=SimpleNamespace(cuda=None))
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr("synthraw.environment.importlib.import_module", lambda name: torch)
    assert EnvironmentInspector().inspect().cpu_cores == 1
