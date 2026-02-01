import yaml

from pythiabns.core import config


def test_injection_config_validation():
    # Test valid NR injection
    inj = config.InjectionConfig(mode="nr", target="BAM:0088:R01")
    assert inj.mode == "nr"
    assert inj.target == "BAM:0088:R01"

    # Test valid analytic injection with params
    inj = config.InjectionConfig(mode="analytic", target="three_sines", parameters={"f1": 150.0})
    assert inj.parameters["f1"] == 150.0


def test_job_matrix_normalization(tmp_path):
    # Test legacy format normalization
    matrix_data = {
        "waveform": ["BAM:0088:R01", "BAM:0089:R01"],
        "snr": [100.0],
        "model": ["three_sines"],
        "sampler": {"plugin": "dynesty"},
        "priors": {"mode": "file", "source": "test.priors"},
    }

    # We need to test the logic that happens in spine.py (or move it to config.py)
    # Since spine.py does the normalization, let's verify JobMatrix can hold legacy values.
    matrix = config.JobMatrix(**matrix_data)
    assert matrix.waveform == ["BAM:0088:R01", "BAM:0089:R01"]
    assert matrix.injection is None


def test_experiment_config_loading(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    content = {
        "name": "Test",
        "matrix": {
            "injection": [{"mode": "analytic", "target": "model1"}],
            "model": ["model1"],
            "sampler": {"plugin": "dynesty"},
            "priors": {"mode": "file", "source": "test.priors"},
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(content, f)

    cfg = config.load_config(config_file)
    assert cfg.name == "Test"
    assert cfg.matrix.injection[0].mode == "analytic"
