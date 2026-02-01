from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pythiabns.data_utils.nr import NumericalWaveform


@pytest.fixture
def mock_simple_waveform(tmp_path):
    wf_file = tmp_path / "custom_wf.txt"
    # time, hp, hc
    data = np.array([[0.0, 1e-22, 0.5e-22], [0.1, 2e-22, 1.0e-22], [0.2, 1e-22, 0.5e-22]])
    np.savetxt(wf_file, data)
    return wf_file


def test_load_simple_file(mock_simple_waveform):
    # Test loading from a simple text file
    wf = NumericalWaveform(str(mock_simple_waveform))
    assert wf.m1 == 1.4
    assert len(wf.time) == 3
    assert wf.hp[0] == 1e-22
    assert hasattr(wf, "_is_si") and wf._is_si


def test_get_post_merger():
    # Setup dummy data with a distinct peak
    wf = MagicMock(spec=NumericalWaveform)
    wf.time = np.array([0, 1, 2, 3, 4])
    wf.hp = np.array([0, 1, 5, 2, 0])
    wf.hc = np.array([0, 0, 0, 0, 0])

    # We call the actual method on the mock (or just use a real object with dummy data)
    # Let's use a real object but bypass __init__
    with patch.object(NumericalWaveform, "__init__", return_value=None):
        wf_obj = NumericalWaveform("dummy")
        wf_obj.time = wf.time
        wf_obj.hp = wf.hp
        wf_obj.hc = wf.hc

        wf_obj.get_post_merger(inplace=True)

        # Peak is at index 2 (val=5)
        assert len(wf_obj.time) == 3
        assert wf_obj.time[0] == 2
        assert wf_obj.hp[0] == 5


@patch("pythiabns.data_utils.nr.h5py.File")
@patch("pythiabns.data_utils.nr.NumericalWaveform._load_metadata")
def test_load_standard_nr(mock_meta, mock_h5, tmp_path):
    # Mocking a standard NR directory structure
    nr_dir = tmp_path / "BAM_0001"
    nr_dir.mkdir()
    (nr_dir / "metadata.txt").touch()

    mock_meta.return_value = {"id_mass_starA": 1.5, "id_mass_starB": 1.3}

    # Mock H5 data structure
    mock_f = MagicMock()
    mock_h5.return_value.__enter__.return_value = mock_f

    # f["/rh_22"] should return a list of keys
    mock_f.__getitem__.side_effect = lambda k: {
        "/rh_22": ["l2_m2_r400.txt"],
        "/rh_22/l2_m2_r400.txt": np.array([[0, 1, 0], [1, 2, 0]]),
    }.get(k, MagicMock())

    with patch("pythiabns.data_utils.nr.os.path.exists", return_value=True):
        wf = NumericalWaveform(str(nr_dir))
        assert wf.m1 == 1.5
        assert wf.m2 == 1.3
        assert len(wf.time) == 2
