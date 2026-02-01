import pytest
from unittest.mock import MagicMock, patch
from pythiabns import spine
from pythiabns.core import config

@patch("pythiabns.spine.NumericalWaveform")
@patch("pythiabns.spine.DetectorNetwork")
@patch("pythiabns.spine.bilby.gw.waveform_generator.WaveformGenerator")
@patch("pythiabns.spine.PriorFactory.create_priors")
@patch("pythiabns.spine.PostMergerLikelihood")
@patch("pythiabns.spine.PocoMCWrapper")
def test_run_simulation_wiring(mock_pocomc, mock_likelihood, mock_priors, mock_wg, mock_network, mock_nr, tmp_path):
    # Setup mocks
    mock_nr_inst = mock_nr.return_value
    mock_nr_inst.metadata_dict = {}
    mock_nr_inst.m1 = 1.4
    mock_nr_inst.m2 = 1.4
    
    mock_network_inst = mock_network.return_value
    mock_network_inst.duration = 1.0
    mock_network_inst.sampling_frequency = 4096.0
    mock_network_inst.start_time = 0.0
    
    # Mock Sim Config
    sim_config = config.SimulationConfig(
        injection=config.InjectionConfig(mode="analytic", target="three_sines"),
        snr=100.0,
        model="three_sines",
        sampler=config.SamplerConfig(plugin="pocomc"),
        priors=config.PriorConfig(mode="file", source="test.priors")
    )

    # Need to register the model effectively?
    # spine.py imports 'pythiabns.models.tutorial_models' via config imports usually.
    # But here we are calling run_simulation directly.
    # The 'three_sines' model is in tutorial_models.py, but we might not have imported it.
    from pythiabns.core.registry import ModelRegistry
    @ModelRegistry.register("three_sines")
    def mock_three_sines(time, **kwargs):
      return {"plus": 0, "cross": 0}
    
    # Run
    plotting_config = config.PlottingConfig(enabled=False)
    spine.run_simulation(sim_config, plotting_config, tmp_path)
    
    # Verifications
    # 1. NR data loaded (conceptually, even if analytic used for config, spine checks waveform prop? No, it uses nr_data for metadata)
    # Actually, if injection mode is 'analytic', spine.py still might try to load NR data if sim_config.waveform was passed? 
    # In the new logic:
    # if inj.mode in ["nr", "file"]: load NR
    # else: nr_data is None.
    # But later: priors = PriorFactory... nr_data.metadata_dict if nr_data else {}
    # So if analytic only, nr_data is None.
    
    # 2. Network setup
    mock_network.assert_called()
    
    # 3. Waveform Generator created
    mock_wg.assert_called()
    
    # 4. Sampler initialized and run
    mock_pocomc.assert_called()
    mock_pocomc.return_value.run.assert_called_once()

@patch("pythiabns.spine.run_simulation")
def test_main_matrix_expansion(mock_run, tmp_path):
    # Create a config file
    config_file = tmp_path / "test.yaml"
    import yaml
    content = {
        "name": "Test",
        "matrix": {
            "injection": [{"mode": "analytic", "target": "modelA"}, {"mode": "analytic", "target": "modelB"}],
            "model": ["model1"],
            "snr": [10.0],
            "sampler": {"plugin": "dynesty"},
            "priors": {"mode": "file", "source": "test.priors"}
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(content, f)
        
    # Patch argv
    with patch("sys.argv", ["spine.py", str(config_file)]):
        spine.main()
        
    # Should call run_simulation twice
    assert mock_run.call_count == 2
