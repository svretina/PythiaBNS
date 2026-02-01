from unittest.mock import MagicMock, patch

import bilby

from pythiabns.core.config import PriorConfig
from pythiabns.inference.priors import PriorFactory


def test_create_priors_basic():
    # Mock registry metadata
    with patch("pythiabns.core.registry.ModelRegistry.get_metadata", return_value={}):
        config = PriorConfig(mode="file", source="non_existent.priors")
        priors = PriorFactory.create_priors(config, "test_model")
        assert isinstance(priors, bilby.core.prior.PriorDict)
        assert len(priors) == 0


@patch("pythiabns.core.registry.RelationRegistry.get")
@patch("pythiabns.core.registry.ModelRegistry.get_metadata")
def test_create_priors_empirical(mock_model_meta, mock_rel_reg):
    mock_model_meta.return_value = {}

    # Mock Relation class
    mock_rel_cls = MagicMock()
    mock_rel_inst = mock_rel_cls.return_value
    mock_rel_inst.predict.return_value = {"f_peak": 2500.0}
    mock_rel_reg.return_value = mock_rel_cls

    config = PriorConfig(mode="empirical", source="test_rel")
    metadata = {"id_mass_starA": 1.4, "id_mass_starB": 1.4}

    # Initialize priors with a dummy f_peak
    base_priors = bilby.core.prior.PriorDict()
    base_priors["f_peak"] = bilby.core.prior.Uniform(1000, 4000, name="f_peak")

    with patch("bilby.core.prior.PriorDict.from_file"):
        with patch("bilby.core.prior.PriorDict", return_value=base_priors):
            priors = PriorFactory.create_priors(config, "test_model", metadata=metadata)

    # Verify f_peak was updated based on prediction (2500 +/- width)
    # The logic used in priors.py for Uniform: fpeak - width, fpeak + width (width=500)
    assert priors["f_peak"].minimum == 2000.0
    assert priors["f_peak"].maximum == 3000.0
