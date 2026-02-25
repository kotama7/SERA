"""Tests for experiment config validation."""

from types import SimpleNamespace

from sera.search.validation import validate_experiment_config


def make_problem_spec(variables=None):
    """Create a mock problem spec with manipulated variables."""
    if variables is None:
        variables = [
            SimpleNamespace(
                name="learning_rate",
                type="float",
                range=[0.0001, 1.0],
                choices=None,
            ),
            SimpleNamespace(
                name="batch_size",
                type="int",
                range=[1, 512],
                choices=None,
            ),
            SimpleNamespace(
                name="optimizer",
                type="categorical",
                range=None,
                choices=["adam", "sgd", "adamw"],
            ),
        ]
    return SimpleNamespace(manipulated_variables=variables)


class TestValidateExperimentConfig:
    """Test whitelist validation of experiment configs."""

    def test_valid_config_all_keys(self):
        """Config with all valid keys and values passes."""
        spec = make_problem_spec()
        config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "optimizer": "adam",
        }
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True
        assert errors == []

    def test_valid_config_subset_keys(self):
        """Config with a subset of valid keys passes."""
        spec = make_problem_spec()
        config = {"learning_rate": 0.1}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True
        assert errors == []

    def test_empty_config(self):
        """Empty config is valid."""
        spec = make_problem_spec()
        config = {}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True
        assert errors == []

    def test_unknown_key_rejected(self):
        """Config with unknown key is invalid."""
        spec = make_problem_spec()
        config = {
            "learning_rate": 0.01,
            "unknown_param": 42,
        }
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert len(errors) == 1
        assert "unknown_param" in errors[0]

    def test_float_out_of_range_low(self):
        """Float value below range is invalid."""
        spec = make_problem_spec()
        config = {"learning_rate": 0.00001}  # min is 0.0001
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert len(errors) == 1
        assert "out of range" in errors[0]

    def test_float_out_of_range_high(self):
        """Float value above range is invalid."""
        spec = make_problem_spec()
        config = {"learning_rate": 5.0}  # max is 1.0
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert "out of range" in errors[0]

    def test_float_at_boundary(self):
        """Float value at boundary is valid."""
        spec = make_problem_spec()
        config = {"learning_rate": 0.0001}  # exactly at min
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True

        config = {"learning_rate": 1.0}  # exactly at max
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True

    def test_int_wrong_type(self):
        """Int variable with float value is invalid."""
        spec = make_problem_spec()
        config = {"batch_size": 32.5}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert "expects int" in errors[0]

    def test_int_out_of_range(self):
        """Int value out of range is invalid."""
        spec = make_problem_spec()
        config = {"batch_size": 1024}  # max is 512
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert "out of range" in errors[0]

    def test_int_at_boundary(self):
        """Int value at boundary is valid."""
        spec = make_problem_spec()
        config = {"batch_size": 1}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True

        config = {"batch_size": 512}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True

    def test_categorical_invalid_choice(self):
        """Categorical value not in choices is invalid."""
        spec = make_problem_spec()
        config = {"optimizer": "rmsprop"}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert "not in choices" in errors[0]

    def test_categorical_valid_choice(self):
        """Categorical value in choices is valid."""
        spec = make_problem_spec()
        config = {"optimizer": "sgd"}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True

    def test_multiple_errors(self):
        """Config with multiple issues reports all errors."""
        spec = make_problem_spec()
        config = {
            "learning_rate": 100.0,  # out of range
            "batch_size": "big",  # wrong type
            "optimizer": "rmsprop",  # invalid choice
            "unknown": True,  # unknown key
        }
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert len(errors) == 4

    def test_float_accepts_int(self):
        """Float variable accepts int value (Python int is a valid float)."""
        spec = make_problem_spec()
        config = {"learning_rate": 1}  # int within float range
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True

    def test_no_manipulated_variables(self):
        """Problem spec with no variables rejects any config keys."""
        spec = SimpleNamespace(manipulated_variables=[])
        config = {"anything": 42}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is False
        assert "Unknown config key" in errors[0]

    def test_no_manipulated_variables_empty_config(self):
        """Empty config with no variables is valid."""
        spec = SimpleNamespace(manipulated_variables=[])
        config = {}
        is_valid, errors = validate_experiment_config(config, spec)
        assert is_valid is True
