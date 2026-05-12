# Copyright 2026 Xin Huang
#
# GNU General Public License v3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html

import pytest
from pydantic import ValidationError

from gaishi.configs import ModelConfig


def test_model_config_valid_logistic_regression():
    cfg = ModelConfig(
        name="logistic_regression",
        params={"C": 1.0, "max_iter": 200, "solver": "liblinear"},
    )

    assert cfg.name == "logistic_regression"
    assert cfg.params["C"] == 1.0
    assert cfg.params["max_iter"] == 200
    assert cfg.params["solver"] == "liblinear"


def test_model_config_valid_extra_trees():
    cfg = ModelConfig(
        name="extra_trees_classifier",
        params={"n_estimators": 500, "max_depth": 8, "n_jobs": -1},
    )

    assert cfg.name == "extra_trees_classifier"
    assert cfg.params["n_estimators"] == 500
    assert cfg.params["max_depth"] == 8
    assert cfg.params["n_jobs"] == -1


def test_model_config_valid_unet_params():
    cfg = ModelConfig(
        name="unet++",
        params={"batch_size": 16, "learning_rate": 0.0005, "val_prop": 0.2},
    )

    assert cfg.name == "unet++"
    assert cfg.params["batch_size"] == 16
    assert cfg.params["learning_rate"] == 0.0005
    assert cfg.params["val_prop"] == 0.2


def test_model_config_unet_allows_site_weighting():
    cfg = ModelConfig(name="unet++", params={"site_weighting": True})
    assert cfg.params["site_weighting"] is True


def test_model_config_invalid_name_raises():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="random_forest",  # not in the Literal
            params={"n_estimators": 100},
        )


def test_model_config_extra_top_level_field_forbidden():
    with pytest.raises(ValidationError):
        ModelConfig(name="logistic_regression", extra_field=1)


def test_model_config_params_default_is_empty_dict():
    cfg = ModelConfig(name="logistic_regression")
    assert cfg.params == {"is_scaled": False}


def test_model_config_unknown_param_raises():
    with pytest.raises(ValidationError):
        ModelConfig(name="logistic_regression", params={"max_iterr": 100})


def test_model_config_unknown_param_extra_trees_raises():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="extra_trees_classifier",
            params={"n_estimatorss": 100},
        )


def test_model_config_unknown_param_unet_plusplus_raises():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="unet++",
            params={"unknown": 1},
        )


def test_model_config_unet_boundaries_are_enforced():
    with pytest.raises(ValidationError):
        ModelConfig(name="unet++", params={"val_prop": 1.2})
