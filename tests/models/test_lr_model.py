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

import json
import os
import shutil

import onnxruntime as ort
import pandas as pd
import pytest
from gaishi.models import LrModel


@pytest.fixture
def file_paths():
    output_dir = "tests/test_LrModel"
    return {
        "training_data": "tests/data/test.training.features",
        "inference_data": "tests/data/test.inference.features",
        "training_output": os.path.join(output_dir, "test.lr.onnx"),
        "inference_output": os.path.join(output_dir, "test.lr.pred.tsv"),
        "model_params": {
            "solver": "newton-cg",
            "penalty": None,
            "max_iter": 10000,
            "random_state": 12345,
        },
        "output_dir": output_dir,
    }


@pytest.fixture
def cleanup_output_dir(request, file_paths):
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(file_paths["output_dir"], ignore_errors=True)


def test_LrModel_train(file_paths, cleanup_output_dir):
    os.makedirs(file_paths["output_dir"], exist_ok=True)

    LrModel.train(
        data=file_paths["training_data"],
        output=file_paths["training_output"],
        **file_paths["model_params"],
    )

    session = ort.InferenceSession(
        file_paths["training_output"],
        providers=["CPUExecutionProvider"],
    )
    metadata = session.get_modelmeta().custom_metadata_map

    assert json.loads(metadata["classes"]) == [0, 1]
    assert len(session.get_inputs()) == 1


def test_LrModel_infer(file_paths, cleanup_output_dir):
    os.makedirs(file_paths["output_dir"], exist_ok=True)

    LrModel.train(
        data=file_paths["training_data"],
        output=file_paths["training_output"],
        **file_paths["model_params"],
    )
    LrModel.infer(
        data=file_paths["inference_data"],
        model=file_paths["training_output"],
        output=file_paths["inference_output"],
        **file_paths["model_params"],
    )

    df = pd.read_csv(file_paths["inference_output"], sep="\t")
    expected_df = pd.read_csv(
        "tests/expected_results/models/test.lr.pred.tsv",
        sep="\t",
    )

    pd.testing.assert_frame_equal(
        df,
        expected_df,
        check_dtype=False,
        check_like=False,
        rtol=1e-5,
        atol=1e-5,
    )
