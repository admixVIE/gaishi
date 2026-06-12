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
import pytest

import gaishi.models
import gaishi.stats
from gaishi.train import train


@pytest.fixture
def file_paths():
    output_dir = "tests/test_train"
    os.makedirs(output_dir, exist_ok=True)

    return {
        "demes": "tests/data/ArchIE_3D19.yaml",
        "config": "tests/data/test.config.yaml",
        "output": os.path.join(output_dir, "test.lr.onnx"),
        "output_dir": output_dir,
    }


@pytest.fixture
def cleanup_output_dir(request, file_paths):
    yield
    shutil.rmtree(file_paths["output_dir"], ignore_errors=True)


def test_train(file_paths, cleanup_output_dir):
    train(
        demes=file_paths["demes"],
        config=file_paths["config"],
        output=file_paths["output"],
    )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        file_paths["output"],
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    metadata = session.get_modelmeta().custom_metadata_map

    assert json.loads(metadata["classes"]) == [0, 1]
    assert len(session.get_inputs()) == 1


def test_train_only_simulation(file_paths, cleanup_output_dir):
    train(
        demes=file_paths["demes"],
        config=file_paths["config"],
        output=file_paths["output"],
        only_simulation=True,
    )

    assert not os.path.exists(file_paths["output"])
