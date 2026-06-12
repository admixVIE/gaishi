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
from typing import Any

import numpy as np
import onnxruntime as ort

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def save_sklearn_classifier_as_onnx(
    model: Any,
    output: str,
    n_features: int,
) -> None:
    """
    Save a fitted scikit-learn classifier as an ONNX model.

    Parameters
    ----------
    model : Any
        Fitted scikit-learn classifier exposing a ``classes_`` attribute.
    output : str
        Path where the serialized ONNX model will be written.
    n_features : int
        Number of input features expected by the classifier.
    """
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
    )

    metadata = onnx_model.metadata_props.add()
    metadata.key = "classes"
    metadata.value = json.dumps([int(class_) for class_ in model.classes_])

    with open(output, "wb") as f:
        f.write(onnx_model.SerializeToString())


def predict_proba_with_onnx_classifier(
    model: str,
    data: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """
    Run probability inference with a saved ONNX classifier.

    Parameters
    ----------
    model : str
        Path to a saved ONNX classifier.
    data : numpy.ndarray
        Feature matrix for inference.

    Returns
    -------
    tuple[list[int], numpy.ndarray]
        Class labels and their corresponding probability matrix.
    """
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    metadata = session.get_modelmeta().custom_metadata_map
    classes = json.loads(metadata["classes"])

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: data.astype(np.float32, copy=False)})
    predictions = outputs[1]

    return classes, predictions
