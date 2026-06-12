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

import inspect
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from gaishi.models import MlModel
from gaishi.models.onnx_utils import predict_proba_with_onnx_classifier
from gaishi.models.onnx_utils import save_sklearn_classifier_as_onnx
from gaishi.registries.model_registry import MODEL_REGISTRY

pd.options.mode.chained_assignment = None


@MODEL_REGISTRY.register("extra_trees_classifier")
class EtcModel(MlModel):
    """
    An extra-trees classifier for training and inferring purposes,
    specifically designed for datasets in genomics or biological contexts.
    This class provides static methods to train an extra-trees classifier
    and to perform inference with an existing model.
    """

    @staticmethod
    def train(
        data: str,
        output: str,
        **model_params,
    ) -> None:
        """
        Train an extra-trees classifier on tabular feature data.

        The input feature table is read from a tab-separated file, a model is
        fitted, and the trained model is written to disk. Any keyword arguments
        in `model_params` are forwarded to the underlying
        :class:`sklearn.ensemble.ExtraTreesClassifier`.
        Parameters
        ----------
        data : str
            Path to the training data file in tab-separated format. The table
            is expected to contain a `Label` column and any required metadata
            columns, which will be dropped before model fitting.
        output : str
            Path where the trained ONNX model will be saved.
        **model_params
            Additional keyword arguments controlling the model and optional
            preprocessing. These are passed to the underlying extra-trees
            classifier (for example `n_estimators`, `max_depth`, `random_state`)
        """
        features = pd.read_csv(data, sep="\t")

        if "Label" not in features.columns:
            raise ValueError(
                "Training data must contain a 'Label' column. Please check your input file."
            )
        labels = features["Label"]
        data = features.drop(
            columns=["Chromosome", "Start", "End", "Sample", "Replicate", "Label"]
        ).values

        is_allowed = inspect.signature(ExtraTreesClassifier).parameters
        clean_params = {k: v for k, v in model_params.items() if k in is_allowed}

        model = ExtraTreesClassifier(**clean_params)
        model.fit(data, labels.astype(int))

        save_sklearn_classifier_as_onnx(
            model=model,
            output=output,
            n_features=data.shape[1],
        )

    @staticmethod
    def infer(
        data: str,
        model: str,
        output: str,
        **model_params,
    ) -> None:
        """
        Perform inference with a trained extra-trees classifier on new data.

        The feature table is read from a tab-separated file, the trained model is
        loaded with ONNX Runtime, and predictions are written to the specified output.
        Any keyword arguments in `model_params` are accepted for API compatibility
        and ignored during inference.

        Parameters
        ----------
        data : str
            Path to the inference data file in tab-separated format. The table
            is expected to contain the same feature columns as the training
            data, along with any metadata columns that will be dropped before
            prediction.
        model : str
            Path to the saved trained ONNX model.
        output : str
            Path where the inference output (e.g. predicted labels or
            probabilities) will be written.
        **model_params
            Additional keyword arguments accepted for API compatibility.
        """
        features = pd.read_csv(data, sep="\t")
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        data = features.drop(columns=["Chromosome", "Start", "End", "Sample"]).to_numpy(
            dtype=np.float32
        )

        classes, predictions = predict_proba_with_onnx_classifier(
            model=model,
            data=data,
        )
        prediction_df = features[["Chromosome", "Start", "End", "Sample"]]

        class_names = {
            "0": "Non_Intro",
            "1": "Intro",
        }

        for i in range(len(classes)):
            class_name = class_names[f"{classes[i]}"]
            prediction_df[f"{class_name}_Prob"] = predictions[:, i]

        prediction_df.sort_values(by=["Sample", "Chromosome", "Start", "End"]).to_csv(
            output, sep="\t", index=False
        )
