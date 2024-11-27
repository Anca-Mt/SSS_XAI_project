"""
Explainer to explainable boosting machine. WARNING: Only works with ExplainableBoostingRegressor() from interpret package.

EBM: https://interpret.ml/docs/ebm.html
"""

import configparser
import pickle
import os

from src.explainers.explainerbase import ExplainerBase

class EBMExplainer(ExplainerBase):
  def __init__(self):
    super().__init__()

  def explain(self, classifier, X, y, dataset_ini):
    """[Runs a lime explanation, and saves results as object attribute.]

    Args:
        classifier ([ClassifierBase]): [The used classifier]
        X ([np.array]): [X]
        y ([np.array]): [y]

    Raises:
        Exception: [.ini file does not exist]
    """
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(scriptdir, f"../parameters/{dataset_ini}/explainableboostingexplainer.ini")
    if not os.path.isfile(os.path.join(filedir)):
      raise Exception("Problem reading explainableboostingexplainer.ini file in EBMExplainer. Is the file existent?")

    config = configparser.ConfigParser()
    config.read(filedir)

    feature_names = config.get("DEFAULT", "feature_names").split()

    self.global_explanation = classifier.explain_global()
    self.local_explanations = classifier.explain_local(X, y)

    if self.global_explanation is None:
      raise ValueError("Global explanation is None.")

    if self.local_explanations is None:
      raise ValueError("Local explanation is None.")

    global_feature_names = self.global_explanation.data()["names"]

    mapped_names = []
    for name in global_feature_names:
      if "x" in name:
        parts = name.split(" x ")
        mapped_name = " x ".join([feature_names[int(part.split("_")[-1]) - 1] for part in parts])
        mapped_names.append(mapped_name)
      else:

        mapped_name = feature_names[int(name.split("_")[-1]) - 1]
        mapped_names.append(mapped_name)

    self.global_explanation.data()["names"] = mapped_names
    for instance_idx in range(len(X)):
      self.local_explanations.data(instance_idx)["names"] = mapped_names

  def save_results(self, output_path):
    pickle.dump(self.global_explanation, open(os.path.join(output_path, "ebm_global_exps.pk"), "wb"))
    pickle.dump(self.local_explanations, open(os.path.join(output_path, "ebm_local_exps.pk"), "wb"))