"""
Main script, taking in input parameters and starting subroutines.
"""

import os
import sys
import contextlib
import sys

import argparse
import pickle

import src.classifiers
import src.explainers
import src.parsers

from src.parsers.factory import ParserFactory
from src.classifiers.factory import ClassifierFactory
from src.explainers.factory import ExplainerFactory

valid_datasets = [
    "CTU-13",
    "NSL-KDD",
    "UNSW-NB15",
]

workdir = os.path.dirname(os.path.realpath(__file__))
dataset_npy_dir = "datasets_npy"


# Pipe output to nowhere
class DummyFile(object):
    def write(self, _): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def test_dataset(dataset_name):
  """[Tests datset to see if all files are present.]
  
  Args:
      dataset_name ([str]): [The dataset name to explain]
  
  Raises:
      Exception: [Path to dataset not provided, wrong, or files are missing]
  
  Returns:
      `['X_train.npy', 'Y_train.npy', 'X_test.npy', 'Y_test.npy', 'X_explain.npy', 'Y_explain.npy']`: [Iff all files exist]
  """

  filenames = ['_'.join([prefix, postfix]) + '.npy' for postfix in ['train', 'test', 'explain'] for prefix in ['X', 'Y']]
  filepaths = [os.path.join('datasets_npy', dataset_name, file) for file in filenames]

  for file in filepaths:
    if not os.path.isfile(file):
      raise Exception(f"Could not find path to {file}.npy.")
  
  return filepaths


"""
For Bart & me: 
- parser strings: npy, csv
- classifiers strings: random forest, decisiontree, svm, ebmclassifier
- explainer strings:  lime, shap, eli5, ebm

These were found in the factory files. 
"""


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(description='The pipeline for the explainability experiments.')

  argparser.add_argument('--parser', default="npy", type=str, help='The argparser as a string.')
  argparser.add_argument('--classifier', default="ebm", type=str, help='The classifier as a string.')
  argparser.add_argument('--explainer', default="all", type=str, help='The explainer as a string. \'all\' for all explainers.')

  argparser.add_argument('-d', '--dataset', default="CTU-13", type=str,  help='Dataset to explain. Mandatory.')

  argparser.add_argument('--load-classifier', type=str, default=None, help='Path to a pickled classifier file. If provided, this classifier will be loaded rather than a new one trained.')
  argparser.add_argument('--output-path', type=str, default="results", help='Output dir for this experiment. Default is the results directory.')

  args = argparser.parse_args()

  X_train_f, y_train_f, X_test_f, y_test_f, X_explain_f, y_explain_f = test_dataset(args.dataset)

  pfactory = ParserFactory()
  fileparser = pfactory.create_parser(args.parser)

  print("Parsing the input files... ", end='', flush=True)
  X_train = fileparser.parse(X_train_f)
  y_train = fileparser.parse(y_train_f)

  if X_test_f and y_test_f:
    X_test = fileparser.parse(X_test_f)
    y_test = fileparser.parse(y_test_f)

  X_explain = fileparser.parse(X_explain_f)
  y_explain = fileparser.parse(y_explain_f)
  print("Finished parsing the input files.")

  output_path = os.path.join(args.output_path, args.dataset)
  if not os.path.isdir(output_path):
     os.makedirs(output_path, exist_ok=True)
    
  load_classifier = args.load_classifier
  if load_classifier:
    if not os.path.isfile(load_classifier):
      raise Exception("Invalid path to classifier provided: {}".format(load_classifier))
    classifier = pickle.load(open(load_classifier, "rb"))
  else:
    cfactory = ClassifierFactory()
    classifier = cfactory.create_classifier(args.classifier)
    
    print("Starting the training of the classifier... ", end='')
    classifier.fit(X_train, y_train)
    classifier.print_wrong_predictions(X_explain, y_explain, output_path)
    
    pickle.dump(classifier, open(os.path.join(output_path, "classifier.pk"), "wb"))
    print("Finished training the classifier.")
  
  explainers = [args.explainer]
  if args.explainer == 'all':
    print("Running all explainers...")
    explainers = ['shap', 'lime', 'eli5', 'ebm']

  for exp in explainers:
    print(f"Starting the explanation step for {exp}... ", end='', flush=True)
    try:
      efactory = ExplainerFactory()
      
      with nostdout():
        explainer = efactory.create_explainer(exp)
        explainer.explain(classifier=classifier, X=X_explain, y=y_explain, dataset_ini=args.dataset)

      output_path = os.path.join(args.output_path, args.dataset, exp)
      if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        
      print(f"Finished explanations.\nSaving results to {output_path}... ", end='', flush=True)
      explainer.save_results(output_path)
      print("Finished saving results.")
      
    except Exception as e:
      print(f"An error occurred while trying to run {exp}: {e}. Continuing...")

print("All tasks finished.")
