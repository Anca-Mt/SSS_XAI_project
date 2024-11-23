"""
Main script, taking in input parameters and starting subroutines.
"""

import os
import sys

workdir = os.path.dirname(os.path.realpath(__file__))

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

dataset_npy_dir = "datasets_npy"

import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass
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

def test_filepaths(args):
  """[Tests the filepaths set by arguments]

  Args:
      args ([args]): [The args as returned by argumentparser]

  Raises:
      Exception: [Path to inputfile not provided or wrong]
  """
  X_train_f = args.xtrain
  if not os.path.isfile(X_train_f):
    raise Exception("Invalid path to xtrain-file: {}".format(X_train_f))

  y_train_f = args.ytrain
  if not os.path.isfile(y_train_f):
    raise Exception("Invalid path to ytrain-file: {}".format(y_train_f))

  X_test_f = args.xtest
  if X_test_f and not os.path.isfile(X_test_f):
    raise Exception("Invalid path to xtest-file: {}".format(X_test_f))

  y_test_f = args.ytest
  if y_test_f and not os.path.isfile(y_test_f):
    raise Exception("Invalid path to ytest-file: {}".format(y_test_f))

  X_explain_f = args.xexplain
  if not os.path.isfile(X_explain_f):
    raise Exception("Invalid path to xexplain-file: {}".format(X_explain_f))

  y_explain_f = args.yexplain
  if not os.path.isfile(y_explain_f):
    raise Exception("Invalid path to yexplain-file: {}".format(y_explain_f))

  inifile = args.ini
  if inifile and not os.path.isfile(inifile):
    raise Exception("Invalid path to ini-file: {}".format(inifile))


"""
For Bart & me: 
- parser strings: npy, csv
- classifiers strings: random forest, decisiontree, svm, ebmclassifier
- explainer strings:  lime, shap, eli5, ebm

These were found in the factory files. 
"""

"""
For either one of us:
- Should be consistent in the future about the dataset folder and filename structure
- If anything is to change, make sure we do it consistently so we have a format to work with and implement more datasets/classifiers
- Idea for the explainer .ini files: here in main, look at what dataset we will be using, and have for instance
shapexplainer_CTU-13.ini, rename it to shapexlainer.ini so we automatically use the right column names

- Main changes to running files:
  * create_subset.py now requires a -d or --dataset parameter to select the correct dataset (currently only 1)
  * main.py no longer requires giving xtrain, ytrain, xtest etc... just the name of the dataset!
  * valid datasets in the files where you have to pick are listed in the array valid_datasets near the top

TODO: create --explainer all to run all explainers on the given dataset
"""


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(description='The pipeline for the explainability experiments.')

  argparser.add_argument('--parser', default="npy", type=str, help='The argparser as a string')
  argparser.add_argument('--classifier', default="decisiontree", type=str, help='The classifier as a string')
  argparser.add_argument('--explainer', default="shap", type=str, help='The explainer as a string. \'all\' for all explainers')

  argparser.add_argument('-d', '--dataset', default="NSL-KDD", type=str,  help='Dataset to explain. Mandatory.')

#   argparser.add_argument('--xtrain', default="CTU-13_npys/Scenario 1/X_train.npy", type=str,  help='Path to X-train. Mandatory.')
#   argparser.add_argument('--ytrain', default="CTU-13_npys/Scenario 1/Y_train.npy", type=str, help='Path to y-train. Mandatory.')

#   argparser.add_argument('--xtest', default="CTU-13_npys/Scenario 1/X_test.npy", type=str, help='Path to X-test. Not mandatory, but some classifiers will need this for training.')
#   argparser.add_argument('--ytest', default="CTU-13_npys/Scenario 1/Y_test.npy", type=str, help='Path to y-test. Not mandatory, but some classifiers will need this for training.')
  
#   argparser.add_argument('--xexplain', default="CTU-13_npys/Scenario 1/X_explain.npy", type=str, help='Path to X-explain. Mandatory.')
#   argparser.add_argument('--yexplain', default="CTU-13_npys/Scenario 1/Y_explain.npy", type=str, help='Path to y-explain. Mandatory.')

  argparser.add_argument('--ini', type=str, default=None, help='The explainer as a string')
  argparser.add_argument('--load-classifier', type=str, default=None, help='Path to a pickled classifier file. If provided, this classifier will be loaded rather than a new one trained.')
  argparser.add_argument('--output-path', type=str, default="results", help='Output dir for this experiment. Default is the script\'s directory.')

  args = argparser.parse_args()

  #test_filepaths(args)

  X_train_f, y_train_f, X_test_f, y_test_f, X_explain_f, y_explain_f = test_dataset(args.dataset)

  inifile = args.ini

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

  # output_path = args.output_path
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
