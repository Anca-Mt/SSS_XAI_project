# Explainable-AI-pipeline
### Adapted by Anca Matei (s4004507) and Bart Remmelzwaal (s2643553)

XAI pipeline for explaining tabular data such as blobs and network traffic (supporting repo for the XAI SoK at `Euro S&P '23`)

This is a modular XAI pipeline in Python (since it has in-built support for many popular models and explainers). The pipeline has three components: 

(1) The parser parses the input data (train and test) in either CSV or NumPy array format. The user can specify to the parser which feature fields should be read by means of providing a configuration file for the parser 

(2) The classifiers are implemented as a wrapper over the ML algorithms provided by scikit-learn. We currently support `decision trees`, `logistic regression`, `explainable boosting machines`, `random forests`, `gradient boosting machines`, and `SVM`s. The wrapper specifies the ML algorithm and its hyperparameters.

(3) Similarly, the explainers are also implemented as wrapper functions and currently provide support for `SHAP`, `LIME`, `EBM`, and `ELI5`. The modules can be extended for added support of custom parsers, models and explainers. For the sake of reproducibility, the pipeline saves the model, predictions and explanations in a file.


Install the required python packages using:
```
pip install -r requirements.txt
```

Currently supported classifiers:

<b>randomforest, decisiontree, svm, ebmclassifier</b>

<u>NB: the EBM explainer only works with the EBMClassifier, and is incompatible with ELI5.</u>

Currently supported datasets: 

<b>CTU-13, NSL-KDD, UNSW-NB15</b>

The raw datasets can be found in ```datasets_raw/```. The datasets have already been preprocessed into the CSV format (```datasets_csv/```) and NumPy format (```datasets_npy/```). If you decide to build these datasets from scratch, preprocessing scripts are provided in the project directory under the name ```preprocessing_dataset_<DATASET>.py```. Since CTU-13 consists of multiple different scenarios, ```combine_all_CTU-13.py``` merges all ```.parquet``` files in the ```datasets_raw/CTU-13/``` directory into one ```.parquet``` file ready for preprocessing.

Example run: 
- ``` python main.py npy randomforest lime CTU-13 ```

Available arguments:
```
--parser            The argparser as a string. (default="npy")
--classifier        The classifier as a string. (default="ebmclassifier")
--explainer         The explainer as a string. 'all' for all explainers. (default="all")
-d --dataset        Dataset to explain. Mandatory. (default="CTU-13")
--load-classifier   Path to a pickled classifier file. If provided, this classifier will be loaded rather than a new one trained. (default=None)
--output-path       Output dir for this experiment. Default is the results directory. (default="results")
```


Add modules: 
The software is a three-stage pipeline with first a parser, then a classifier, and in the end an explainer. When adding a new module, please follow
the structure of the existing directories. The same goes for potential desired .ini files. The parser .py for example should be in src/parsers/parsers/. Then, add the parser to the factory.py module in the parser subdir dir. This way, all the module implementations are hidden away from the main module. 

One sidenote: The current parsers seem simple, and probably do not have to implemented that way. I chose to do that in case some extra implementation
to them is desired, say an automatic scaling that should take place at this stage. 

If you use the XAI pipeline in a scientific work, consider citing the following paper:

```
@inproceedings{nadeem2023sok,
  title={SoK: Explainable Machine Learning for Computer Security Applications},
  author={Nadeem, Azqa and Vos, Dani{\"e}l and Cao, Clinton and Pajola, Luca and Dieck, Simon and Baumgartner, Robert and Verwer, Sicco},
  booktitle={In proceedings of EuroS\&P},
  publisher={IEEE},
  year={2023}
}
```
