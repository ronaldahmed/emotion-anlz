import argparse
import numpy as np
import scipy as sp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from time import time
from scipy.stats import randint as sp_randint
from reader import Reader
from feature_factory import FeatureFactory
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

import warnings
import pdb

np.random.seed(42)
warnings.filterwarnings("ignore")

class log_uniform():        
  def __init__(self, a=-1, b=0, base=10):
    self.loc = a
    self.scale = b - a
    self.base = base

  def rvs(self, size=1, random_state=None):
    uniform = sp.stats.uniform(loc=self.loc, scale=self.scale)
    return np.power(self.base, uniform.rvs(size=size, random_state=random_state))


class hidden_layer_distr():        
  def __init__(self, nh=1, _min=10, _max=100):
    self.nlayers = nh
    self._min = _min
    self._max = _max

  def rvs(self, size=1, random_state=None):
    rint_list = [sp_randint(self._min,self._max) for _ in range(self.nlayers)]
    return tuple([rint.rvs(size=size,random_state=random_state) for rint in rint_list])

  


# Utility function to report best scores
def report(results, n_top=3):
  for i in range(1, n_top + 1):
    candidates = np.flatnonzero(results['rank_test_score'] == i)
    for candidate in candidates:
      print("Model with rank: {0}".format(i))
      print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            results['mean_test_score'][candidate],
            results['std_test_score'][candidate]))
      print("Parameters: {0}".format(results['params'][candidate]))
      print("")


def tune(model,params,x,y,n_iter_search=20,njobs=1):
  
  random_search = RandomizedSearchCV(model, param_distributions=params,
                                   n_iter=n_iter_search, cv=5, iid=False,n_jobs=njobs)
  start = time()
  random_search.fit(x,y)
  print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
  report(random_search.cv_results_)
  return random_search.best_params_




if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument("--clsf", help="classifier name [logreg,svm,rf]", type=str, default="linear")
  p.add_argument("--tune", help="Tune hyper-params using random search - cv", action='store_true')
  p.add_argument("--njobs", help="Number of jobs / cpus", type=int, default=4)

  args = p.parse_args()

  print("Loading SSEC corpus...")
  reader = Reader("all_ssec.tsv")

  print("Setting up FeatureFactory...")
  featfact = FeatureFactory()
  X_train = featfact.fit_transform(reader.train_txt)
  X_test  = featfact.transform(reader.test_txt)
  Y_train,Y_test = reader.train_labels, reader.test_labels
  
  print("Train shape:",X_train.shape)
  print("Test shape:",X_test.shape)  

  pdb.set_trace()

  # if   args.clsf == "logreg":
  #   model = LogisticRegression()
  #   params = {"C": log_uniform(-5,1)}
  

  model = ''
  def_param = {}
  if args.clsf == "rf":
    model = RandomForestClassifier()
    params = {
      "n_estimators": sp_randint(10, 100),
      "criterion": ["gini","entropy"],
      "max_depth" : sp_randint(10, 100)
    }
    def_param = {
      "n_estimators": 10,
      "criterion": "gini",
      "max_depth" : 10
    }
  
  elif args.clsf == "knn":
    model = KNeighborsClassifier()
    params = {
      "n_neighbors": sp_randint(5, 20),
      "weights": ["uniform","distance"],
    }
    def_param = {
      "n_neighbors": 5,
      "weights": "uniform",
    }

  elif args.clsf == "mlp":
    model = MLPClassifier(max_iter=200)
    params = {
      "batch_size": sp_randint(20, 200),
      "learning_rate_init": log_uniform(-5,-1),
      "alpha": log_uniform(-5,0),
      "activation": ["tanh","relu"],
      "hidden_layer_sizes": [(100,),(50,),(150,),(10,),
                            (100,100),(50,50),(10,10),
                            (100,50),(50,100),
                            (50,50,50),(10,10,10),],
      #"hidden_layer_sizes": hidden_layer_distr(2,10,100),
    }
    def_param = {
      "batch_size": 158,
      "learning_rate_init": 7.56e-5,
      "alpha": 5.99e-5,
      "activation": "relu",
      "hidden_layer_sizes": (100,50),
    }

  #

  if args.tune:
    print("Tuning hyperparameters ...")
    best_params = tune(model,params,X_train,Y_train,
                       n_iter_search=100,
                       njobs=args.njobs)
  else:
    best_params = def_param

  print("="*100)
  print("Training with optimal hyperparam set...")
  print(best_params)

  model.set_params(**best_params)
  model.fit(X_train,Y_train)
  pred_test = model.predict(X_test)

  # print( classification_report(Y_test,pred_test,target_names=reader.get_label_names(),digits=4) )
  print( classification_report(Y_train,model.predict(X_train),target_names=reader.get_label_names(),digits=4) )
