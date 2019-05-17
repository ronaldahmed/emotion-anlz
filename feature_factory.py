from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import pdb

class FeatureFactory:
  def __init__(self,_n_comps=500):
    self.stopwords = stopwords.words('english')

    vectorizer = TfidfVectorizer(stop_words=self.stopwords)
    dim_red = TruncatedSVD(n_components=_n_comps,random_state=42)

    self.pipeline = Pipeline(steps=[("vectorizer",vectorizer),
                                    ("svd",dim_red)])
    
  def fit(self,text):
    self.pipeline.fit(text)

  def fit_transform(self,text):
    return self.pipeline.fit_transform(text)

  def transform(self,text):
    return self.pipeline.transform(text)
    

  