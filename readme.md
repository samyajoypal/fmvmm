# Clustering Compositional Data using Dirichlet Mixture Model

This article shows some basic functions available in the python package. The python package is based on the paper https://doi.org/10.1371/journal.pone.0268438

## Installation:

1. You must have python installed in your device.
2. Download the repository
3. To make sure you have all the necessary libraries required, please run the following command from root directory.    "pip install -r requirements.txt"

The follwing shows some examples using the python package.


```python
# Let us import some libraries
import numpy as np
import pandas as pd
```


```python
# Create some dummy data from four different Dirichlet Distributions of dimension 3


np.random.seed(0)
s1 = pd.DataFrame(np.random.dirichlet((12, 12, 4), 500))
s2 = pd.DataFrame(np.random.dirichlet((12, 25, 55), 100))
s3 = pd.DataFrame(np.random.dirichlet((20, 20, 20), 300))
s4 = pd.DataFrame(np.random.dirichlet((0.25, 0.55, 4), 400))
initial_clusters=sum([[0]*500,[1]*100,[2]*300,[3]*400],[])



s=pd.concat([s1,s2,s3,s4], ignore_index= True)
s_copy=s
s_copy['Initial_Clusters'] = initial_clusters


s_shuffled= s_copy.sample(frac=1)
s_shuffled1=[s_shuffled[0],s_shuffled[1],s_shuffled[2]]
s_shuffled1=pd.DataFrame(s_shuffled1)
s_shuffled1=s_shuffled1.transpose()
```

Now let us try to fit a Dirichlet Mixture Model to the data we have just created. There are different methods and initialization techniques available.

Possible choices of methods are "meanprecision" and "fixedpoint".

Possible choices of initialization are "KMeans", "GMM" and "random".


```python
# First import the DMM class

from mixtures.DMM_Class import DMM

#Let us initialize DMM with data and number of clusters along with default parameters.

model=DMM(number_of_clusters=4,sample= s_shuffled1, method="meanprecision", initialization="KMeans", tol=0.0001)

# To initialize the model with default parameters, it is sufficient just to supply number of clusters and the data.
# e.g. model= DMM(4,s_shuffled1)

model.fit() #To fit the model
```

    Model Fitting Done Successfully



```python
# Getting the values of estimated parameters

pi_hat, alpha_hat= model.get_params()
```

    The estimated pi values are  [0.3791510920699027, 0.305291232234975, 0.23724891767031334, 0.07830875802480888]
    The estimated alpha values are  [[14.426680889077614, 14.31890168542998, 4.453453467156408], [0.24233067680285156, 0.5479492716101892, 4.4378950314458905], [19.853177474165708, 19.98059620717043, 20.619761747567047], [10.894462473531632, 23.10056671037352, 51.236125582726714]]



```python
# Getting AIC and BIC of the model

print("AIC of the model is", model.aic())
print("BIC of the model is", model.bic())
```

    AIC of the model is 14.12176839212771
    BIC of the model is 91.67356154387213


If true labels of the data points are known, it is possible to get approximate measures such as accuracy, precision, recall and $F_1$ score.


```python
#import necessary functions

from utils.utils_mixture import acc_check, prec_check, rec_check, f_score

print("Accuracy of the model is", acc_check(s_shuffled['Initial_Clusters'], model.predict()))
print("Precision of the model is", prec_check(s_shuffled['Initial_Clusters'], model.predict()))
print("Recall of the model is", rec_check(s_shuffled['Initial_Clusters'], model.predict()))
print("F_1 score of the model is", f_score(s_shuffled['Initial_Clusters'], model.predict()))
```

    Accuracy of the model is 0.9446153846153846
    Precision of the model is 0.9512083333333333
    Recall of the model is 0.9286522071560359
    F_1 score of the model is 0.9397949466881848
