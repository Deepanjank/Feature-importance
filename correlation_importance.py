from sklearn.datasets import load_diabetes
from numpy import cov
from numpy import transpose

diabetes = load_diabetes()
for i in range(len(transpose(diabetes.data))):
    covariance = cov(transpose(diabetes.data)[i], diabetes.target)
    print(f"{diabetes.feature_names[i]:<8}"
            f"{abs(covariance[0][1]):.3f}")