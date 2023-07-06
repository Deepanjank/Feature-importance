from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

diabetes = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(
    diabetes.data, diabetes.target, random_state=0)

model = Ridge(alpha=1e-2).fit(X_train, y_train)

for i in range(len(model.coef_)):
    print(f"{diabetes.feature_names[i]:<8}"
            f"{abs(model.coef_[i]):.3f}")