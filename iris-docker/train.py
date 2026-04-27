from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2%}")
print("Model trained successfully inside docker!")

# ensure output dir and write results there
output_dir = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "results.txt")
with open(out_path, "w") as f:
    f.write(f"Accuracy: {acc:.2%}")
print(f"Wrote results to: {out_path}")