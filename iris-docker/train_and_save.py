import time
import psycopg2
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def wait_for_db(retries=10, delay=3):
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(
                host="db",   # ← the service name, not localhost!
                port=5432,
                user="iris_user",
                password="iris_pass",
                dbname="iris_db"
            )
            conn.close()
            print(f"[db] Connected on attempt {attempt}")
            return
        except psycopg2.OperationalError:
            print(f"[db] Not ready ({attempt}/{retries}), waiting...")
            time.sleep(delay)
    raise RuntimeError("Could not connect to PostgreSQL.")

wait_for_db()

conn = psycopg2.connect(
    host="db", port=5432, user="iris_user",
    password="iris_pass", dbname="iris_db"
)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS training_runs (
        id        SERIAL PRIMARY KEY,
        run_at    TIMESTAMP DEFAULT NOW(),
        accuracy  NUMERIC(5,4),
        n_train   INT,
        n_test    INT,
        max_depth INT
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id              SERIAL PRIMARY KEY,
        run_id          INT REFERENCES training_runs(id),
        sepal_length    NUMERIC(4,2),
        sepal_width     NUMERIC(4,2),
        petal_length    NUMERIC(4,2),
        petal_width     NUMERIC(4,2),
        actual_label    TEXT,
        predicted_label TEXT,
        correct         BOOLEAN
    );
""")
conn.commit()
print("[db] Tables ready.")

iris = load_iris()
X, y = iris.data, iris.target
labels = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"[model] Accuracy: {accuracy:.2%}")

# Save the training run, get back its id
cur.execute("""
    INSERT INTO training_runs (accuracy, n_train, n_test, max_depth)
    VALUES (%s, %s, %s, %s) RETURNING id;
""", (float(accuracy), len(X_train), len(X_test), 3))
run_id = cur.fetchone()[0]

# Save every prediction row
rows = [(run_id, float(f[0]), float(f[1]), float(f[2]), float(f[3]),
         labels[a], labels[p], bool(a == p))
        for f, a, p in zip(X_test, y_test, preds)]

cur.executemany("""
    INSERT INTO predictions
      (run_id, sepal_length, sepal_width, petal_length, petal_width,
       actual_label, predicted_label, correct)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
""", rows)
conn.commit()
print(f"[db] Saved run #{run_id} with {len(rows)} predictions.")
cur.close(); conn.close()
print("[done] All results in PostgreSQL.")