"""
Loan Approval Classification
============================
Logistic Regression pipeline to predict loan approval status.

Structure
---------
1. Config            -> saari settings ek jagah
2. Animation helpers -> spinner, typing effect, progress bar (halka animation)
3. Data              -> load, inspect, prepare (clean + split)
4. Visualization     -> scatter, animated class distribution, animated confusion matrix
5. Model             -> preprocessing pipeline, train, predict
6. Evaluation        -> metrics + report
7. Save / Predict    -> model save, new applicant prediction
8. main()            -> poora flow

Run:
    python loan_approval_classification.py
    python loan_approval_classification.py --data loan_data.csv --no-anim
    python loan_approval_classification.py --no-plots
"""

import argparse
import itertools
import sys
import threading
import time
from contextlib import contextmanager

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 1. CONFIG
# =============================================================================
class Config:
    DATA_PATH = "loan_data.csv"
    MODEL_PATH = "loan_approval_model.pkl"
    TARGET = "loan_status"
    DROP_COLUMNS = ["person_age", "person_income", "credit_score"]
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    MAX_ITER = 1000
    ANIMATE = True   # terminal + plot animation on/off
    SHOW_PLOTS = True


# =============================================================================
# 2. ANIMATION HELPERS (light & optional)
# =============================================================================
def type_print(text, delay=0.008):
    """Typing effect for headings."""
    if not Config.ANIMATE:
        print(text)
        return
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def section(title):
    """Step heading with a small typing animation."""
    print()
    type_print("=" * 60)
    type_print(f"  {title}")
    type_print("=" * 60)


@contextmanager
def spinner(message, done_message=None):
    """Spinner jab tak koi kaam chal raha ho:  with spinner('Training'): ..."""
    if not Config.ANIMATE:
        print(f"{message}...")
        yield
        print(done_message or f"{message} - done")
        return

    stop = threading.Event()

    def spin():
        for frame in itertools.cycle("|/-\\"):
            if stop.is_set():
                break
            sys.stdout.write(f"\r{frame} {message}...")
            sys.stdout.flush()
            time.sleep(0.1)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        yield
        time.sleep(0.3)  # taaki spinner thoda dikhe
    finally:
        stop.set()
        t.join()
    sys.stdout.write(f"\r[OK] {done_message or message + ' - done'}      \n")
    sys.stdout.flush()


def progress_bar(label, value, width=30):
    """Metric ke liye filling bar (value 0 se 1 ke beech)."""
    steps = int(round(value * width))
    for i in range(steps + 1):
        bar = "#" * i + "-" * (width - i)
        shown = value * 100 if i == steps else i / width * 100
        end = "\n" if i == steps else ""
        sys.stdout.write(f"\r{label:<10} [{bar}] {shown:5.1f}%{end}")
        sys.stdout.flush()
        if Config.ANIMATE:
            time.sleep(0.015)


# =============================================================================
# 3. DATA
# =============================================================================
def load_data(path):
    with spinner("Dataset load ho raha hai", "Dataset Loaded Successfully"):
        df = pd.read_csv(path)
    print(df.head())
    return df


def inspect_data(df):
    section("Dataset Information")
    print("Shape (rows, columns):", df.shape)
    print("Column Names:", list(df.columns))
    print("\nData Types and Non-Null Counts:")
    df.info()
    print("\nStatistical Summary:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\nLoan Status Distribution:")
    print(df[Config.TARGET].value_counts())


def prepare_data(df):
    """Features/target alag karo, NaN target hatao, train-test split karo."""
    section("Prepare Data")
    X = df.drop(columns=Config.DROP_COLUMNS + [Config.TARGET])
    y = df[Config.TARGET]
    print("Features Shape:", X.shape)
    print("Target Shape:", y.shape)

    # Target mein NaN wali rows hata do
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y,
    )
    print("X_train Shape:", X_train.shape)
    print("X_test Shape :", X_test.shape)
    return X_train, X_test, y_train, y_test


# =============================================================================
# 4. VISUALIZATION
# =============================================================================
def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernel" in str(type(get_ipython()))
    except Exception:
        return False


def show_animation(fig, anim):
    """Script mein plt.show(), Colab/Jupyter mein inline HTML animation."""
    if _in_notebook():
        from IPython.display import HTML, display
        plt.close(fig)
        display(HTML(anim.to_jshtml()))
    else:
        plt.show()


def plot_income_vs_loan(df):
    """Scatter plot (extreme outliers ko hide karne ke liye 99th percentile tak)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    limit = df["person_income"].quantile(0.99)
    sns.scatterplot(
        data=df[df["person_income"] <= limit],
        x="person_income",
        y="loan_amnt",
        hue=Config.TARGET,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Person Income vs Loan Amount (up to 99th percentile)")
    ax.set_xlabel("Person Income")
    ax.set_ylabel("Loan Amount")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_status_distribution(df):
    """Bars neeche se upar dheere-dheere grow hote hain."""
    counts = df[Config.TARGET].value_counts().sort_index()
    labels = [str(int(i)) for i in counts.index]
    targets = counts.values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, np.zeros_like(targets), color=["#3274a1", "#e1812c"])
    ax.set_ylim(0, targets.max() * 1.1)
    ax.set_title("Loan Status Distribution")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Count")

    frames = 30

    def update(frame):
        progress = (frame + 1) / frames
        for bar, tgt in zip(bars, targets):
            bar.set_height(tgt * progress)
        return bars

    if Config.ANIMATE:
        anim = FuncAnimation(fig, update, frames=frames, interval=30, repeat=False)
        show_animation(fig, anim)
    else:
        update(frames - 1)
        plt.show()


def plot_confusion_matrix(cm):
    """Confusion matrix numbers 0 se final value tak count-up hote hain."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.zeros_like(cm), cmap="Blues", vmin=0, vmax=cm.max())
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")

    texts = [[ax.text(j, i, "0", ha="center", va="center") for j in range(2)] for i in range(2)]
    frames = 25

    def update(frame):
        progress = (frame + 1) / frames
        current = (cm * progress).astype(int) if frame < frames - 1 else cm
        im.set_data(current)
        for i in range(2):
            for j in range(2):
                color = "white" if current[i, j] > cm.max() * 0.5 else "black"
                texts[i][j].set_text(str(current[i, j]))
                texts[i][j].set_color(color)
        return [im]

    if Config.ANIMATE:
        anim = FuncAnimation(fig, update, frames=frames, interval=40, repeat=False)
        show_animation(fig, anim)
    else:
        update(frames - 1)
        plt.show()


# =============================================================================
# 5. MODEL
# =============================================================================
def build_model(X_train):
    """Preprocessing (impute + scale + one-hot) + Logistic Regression ek pipeline mein."""
    numerical_features = X_train.select_dtypes(include=np.number).columns
    categorical_features = X_train.select_dtypes(exclude=np.number).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=Config.MAX_ITER)),
    ])


def train_model(model, X_train, y_train):
    with spinner("Model train ho raha hai", "Model Trained Successfully"):
        model.fit(X_train, y_train)
    return model


# =============================================================================
# 6. EVALUATION
# =============================================================================
def evaluate_model(model, X_test, y_test):
    section("Model Evaluation")
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    for name, value in metrics.items():
        progress_bar(name, value)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    if Config.SHOW_PLOTS:
        plot_confusion_matrix(cm)
    return metrics


# =============================================================================
# 7. SAVE / PREDICT
# =============================================================================
def save_model(model, path):
    with spinner("Model save ho raha hai", "Model Saved Successfully!"):
        joblib.dump(model, path)


def predict_applicant(model, applicant):
    """applicant: single-row DataFrame (same columns jo X mein the)."""
    pred = model.predict(applicant)[0]
    proba = model.predict_proba(applicant)[0]
    print("\nApplicant Details:")
    print(applicant.to_string())
    print(f"\nPredicted Loan Status : {pred}")
    print(f"Probability [class 0, class 1]: {np.round(proba, 4)}")
    return pred, proba


# =============================================================================
# 8. MAIN
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Loan Approval Classification")
    p.add_argument("--data", default=Config.DATA_PATH, help="CSV file ka path")
    p.add_argument("--no-anim", action="store_true", help="animation band karo")
    p.add_argument("--no-plots", action="store_true", help="plots band karo")
    # parse_known_args taaki Colab/Jupyter mein bhi chale
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    Config.ANIMATE = not args.no_anim
    Config.SHOW_PLOTS = not args.no_plots

    section("Loan Approval Classification")
    df = load_data(args.data)
    inspect_data(df)

    if Config.SHOW_PLOTS:
        section("Data Visualization")
        plot_income_vs_loan(df)
        plot_status_distribution(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    section("Model Training")
    model = build_model(X_train)
    train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    section("New Loan Prediction")
    predict_applicant(model, X_test.iloc[[0]])

    section("Save Model")
    save_model(model, Config.MODEL_PATH)


if __name__ == "__main__":
    main()
