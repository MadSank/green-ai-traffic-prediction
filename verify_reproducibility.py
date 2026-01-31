import pandas as pd
import sys
import os

EXPECTED_RESULTS = {
    "Logistic Regression": {"Accuracy": 0.9803, "F1 Score": 0.9063},
    "Decision Tree": {"Accuracy": 0.9831, "F1 Score": 0.9201},
    "Naive Bayes": {"Accuracy": 0.9864, "F1 Score": 0.9345},
    "Random Forest": {"Accuracy": 0.9837, "F1 Score": 0.9200},
    "Gradient Boosting": {"Accuracy": 0.9841, "F1 Score": 0.9232},
    "KNN": {"Accuracy": 0.9831, "F1 Score": 0.9183},
}

TOLERANCE = 0.002


def verify_results(results_file="results/results_table.csv"):
    """
    Compare generated results against values reported in the paper.
    """

    print("Reproducibility verification started")

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Run the main experiment script first")
        return 1

    try:
        results = pd.read_csv(results_file)
    except Exception as exc:
        print(f"Error reading results file: {exc}")
        return 1

    print(f"Loaded results from {results_file}")
    print(f"Models found: {len(results)}")

    all_passed = True

    for model_name, expected in EXPECTED_RESULTS.items():
        row = results[results["Model"] == model_name]

        if row.empty:
            print(f"{model_name}: missing from results file")
            all_passed = False
            continue

        row = row.iloc[0]

        acc_diff = abs(row["Accuracy"] - expected["Accuracy"])
        f1_diff = abs(row["F1 Score"] - expected["F1 Score"])

        acc_ok = acc_diff <= TOLERANCE
        f1_ok = f1_diff <= TOLERANCE

        status = "PASS" if acc_ok and f1_ok else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"{status} - {model_name}")
        print(
            f"Accuracy: {row['Accuracy']:.4f} "
            f"(expected {expected['Accuracy']:.4f}, diff {acc_diff:.4f})"
        )
        print(
            f"F1 Score: {row['F1 Score']:.4f} "
            f"(expected {expected['F1 Score']:.4f}, diff {f1_diff:.4f})"
        )

    if all_passed:
        print("All results match reported values within tolerance")
        return 0

    print("Some results differ from expected values")
    print("Possible causes:")
    print("Library version differences")
    print("Random seed mismatch")
    print("Dataset corruption or modification")
    print("Minor numerical variation across systems")
    return 1


if __name__ == "__main__":
    print("Reproducibility check")
    print("Energy-Efficient Machine Learning for Traffic Congestion Prediction")

    exit_code = verify_results()
    sys.exit(exit_code)
