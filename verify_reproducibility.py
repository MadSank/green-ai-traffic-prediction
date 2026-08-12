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

TOLERANCE = 0.05


def verify_results(results_file="results/results_table.csv"):

    print("Reproducibility verification started")
    print("-" * 60)

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Run the main experiment script first")
        return 1

    try:
        results = pd.read_csv(results_file)
    except Exception as exc:
        print(f"Error reading results file: {exc}")
        return 1

    print(f"Loaded results from: {results_file}")
    print(f"Models found: {len(results)}")
    print()

    all_passed = True

    for model_name, expected in EXPECTED_RESULTS.items():

        row = results[results["Model"] == model_name]

        if row.empty:
            print(f"FAIL - {model_name}: missing from results")
            all_passed = False
            continue

        row = row.iloc[0]

        actual_acc = float(row["Accuracy"])
        actual_f1 = float(row["F1 Score"])

        acc_diff = abs(actual_acc - expected["Accuracy"])
        f1_diff = abs(actual_f1 - expected["F1 Score"])

        acc_ok = acc_diff <= TOLERANCE
        f1_ok = f1_diff <= TOLERANCE

        if acc_ok and f1_ok:
            print(f"PASS - {model_name}")
        else:
            print(f"FAIL - {model_name}")
            all_passed = False

        print(
            f"  Accuracy : {actual_acc:.4f} "
            f"(reported {expected['Accuracy']:.4f}, "
            f"diff {acc_diff:.4f})"
        )

        print(
            f"  F1 Score : {actual_f1:.4f} "
            f"(reported {expected['F1 Score']:.4f}, "
            f"diff {f1_diff:.4f})"
        )

        print()

    print("-" * 60)

    if all_passed:
        print("REPRODUCIBILITY CHECK PASSED")
        print("All results are within the accepted tolerance.")
        return 0

    print("REPRODUCIBILITY CHECK FAILED")
    print()
    print("Possible causes:")
    print("- Different library versions")
    print("- Different numerical backends")
    print("- Dataset differences")
    print("- Different random seed")
    print("- Hardware-dependent numerical variation")

    return 1


if __name__ == "__main__":
    print("Reproducibility Check")
    print("Energy-Efficient Machine Learning for Traffic Congestion Prediction")
    print()

    exit_code = verify_results()
    sys.exit(exit_code)
