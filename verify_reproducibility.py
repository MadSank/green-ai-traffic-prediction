"""
Reproducibility Verification Script

This script verifies that your results match the expected values from the paper:
"Energy-Efficient Machine Learning for Traffic Congestion Prediction"

Run this after running traffic_prediction.py to verify reproducibility.
"""

import pandas as pd
import sys
import os

# Expected values from Table I in the paper
EXPECTED_RESULTS = {
    'Logistic Regression': {
        'Accuracy': 0.9803,
        'F1 Score': 0.9063
    },
    'Decision Tree': {
        'Accuracy': 0.9831,
        'F1 Score': 0.9201
    },
    'Naive Bayes': {
        'Accuracy': 0.9864,
        'F1 Score': 0.9345
    },
    'Random Forest': {
        'Accuracy': 0.9837,
        'F1 Score': 0.9200
    },
    'Gradient Boosting': {
        'Accuracy': 0.9841,
        'F1 Score': 0.9232
    },
    'KNN': {
        'Accuracy': 0.9831,
        'F1 Score': 0.9183
    }
}

# Acceptable tolerance (±0.2%)
TOLERANCE = 0.002

def verify_results(results_file='results/results_table.csv'):
    """
    Verify that generated results match expected values from the paper.
    
    Args:
        results_file: Path to the results CSV file
        
    Returns:
        0 if all results verified successfully, 1 otherwise
    """
    
    print("\n" + "=" * 70)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 70)
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"\n❌ ERROR: Results file not found: {results_file}")
        print("\nPlease run the main script first:")
        print("  python traffic_prediction.py")
        print("\nThis will generate the results file needed for verification.")
        return 1
    
    # Load results
    try:
        results = pd.read_csv(results_file)
    except Exception as e:
        print(f"\n❌ ERROR: Could not read results file: {e}")
        return 1
    
    print(f"\n✓ Results file loaded: {results_file}")
    print(f"  Found {len(results)} models\n")
    print("-" * 70)
    
    all_passed = True
    
    # Verify each model
    for model_name, expected in EXPECTED_RESULTS.items():
        # Find model in results
        model_rows = results[results['Model'] == model_name]
        
        if len(model_rows) == 0:
            print(f"❌ {model_name}: NOT FOUND in results file")
            all_passed = False
            continue
        
        row = model_rows.iloc[0]
        
        # Calculate differences
        acc_diff = abs(row['Accuracy'] - expected['Accuracy'])
        f1_diff = abs(row['F1 Score'] - expected['F1 Score'])
        
        # Check if within tolerance
        acc_passed = acc_diff <= TOLERANCE
        f1_passed = f1_diff <= TOLERANCE
        
        # Determine status
        if acc_passed and f1_passed:
            status = '✓'
        else:
            status = '✗'
            all_passed = False
        
        # Print results
        print(f"{status} {model_name}:")
        print(f"    Accuracy: {row['Accuracy']:.4f} " +
              f"(expected: {expected['Accuracy']:.4f}, " +
              f"diff: {acc_diff:.4f})")
        print(f"    F1-Score: {row['F1 Score']:.4f} " +
              f"(expected: {expected['F1 Score']:.4f}, " +
              f"diff: {f1_diff:.4f})")
        
        if not (acc_passed and f1_passed):
            print(f"    ⚠️  Difference exceeds tolerance (±{TOLERANCE})")
        
        print()
    
    # Print final verdict
    print("=" * 70)
    if all_passed:
        print("✓✓✓ ALL RESULTS VERIFIED SUCCESSFULLY! ✓✓✓")
        print("=" * 70)
        print("\nYour implementation perfectly reproduces the paper results!")
        print("Results match within ±0.2% tolerance.")
        return 0
    else:
        print("✗ SOME RESULTS DIFFER FROM EXPECTED VALUES")
        print("=" * 70)
        print("\nPossible causes:")
        print("1. Different library versions")
        print("   → Check: pip list | grep -E 'scikit-learn|numpy|pandas'")
        print("   → Expected: scikit-learn==1.3.0, numpy==1.24.3, pandas==2.0.3")
        print("\n2. Random seed not set correctly")
        print("   → Verify: np.random.seed(42) is at the start of the script")
        print("\n3. Dataset corruption")
        print("   → Re-download metr-la.h5 from Kaggle")
        print("   → Verify shape: (34272, 207)")
        print("\n4. Hardware/system differences")
        print("   → Small numerical differences are normal")
        print("   → Differences should still be < 0.2%")
        return 1


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("Reproducibility Verification for:")
    print("Energy-Efficient ML for Traffic Congestion Prediction")
    print("SASIGD 2026")
    print("🔬 " * 20)
    
    exit_code = verify_results()
    
    if exit_code == 0:
        print("\n" + "🎉 " * 20)
        print("REPRODUCIBILITY CERTIFIED!")
        print("🎉 " * 20 + "\n")
    
    sys.exit(exit_code)
