# Green AI Traffic Congestion Prediction

Energy-efficient machine learning for urban traffic congestion prediction. Complete reproducible implementation for **SASIGD 2026** paper.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Paper

**Title**: Energy-Efficient Machine Learning for Traffic Congestion Prediction: A Green AI Approach for Smart Cities

**Conference**: SASIGD 2026 - Sustainable AI and Social Impact for Global Development

**Authors**: Madhu Sanku

## Summary

We compared 6 machine learning models for traffic congestion prediction on the METR-LA dataset (207 sensors, 34K samples). Key finding: **Lightweight models achieve 93% F1-score while using 18,000× less memory than KNN**.

### Results Table

| Model | Accuracy | F1-Score | Training Time | Inference Time | Model Size |
|-------|----------|----------|---------------|----------------|------------|
| **Naive Bayes** | **0.9864** | **0.9345** | **0.14s** | **4.85×10⁻⁶s** | **7.19 KB** |
| Logistic Regression | 0.9803 | 0.9063 | 0.49s | 5.35×10⁻⁷s | 2.44 KB |
| Decision Tree | 0.9831 | 0.9201 | 2.73s | 5.05×10⁻⁷s | 6.95 KB |
| Random Forest | 0.9837 | 0.9200 | 5.48s | 1.75×10⁻⁶s | 428.59 KB |
| Gradient Boosting | 0.9841 | 0.9232 | 39.21s | 2.16×10⁻⁶s | 72.93 KB |
| KNN | 0.9831 | 0.9183 | 0.011s | 9.21×10⁻⁵s | 44,551.78 KB |

Naive Bayes - Best F1-score with minimal energy cost!

## To Implement

### 1. Install Dependencies

```bash
pip install h5py numpy pandas scikit-learn joblib
```

Or use exact versions for perfect reproducibility:

```bash
pip install h5py==3.9.0 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.2
```

### 2. Download Dataset

1. Visit: [METR-LA on Kaggle](https://www.kaggle.com/datasets/madmuthu/metr-la)
2. Download `metr-la.h5` 
3. Place it in the same folder as the Python scripts

### 3. Run Experiments

```bash
python traffic_prediction.py
```

Expected output:
```
[1/6] Loading METR-LA dataset...
✓ Dataset loaded: 34272 samples, 207 sensors

[2/6] Creating features and labels...
Class distribution:
  Free-flow (0): 17136 samples (50.00%)
  Congestion (1): 17136 samples (50.00%)

...

✓ Experiment completed successfully!
```

### 4. Verify Reproducibility

```bash
python verify_reproducibility.py
```

Expected output:
```
✓ Logistic Regression: Accuracy 0.9803, F1 0.9063
✓ Decision Tree: Accuracy 0.9831, F1 0.9201
✓ Naive Bayes: Accuracy 0.9864, F1 0.9345
...
✓✓✓ ALL RESULTS VERIFIED SUCCESSFULLY! ✓✓✓
```

## Files

- **`traffic_prediction.py`** - Main implementation (all 6 models)
- **`verify_reproducibility.py`** - Verification script
- **`metr-la.h5`** - Dataset (download separately)
- **`results/`** - Generated results folder
  - `results_table.csv` - Performance metrics
  - `*.joblib` - Trained models

## 🔬 Methodology

### Dataset
- **Source**: METR-LA (Los Angeles traffic sensors)
- **Sensors**: 207 loop detectors
- **Period**: March-June 2012
- **Samples**: 34,272 (5-minute intervals)
- **Features**: Traffic speed (km/h)

### Preprocessing
1. **Target Definition**: Congestion = average speed < 40 km/h
2. **Temporal Shift**: Use time t to predict t+1 (prevents leakage)
3. **Train-Test Split**: 80-20 chronological split
4. **Scaling**: StandardScaler normalization

### Models
All models use scikit-learn with default hyperparameters except:
- Logistic Regression: `max_iter=1000`
- Decision Tree: `max_depth=10`
- Random Forest: `n_estimators=50, max_depth=10`
- Gradient Boosting: `n_estimators=50`
- KNN: `n_neighbors=5`

### Metrics
- **Performance**: Accuracy, F1-Score
- **Energy Proxies**: Training time, inference time, model size

## Key Insights

1. **Lightweight models are competitive**: F1-score within 2% of ensemble methods
2. **Massive efficiency gains**: 18,000× smaller than KNN, <1s training
3. **Ideal for edge deployment**: Sub-microsecond inference, <8 KB storage
4. **Green AI validated**: High performance ≠ high energy cost

## Use Cases

This approach is ideal for:
- Edge devices with limited resources
- Real-time traffic management systems  
- Frequent model retraining scenarios
- Large-scale smart city deployments
- Energy-conscious AI applications

## Requirements

- Python 3.8+
- 8GB RAM (minimum)
- 500MB disk space
- Libraries: h5py, numpy, pandas, scikit-learn, joblib

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{author2026green,
  title={Energy-Efficient Machine Learning for Traffic Congestion Prediction: A Green AI Approach for Smart Cities},
  author={Madhu Sanku},
  booktitle={Proceedings of SASIGD 2026},
  year={2026}
}
```


## Acknowledgments

- **Dataset**: METR-LA from Kaggle ([link](https://www.kaggle.com/datasets/madmuthu/metr-la))
- **Inspiration**: Li et al. (ICLR 2018), Schwartz et al. (CACM 2020)
- **Conference**: SASIGD 2026

## Contact

For questions:
- Open an issue in this repository
- Email: [sankumadhu88@gmail.com]

---
