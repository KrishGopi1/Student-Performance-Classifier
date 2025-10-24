# Student Performance Classifier

A complete machine learning pipeline that predicts student academic performance based on metrics such as attendance, prior grades, study hours, and more.

## Structure

```
data/
  student_data.csv
  sample_input.csv
generate_data.py
train.py
predict.py
requirements.txt
README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generate data

Create synthetic dataset:
```bash
python generate_data.py
```

## Train model

Train and save a tuned Random Forest classifier:
```bash
python train.py
```

## Predict

Predict performance for new input data (without the `performance` column):
```bash
python predict.py --input data/sample_input.csv --output data/predictions.csv
```

## Notes

- Model and encoder are saved together in `model/model.joblib`.
- `sample_input.csv` is provided to test the pipeline.
- Recommended Python 3.10+.
