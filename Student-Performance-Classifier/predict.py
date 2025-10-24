import argparse
import joblib
import pandas as pd

def main(model_path, input_path, output_path):
    model, le = joblib.load(model_path)
    df = pd.read_csv(input_path)
    preds = model.predict(df)
    labels = le.inverse_transform(preds)
    out = pd.DataFrame({'predicted_performance': labels})
    out.to_csv(output_path, index=False)
    print('Predictions written to', output_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=False, default='model/model.joblib')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    main(args.model, args.input, args.output)
