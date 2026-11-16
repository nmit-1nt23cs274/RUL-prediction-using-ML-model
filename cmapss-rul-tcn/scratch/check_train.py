import torch
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import split_by_unit
from src.data_utils import load_raw, add_train_rul, drop_low_info_sensors, make_sequences
from src.tcn_model import TCN

def main():
    checkpoint = torch.load('outputs/FD001/tcn_FD001.pt', map_location='cpu', weights_only=False)
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['normalizer']
    model_args = checkpoint['args']
    
    model = TCN(input_size=len(feature_cols), num_channels=model_args.get('channels', [32, 32, 64, 64]))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    train_df = load_raw('data/FD001/train_FD001.txt')
    train_df = add_train_rul(train_df)
    train_df = drop_low_info_sensors(train_df, [f's{i}' for i in range(1, 22)])
    X_train = scaler.transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    
    seq_train_X, seq_train_y = make_sequences(
        train_df.assign(**{c: X_train[:, i] for i, c in enumerate(feature_cols)}), 
        feature_cols, 
        30
    )
    
    with torch.no_grad():
        preds = model(torch.tensor(seq_train_X[:200], dtype=torch.float32)).numpy()
        
    print('Pred mean:', np.mean(preds))
    print('Pred std:', np.std(preds))
    print('True mean:', np.mean(seq_train_y[:200]))
    print('True std:', np.std(seq_train_y[:200]))

if __name__ == "__main__":
    main()

