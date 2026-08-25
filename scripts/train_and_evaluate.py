import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.fno3d import FNO3D
import random
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class BladeNetValDataset(Dataset):
    def __init__(self, npz_files, norm_stats):
        self.data_cache = []
        self.mean, self.std = norm_stats['mean'], norm_stats['std']
        for f in npz_files:
            d = np.load(f)
            X = np.concatenate([d['coordinates'][::2,::2,::2,:], np.expand_dims(d['sdf'][::2,::2,::2], -1)], axis=-1).transpose(3,0,1,2)
            vx = d['velocity'][::2,::2,::2]
            Y = np.stack([d['density'][::2,::2,::2], vx, np.zeros_like(vx), np.zeros_like(vx), d['pressure'][::2,::2,::2]], axis=0)
            self.data_cache.append((torch.from_numpy(X).float(), (torch.from_numpy(Y).float() - self.mean) / self.std, f, torch.from_numpy(Y).float()))
    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

def compute_continuity(unnorm_tensor):
    rho = unnorm_tensor[:, 0]
    ux = unnorm_tensor[:, 1]
    flux = rho * ux
    return (flux[:, 2:, 1:-1, 1:-1] - flux[:, :-2, 1:-1, 1:-1]) / 2.0

def train_model(l_val, tl, vl, device, mean_t, std_t, seed=42, save_path=None):
    set_seed(seed)
    model = FNO3D(modes1=8, modes2=8, modes3=8, width=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    mean_t, std_t = mean_t.to(device), std_t.to(device)
    
    best_val_loss = float('inf')
    best_state = None
    
    for ep in range(10):
        model.train()
        for X, Y, _, _ in tl:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            if l_val > 0:
                loss += l_val * torch.mean(compute_continuity(pred * std_t + mean_t)**2)
            loss.backward()
            optimizer.step()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, Y, _, _ in vl:
                X, Y = X.to(device), Y.to(device)
                v_loss += criterion(model(X), Y).item() * X.size(0)
        v_loss /= len(vl.dataset)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = model.state_dict().copy()
            
    model.load_state_dict(best_state)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        
    return model

def evaluate_test(model, test_loader, device, mean_t, std_t, save_dir=None):
    model.eval()
    mean_t, std_t = mean_t.to(device), std_t.to(device)
    all_l2, all_cont = [], []
    with torch.no_grad():
        for X, Y, fnames, Y_gt in test_loader:
            X = X.to(device)
            pred_unnorm = model(X) * std_t + mean_t
            for i in range(X.size(0)):
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(os.path.join(save_dir, os.path.basename(fnames[i]).replace('.npz', '_pred.npy')), pred_unnorm[i].cpu().numpy())
                
                l2 = np.linalg.norm(pred_unnorm[i].cpu().numpy() - Y_gt[i].numpy()) / np.linalg.norm(Y_gt[i].numpy())
                cont = np.mean(np.abs(compute_continuity(pred_unnorm[i].unsqueeze(0).cpu().numpy())))
                all_l2.append({'geometry_id': os.path.basename(fnames[i]), 'l2': l2, 'cont': cont})
    return all_l2

def main(run_low_data=False):
    selected_lambda = 1.0  # Validation-selected physics weight
    print(f"Using validation-selected lambda: {selected_lambda}")
    
    train_files = sorted(glob.glob("data/pilot/train/*.npz"))
    val_files = sorted(glob.glob("data/pilot/val/*.npz"))
    test_files = sorted(glob.glob("data/pilot/test/*.npz"))
    
    print("Computing Dataset Normalization Statistics...")
    all_y = []
    for f in train_files:
        d = np.load(f)
        vx = d['velocity'][::2,::2,::2]
        all_y.append(np.stack([d['density'][::2,::2,::2], vx, np.zeros_like(vx), np.zeros_like(vx), d['pressure'][::2,::2,::2]], axis=0))
    all_y_arr = np.stack(all_y)
    mean_t = torch.from_numpy(all_y_arr.mean(axis=(0,2,3,4)).reshape(5,1,1,1)).float()
    std_t = torch.from_numpy(all_y_arr.std(axis=(0,2,3,4)).reshape(5,1,1,1)).float()
    std_t[std_t < 1e-6] = 1.0
    
    train_ds = BladeNetValDataset(train_files, {'mean': mean_t, 'std': std_t})
    val_ds = BladeNetValDataset(val_files, {'mean': mean_t, 'std': std_t})
    test_ds = BladeNetValDataset(test_files, {'mean': mean_t, 'std': std_t})
    
    tl = DataLoader(train_ds, batch_size=4, shuffle=True)
    vl = DataLoader(val_ds, batch_size=4, shuffle=False)
    testl = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("results/checkpoints", exist_ok=True)
    
    # 1. Primary Training (if checkpoints missing)
    fno_path = "results/checkpoints/lambda_0.0.pt"
    if not os.path.exists(fno_path):
        print("Training Baseline FNO (lambda=0.0)...")
        train_model(0.0, tl, vl, device, mean_t, std_t, save_path=fno_path)
    else:
        print("Found existing Baseline FNO checkpoint.")

    pi_path = f"results/checkpoints/lambda_{selected_lambda}.pt"
    if not os.path.exists(pi_path):
        print(f"Training PI-FNO (lambda={selected_lambda})...")
        train_model(selected_lambda, tl, vl, device, mean_t, std_t, save_path=pi_path)
    else:
        print("Found existing PI-FNO checkpoint.")
    
    # 2. Evaluate final models on TEST
    print("Evaluating FNO on Final Test Set...")
    model_fno = FNO3D(modes1=8, modes2=8, modes3=8, width=16).to(device)
    model_fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
    fno_res = evaluate_test(model_fno, testl, device, mean_t, std_t, "results/predictions/FNO")
    
    print(f"Evaluating PI-FNO on Final Test Set...")
    model_pi = FNO3D(modes1=8, modes2=8, modes3=8, width=16).to(device)
    model_pi.load_state_dict(torch.load(pi_path, map_location=device, weights_only=True))
    pi_res = evaluate_test(model_pi, testl, device, mean_t, std_t, "results/predictions/PIFNO")
    
    os.makedirs("results/evaluation", exist_ok=True)
    pd.DataFrame(fno_res).to_csv("results/predictions/FNO_TEST_METRICS.csv", index=False)
    pd.DataFrame(pi_res).to_csv("results/predictions/PIFNO_TEST_METRICS.csv", index=False)
    
    df_f = pd.DataFrame(fno_res)
    df_p = pd.DataFrame(pi_res)
    df_comp = pd.DataFrame({
        'geometry_id': df_f['geometry_id'],
        'FNO_L2': df_f['l2'],
        'PIFNO_L2': df_p['l2'],
        'L2_Diff': df_p['l2'] - df_f['l2'],
        'FNO_Cont': df_f['cont'],
        'PIFNO_Cont': df_p['cont'],
        'Cont_Diff': df_f['cont'] - df_p['cont']
    })
    df_comp.to_csv("results/evaluation/FINAL_COMPARISON.csv", index=False)
    
    # 3. Optional Low-Data Experiment (N=25)
    if run_low_data:
        print("Running Low Data Training (N=25)...")
        train_25 = train_files[:25]
        ds_25 = BladeNetValDataset(train_25, {'mean': mean_t, 'std': std_t})
        tl_25 = DataLoader(ds_25, batch_size=4, shuffle=True)
        
        low_data_res = []
        os.makedirs("results/evaluation/low_data", exist_ok=True)
        
        for s in [42, 123, 2026]:
            print(f"Low-Data: Training Seed {s} FNO...")
            mf = train_model(0.0, tl_25, vl, device, mean_t, std_t, seed=s)
            rf = evaluate_test(mf, testl, device, mean_t, std_t)
            
            print(f"Low-Data: Training Seed {s} PIFNO...")
            mp = train_model(selected_lambda, tl_25, vl, device, mean_t, std_t, seed=s)
            rp = evaluate_test(mp, testl, device, mean_t, std_t)
            
            f_mean_l2 = np.mean([x['l2'] for x in rf])
            p_mean_l2 = np.mean([x['l2'] for x in rp])
            f_mean_c = np.mean([x['cont'] for x in rf])
            p_mean_c = np.mean([x['cont'] for x in rp])
            
            low_data_res.append({
                'seed': s,
                'FNO_L2': f_mean_l2,
                'PIFNO_L2': p_mean_l2,
                'L2_Diff': p_mean_l2 - f_mean_l2,
                'FNO_Cont': f_mean_c,
                'PIFNO_Cont': p_mean_c,
                'Cont_Diff': f_mean_c - p_mean_c,
                'Cont_Imp_Pct': (1 - p_mean_c/f_mean_c)*100
            })
            
        pd.DataFrame(low_data_res).to_csv("results/evaluation/low_data_results.csv", index=False)
        
    print("Pipeline Execution Complete.")

def run_smoke_test():
    print("Running execution smoke test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO3D(modes1=8, modes2=8, modes3=8, width=16).to(device)
    dummy_input = torch.randn(2, 4, 128, 32, 16).to(device)
    try:
        output = model(dummy_input)
        loss = torch.mean(output**2)
        loss.backward()
        print("Smoke test passed: Model successfully instantiated, completed forward pass, and computed gradients.")
        sys.exit(0)
    except Exception as e:
        print(f"Smoke test failed: {e}")
        sys.exit(1)

def cli_main():
    parser = argparse.ArgumentParser(description="Train and Evaluate PI-FNO")
    parser.add_argument('--smoke-test', action='store_true', help='Run a synthetic forward/backward pass to test environment')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--run-low-data', action='store_true', help='Execute the optional N=25 low-data confirmation experiment')
    args = parser.parse_args()
    
    if args.smoke_test:
        run_smoke_test()
        
    if not os.path.exists("data/pilot/test"):
        print("Error: Required dataset directories not found in 'data/pilot/'.")
        print("Please read 'scripts/download_data.py' for instructions to acquire the dataset before executing the primary pipeline.")
        sys.exit(1)
        
    os.makedirs("results/predictions/FNO", exist_ok=True)
    os.makedirs("results/predictions/PIFNO", exist_ok=True)
    
    main(run_low_data=args.run_low_data)

if __name__ == "__main__":
    cli_main()
