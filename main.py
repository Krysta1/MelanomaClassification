import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from MelanomaDataset import get_transforms, MelanomaDataset
from args_config import args
from torch.utils.data import DataLoader
from build_model import ResNet50
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torchtoolbox.transform as transforms
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import time
import datetime
import warnings
from tqdm import tqdm

warnings.simplefilter('ignore')
root_path = "/home/xinsheng/skinImage/"
skf = GroupKFold(n_splits=5)

def main():
    train_transforms = transforms.Compose([
            
            # AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Microscope(p=0.5),
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    train_df = pd.read_csv(root_path + "train.csv")
    test_df = pd.read_csv(root_path + "test.csv")


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    epochs = 10  # Number of epochs to run
    es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
    TTA = 3 # Test Time Augmentation rounds

    oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
    preds = torch.zeros((len(test_df), 1), dtype=torch.float32, device=device)  # Predictions for test test

    skf = KFold(n_splits=5, shuffle=True, random_state=47)
    test = MelanomaDataset(df=test_df,
                       imfolder='/home/xinsheng/skinImage/jpeg/test/', 
                       train=False,
                       transforms=train_transforms)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
        print('=' * 20, 'Fold', fold, '=' * 20)
        
        model_path = f'model_{fold}.pth'  # Path and filename to save model to
        best_val = 0  # Best validation score within this fold
        patience = es_patience  # Current patience counter
        model = ResNet50(1)
        model = DataParallel(model)  # New model for each fold
        model = model.to(device)
        
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
        criterion = nn.BCEWithLogitsLoss()
        
        train = MelanomaDataset(df=train_df.iloc[train_idx].reset_index(drop=True), 
                                imfolder='/home/xinsheng/skinImage/jpeg/train/', 
                                train=True, 
                                transforms=train_transforms)
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True), 
                                imfolder='/home/xinsheng/skinImage/jpeg/train/', 
                                train=True, 
                                transforms=test_transforms)
        
        train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(dataset=val, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=2)
        
        for epoch in range(epochs):
            start_time = time.time()
            correct = 0
            epoch_loss = 0
            model.train()
            print(f"starting training in epoch {epoch + 1}")
            for x, y in tqdm(train_loader):
                # print(x)
                x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)
                x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)
                y = torch.tensor(y, device=device, dtype=torch.float32)
                optim.zero_grad()
                z = model(x)
                loss = criterion(z, y.unsqueeze(1))
                loss.backward()
                optim.step()
                pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
                correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
                epoch_loss += loss.item()
            train_acc = correct / len(train_idx)
            
            model.eval()  # switch model to the evaluation mode
            print(f"starting eval in epoch {epoch + 1}")
            val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
            with torch.no_grad():  # Do not calculate gradient since we are only predicting
                # Predicting on validation set
                for j, (x_val, y_val) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                    # x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                    y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                    z_val = model(x_val)
                    val_pred = torch.sigmoid(z_val)
                    # print(val_pred.shape, x_val[0].shape)
                    val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
                val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
                val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu())
                
                print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                epoch + 1, 
                epoch_loss, 
                train_acc, 
                val_acc, 
                val_roc, 
                str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
                
                scheduler.step(val_roc)
                    
                if val_roc >= best_val:
                    best_val = val_roc
                    patience = es_patience  # Resetting patience since we have new best validation accuracy
                    torch.save(model, model_path)  # Saving current best model
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                        break
                    
        model = torch.load(model_path)  # Loading best model of this fold
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            # Predicting on validation set once again to obtain data for OOF
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val.shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()
            
            # Predicting on test set
            tta_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
            for _ in range(TTA):
                for i, x_test in enumerate(test_loader):
                    x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                    x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    tta_preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test.shape[0]] += z_test
            preds += tta_preds / TTA
        
    preds /= skf.n_splits
    
    return preds
    

if __name__ == '__main__':
    preds = main()
