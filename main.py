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


root_path = "/home/xinsheng/skinImage/"



def read_data():
    # define transforms for train dataset and test dataset.
    train_transforms = transforms.Compose([
            # AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Microscope(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    train_she = pd.read_csv(root_path + "train.csv")
    test_sheet = pd.read_csv(root_path + "test.csv")
    
    # train_set = MelanomaDataset(train_sheet, root_path + "jpeg/train/", True, train_transforms, None)
    split_from = int(0.8 * len(train_she))
    train_sheet = train_she[: split_from]
    validation_sheet = train_she[split_from:].reset_index()

    train_set = MelanomaDataset(train_sheet, root_path + "jpeg/train/", True, train_transforms, None)
    validation_set = MelanomaDataset(validation_sheet, root_path + "jpeg/train/", False, test_transforms, None)
    test_set = MelanomaDataset(test_sheet, root_path + "jpeg/test/", False, train_transforms, None)
    print(len(train_set), len(validation_set), len(test_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader



def main():
    train_transforms = transforms.Compose([
            # AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Microscope(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    train_she = pd.read_csv(root_path + "train.csv")
    test_sheet = pd.read_csv(root_path + "test.csv")
    
    # train_set = MelanomaDataset(train_sheet, root_path + "jpeg/train/", True, train_transforms, None)
    split_from = int(0.8 * len(train_she))
    train_sheet = train_she[: split_from]
    validation_sheet = train_she[split_from:].reset_index()

    train_set = MelanomaDataset(train_sheet, root_path + "jpeg/train/", True, train_transforms, None)
    validation_set = MelanomaDataset(validation_sheet, root_path + "jpeg/train/", False, test_transforms, None)
    test_set = MelanomaDataset(test_sheet, root_path + "jpeg/test/", False, train_transforms, None)
    print(len(train_set), len(validation_set), len(test_set))

    model = ResNet50(1)
    model = DataParallel(model)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model.to(device)

    epochs = 12  # Number of epochs to run
    es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
    TTA = 3 # Test Time Augmentation rounds

    oof = np.zeros((len(train_sheet), 1))  # Out Of Fold predictions
    preds = torch.zeros((len(test_sheet), 1), dtype=torch.float32, device=device)  # Predictions for test test

      
        
    model_path = f'model_{fold}.pth'  # Path and filename to save model to
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
    criterion = nn.BCEWithLogitsLoss()
    
    train = train_loader
    val = val_loader
    test = test_loader
    
    train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=4)
    
    for epoch in range(epochs):
        # start_time = time.time()
        correct = 0
        epoch_loss = 0
        model.train()
        best_val = 0
        
        for x, y in train_loader:
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
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
            val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
            val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu())
            
            print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} |'.format(
                    epoch + 1, epoch_loss, train_acc, val_acc, val_roc)

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
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()
            
            # Predicting on test set
            tta_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
            for _ in range(TTA):
                for i, x_test in enumerate(test_loader):
                    x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                    x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    tta_preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test[0].shape[0]] += z_test
            preds += tta_preds / TTA
        
    preds /= skf.n_splits


if __name__ == '__main__':
    main()
