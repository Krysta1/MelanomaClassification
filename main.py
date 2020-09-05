import torch
import torch.nn as nn
import numpy as np
from MelanomaDataset import MelanomaDataset, Microscope, get_transforms
from torch.utils.data import DataLoader
from build_model import ResNet50, EfficientNetwork
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import time, os
import warnings
from tqdm import tqdm
import visdom
from visdomClass import VisdomLinePlotter
from utils import UnNormalize, set_seed, parse_args, GradualWarmupSchedulerV2
warnings.simplefilter('ignore')

DEBUG = False


def train(train_df, train_idx, model, device, optim, criterion, epoch, meta_features):
    # load training dataset
    # get_transforms[0] is transform for training
    if DEBUG:
        train = MelanomaDataset(df=train_df.iloc[train_idx].reset_index(drop=True)[:192],
                                imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/train/',
                                train=True,
                                transforms=get_transforms()[0],
                                meta_features = meta_features)
    else:
        train = MelanomaDataset(df=train_df.iloc[train_idx].reset_index(drop=True),
                                imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/train/',
                                train=True,
                                transforms=get_transforms()[0],
                                meta_features = meta_features)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=args.nums_worker)
    correct = 0
    epoch_loss = 0
    model.train()
    for (image, meta_data), y in tqdm(train_loader):
        image = torch.tensor(image, device=device, dtype=torch.float32)
        meta_data = torch.tensor(meta_data, device=device, dtype=torch.float32)
        origin_image = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image[0])
        viz.images(origin_image)
        # for torchtoolbox transforms
        # y = torch.tensor(y, device=device, dtype=torch.float32)
        # for albumentations transforms
        y = y.to(device)
        optim.zero_grad()
        z = model(image, meta_data)
        # print(z.shape)

        # print(z)
        loss = criterion(z, y.unsqueeze(1))
        loss.backward()
        optim.step()
        pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
        correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
        epoch_loss += loss.item()
    train_acc = correct / len(train_idx)
    viz2.plot('loss', 'train', 'Class Loss', epoch, epoch_loss / len(train_idx))
    return train_acc, epoch_loss


def val(train_df, val_idx, model, device, scheduler, meta_features):
    # loading validation dataset
    if DEBUG == "True":
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True)[:96],
                              imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/train/',
                              train=True,
                              transforms=get_transforms()[1],
                                meta_features = meta_features)
    else:
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True),
                              imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/train/',
                              train=True,
                              transforms=get_transforms()[1],
                                meta_features = meta_features)
    val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=False, num_workers=args.nums_worker)
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
    with torch.no_grad():  # Do not calculate gradient since we are only predicting
        # Predicting on validation set
        for j, (x_val, y_val) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
            # y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val[0], x_val[1])
            # print(z_val)
            val_pred = torch.sigmoid(z_val)
            # print(val_pred.shape, x_val[0].shape)
            val_preds[j * val_loader.batch_size:j * val_loader.batch_size + x_val[0].shape[0]] = val_pred
            # print(val_pred)
        val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
        val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu())
        scheduler.step(val_roc)
        # viz3.plot('loss', 'val', 'Class Loss', epoch, epoch_loss / len(val_idx))
        return val_acc, val_roc


def test(best_roc, fold, nums_meta_features, device, train_df, val_idx, test_df, meta_features, oof):
    best_model_path = model_dir + [file for file in os.listdir(model_dir) if str(round(best_roc, 3)) in file and "Fold" + str(fold) in file][0]

    if DEBUG == "True":
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True)[:96],
                              imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/train/',
                              train=True,
                              transforms=get_transforms()[1],
                                meta_features = meta_features)
        test = MelanomaDataset(df=test_df[:96],
                               imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/test/',
                               train=False,
                               transforms=get_transforms()[0],
                                meta_features = meta_features)
    else:
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True),
                              imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/train/',
                              train=True,
                              transforms=get_transforms()[1],
                                meta_features = meta_features)
        test = MelanomaDataset(df=test_df,
                               imfolder='/home/xinsheng/skinImage/data/jpeg-melanoma-256/test/',
                               train=False,
                               transforms=get_transforms()[0],
                                meta_features = meta_features)

    val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=False, num_workers=args.nums_worker)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=args.nums_worker)
    # preds = torch.zeros((len(test_df), 1), dtype=torch.float32, device=device)  # Predictions for test test
    preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)

    # add meta feature from the csv file
    model = DataParallel(EfficientNetwork(1, 2, 2, meta_features).to(device))
    model.load_state_dict(torch.load(best_model_path))  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val), 1), dtype=torch.float32, device=device)
    print(f"-------------start testing---------------")
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
            # y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            y_val = y_val.to(device)
            z_val = model(x_val[0], x_val[1])
            val_pred = torch.sigmoid(z_val)
            val_preds[j * val_loader.batch_size:j * val_loader.batch_size + x_val[0].shape[0]] = val_pred
        oof[val_idx] = val_preds.cpu().numpy()

        # Predicting on test set
        tta_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
        for i in range(args.TTA):
            print(f"processing {i + 1}th TTA")
            for i, x_test in tqdm(enumerate(test_loader), total=len(test_loader)):
                x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                z_test = model(x_test[0], x_test[1])
                z_test = torch.sigmoid(z_test)
                tta_preds[i * test_loader.batch_size:i * test_loader.batch_size + x_test[0].shape[0]] += z_test
        preds += tta_preds
    preds /= args.TTA
    return preds


def main():
    train_df = pd.read_csv("./data/train-jpeg-256.csv")
    test_df = pd.read_csv("./data/test-jpeg-256.csv")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    epochs = args.epochs  # Number of epochs to run
    es_patience = args.patience  # Early Stopping patience - for how many epochs with no improvements to wait

    # process the meta features from csv files.
    meta_features = ['sex', 'age_approx', 'n_images'] + [x for x in train_df.columns if x.startswith('site_')]
    print(meta_features)
    nums_meta_features = len(meta_features)
    oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
    skf = KFold(n_splits=5, shuffle=True, random_state=47)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
        best_roc = None
        # if fold == 1:
        with open(model_dir + f"logs_{args.version}.txt", 'a+') as f:
            print('-' * 10, 'Fold:', fold, '-' * 10, file=f)
        print('-' * 10, 'Fold:', fold, '-' * 10)

        best_val = 0  # Best validation score within this fold
        patience_f = es_patience  # Current patience counter
        # model = ResNet50(1)
        model = EfficientNetwork(1, 2, 2, meta_features)
        model = DataParallel(model).to(device)  # New model for each fold
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        # scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
        # scheduler = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
        scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, torch.tensor(args.epochs - 1).to(device))
        # scheduler = GradualWarmupSchedulerV2(optim, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

        pos_num = pd.value_counts(train_df['target']==0)[0]
        neg_num = pd.value_counts(train_df['target']==0)[1]
        criterion = nn.BCEWithLogitsLoss(pos_weight= torch.Tensor([neg_num / pos_num]).to(device))
        # criterion = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            print(f"starting training in epoch {epoch + 1}")
            train_acc, epoch_loss = train(train_df, train_idx, model, device, optim, criterion, epoch, meta_features)
            print(f"starting validation in epoch {epoch + 1}")
            val_acc, val_roc = val(train_df, val_idx, model, device, scheduler, meta_features)

            # print to the log file
            with open(model_dir + f"logs_{args.version}.txt", 'a+') as f:
                print('Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'. \
                      format(epoch + 1, epochs, epoch_loss, train_acc, val_acc, val_roc),
                      file=f)
            # Print to console
            print('Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'. \
                  format(epoch + 1, epochs, epoch_loss, train_acc, val_acc, val_roc))

            if not best_roc:  # If best_roc = None
                best_roc = val_roc
                torch.save(model.state_dict(),
                           model_dir + f"{args.version}_Fold{fold}_Epoch{epoch + 1}_ValidAcc_{val_acc:.3f}_ROC_{val_roc:.3f}.pth")
                continue

            if val_roc > best_roc:
                best_roc = val_roc
                # Reset patience (because we have improvement)
                patience_f = es_patience
                torch.save(model.state_dict(),
                           model_dir + f"{args.version}_Fold{fold}_Epoch{epoch + 1}_ValidAcc_{val_acc:.3f}_ROC_{val_roc:.3f}.pth")
            else:
                # Decrease patience (no improvement in ROC)
                patience_f = patience_f - 1
                if patience_f == 0:
                    with open(model_dir + f"logs_{args.version}.txt", 'a+') as f:
                        print('Early stopping (no improvement since 3 models) | Best ROC: {}'. \
                              format(best_roc), file=f)
                    print('Early stopping (no improvement since 3 models) | Best ROC: {}'. \
                          format(best_roc))
                    break
        # else:
        #     break

        preds = test(best_roc, fold, nums_meta_features, device, train_df, val_idx, test_df, meta_features, oof)
        get_submission(preds, fold)

    # After 5 folds training and testing. Fill the whole oof array.
    # save the oof for later use
    pd.Series(oof.reshape(-1,)).to_csv(model_dir + f'oof_{args.version}.csv', index=False)


def get_submission(preds, fold):
    preds = preds.cpu().numpy()
    ss = pd.read_csv('/home/xinsheng/skinImage/sample_submission.csv')
    ss['target'] = preds
    file_name = f'{args.version}_Fold_{fold}.csv'
    ss.to_csv(model_dir + file_name, index=False)
    print(f"Submission file for {file_name} saved! Good Luck...")


if __name__ == '__main__':
    # visdom class for train loss and validation loss
    viz = visdom.Visdom()
    viz2 = VisdomLinePlotter(env_name="Train Loss")
    # viz3 = VisdomLinePlotter(env_name="Val loss")

    set_seed()

    args = parse_args()
    if DEBUG:
        args.epochs = 1
        args.TTA = 1
    model_dir = args.model_path + args.version + "/"
    print(f"model saved in {model_dir}! Version is {args.version}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    main()
