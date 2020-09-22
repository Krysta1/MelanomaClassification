import torch
import torch.nn as nn
import numpy as np
import argparse
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
from utils import UnNormalize, set_seed, GradualWarmupSchedulerV2
warnings.simplefilter('ignore')

data_path = "/home/xinsheng/PycharmProjects/U-2-Net/"


def train(train_loader, train_data, model, device, optim, criterion, epoch, meta_features):
    correct = 0
    epoch_loss = 0
    model.train()
    for x, y in tqdm(train_loader):
        # origin_image = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x[0])
        # viz.images(origin_image)
        # viz.image(x[0])
        y = y.to(device)
        optim.zero_grad()
        z = model(x)
        # print(z.shape)
        loss = criterion(z, y.unsqueeze(1))
        loss.backward()
        optim.step()

        pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
        correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
        epoch_loss += loss.item()
    train_acc = correct / len(train_data)
    # viz2.plot('loss', 'train', 'Class Loss', epoch, epoch_loss / len(train_idx))
    return train_acc, epoch_loss


def val(val_loader, val_d, model, device, scheduler, meta_features):
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros(size=(len(val_d), 1), dtype=torch.float32, device=device)
    with torch.no_grad():  # Do not calculate gradient since we are only predicting
        # Predicting on validation set
        for j, (x_val, y_val) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if args.use_meta_features:
                l = x_val[0].shape[0]
            else:
                l = x_val.shape[0]
            z_val = model(x_val)
            # print(z_val)
            val_pred = torch.sigmoid(z_val)
            # print(val_pred.shape, x_val[0].shape)
            val_preds[j * l: j * l + l] = val_pred
            # print(val_pred)
        val_acc = accuracy_score(val_d['target'].values, torch.round(val_preds.cpu()))
        val_roc = roc_auc_score(val_d['target'].values, val_preds.cpu())
        # viz3.plot('loss', 'val', 'Class Loss', epoch, epoch_loss / len(val_idx))
        return val_acc, val_roc


def test(best_roc, fold, device, val_loader,test_loader, val_d, test_df, meta_features, oof):
    best_model_path = model_dir + [file for file in os.listdir(model_dir) if str(round(best_roc, 3)) in file and "Fold" + str(fold) in file][0]

    preds = torch.zeros((len(test_df), 1), dtype=torch.float32, device=device)
    # add meta feature from the csv file
    model = DataParallel(EfficientNetwork(1, args.arch, meta_features).to(device))
    model.load_state_dict(torch.load(best_model_path))  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        # print(f"-------------saving results to oof---------------")
        # val_preds = torch.zeros((len(val_d), 1), dtype=torch.float32, device=device)
        # for j, (x_val, y_val) in tqdm(enumerate(val_loader), total=len(val_loader)):
        #     y_val = y_val.to(device)
        #     if args.use_meta_features:
        #         l = x_val[0].shape[0]
        #     else:
        #         l = x_val.shape[0]
        #     z_val = model(x_val)
        #     val_pred = torch.sigmoid(z_val)
        #     val_preds[j * l:j * l + l] = val_pred
        # oof[val_idx] = val_preds.cpu().numpy()

        # Predicting on test set
        # tta_preds = torch.zeros((len(test_df), 1), dtype=torch.float32, device=device)
        for j in range(args.TTA):
            print(f"processing {j + 1}th TTA")
            for i, x_test in tqdm(enumerate(test_loader), total=len(test_loader)):
                if args.use_meta_features:
                    l = x_test[0].shape[0]
                else:
                    l = x_test.shape[0]
                z_test = model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[i * test_loader.batch_size:i * test_loader.batch_size + l] += z_test
    preds /= args.TTA
    return preds


def main():
    # train_df = pd.read_csv("./data/train-jpeg-256.csv")
    test_df = pd.read_csv("./data/test-jpeg-256.csv")
    # extra malignant data
    train_df = pd.read_csv("./data/train-jpeg-256.csv")

    pos_train = train_df[train_df['target'] == 1].reset_index(drop=True)
    pos_split = int(0.8 * len(pos_train))
    pos_train_df = pos_train[:pos_split]
    pos_val_df = pos_train[pos_split:]

    neg_train = train_df[train_df['target'] == 0].reset_index(drop=True)
    neg_split = int(0.8 * len(neg_train))
    neg_train_df = neg_train[:neg_split]
    neg_val_df = neg_train[neg_split:]
    
    # split all pos and neg data into 8:2
    train_data = pd.concat((pos_train_df, neg_train_df), axis=0).reset_index(drop=True)
    val_data = pd.concat((pos_val_df, neg_val_df), axis=0).reset_index(drop=True)
    print(f"Have total {len(train_data)} training data {len(train_data[train_data['target'] == 1])} positive data")
    print(f"Have total {len(val_data)} validation data {len(val_data[val_data['target'] == 1])} positive data")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Number of epochs to run
    epochs = args.epochs  
    # Early Stopping patience - for how many epochs with no improvements to wait
    es_patience = args.patience

    # process the meta features from csv files.
    if args.use_meta_features:
        meta_features = ['sex', 'age_approx', 'n_images'] + [x for x in train_df.columns if x.startswith('site_')]
        print('using meta features')
    else:
        meta_features = None
        print(f'no meta features used')
    print(f"current meta feature is {meta_features}")

    # Out Of Fold predictions
    oof = np.zeros((len(train_df), 1))  
    skf = KFold(n_splits=5, shuffle=True, random_state=47)
    best_roc = None
    fold = 1
    with open(model_dir + f"logs_{args.version}.txt", 'a+') as f:
        print('-' * 10, 'Fold:', fold, '-' * 10, file=f)
    print('-' * 10, 'Fold:', fold, '-' * 10)

    best_val = 0  # Best validation score within this fold
    patience_f = es_patience  # Current patience counter

    # model = ResNet50(1)
    model = EfficientNetwork(1, args.arch, meta_features)
    model = DataParallel(model).to(device)  # New model for each fold
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.4)
    # pos_num and neg_num is used for pos_weight parameter
    print(args.pos_weight)
    if args.pos_weight:
        pos_num = pd.value_counts(train_df['target']==0)[0]
        neg_num = pd.value_counts(train_df['target']==0)[1]
        print(neg_num / pos_num)
        criterion = nn.BCEWithLogitsLoss(pos_weight= torch.Tensor([neg_num / pos_num]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    print(f"current transform is {args.transform}")
    if args.transform == "True":
        train_transform, test_transform = get_transforms()
    else:
        train_transform, test_transform = None, None
    
    train_d = train_data
    val_d = val_data
    if args.debug:
        train_data = MelanomaDataset(df=train_d[:192],
                        imfolder=args.train_path,
                                type="train",
                        train=True,
                        transforms=train_transform,
                        meta_features = meta_features)
        val_data = MelanomaDataset(df=val_d[:96],
                        imfolder=args.train_path,
                                type="train",
                        train=True,
                        transforms=test_transform,
                        meta_features = meta_features)
        test_data = MelanomaDataset(df=test_df[:96],
                        imfolder=args.test_path,
                                type="test",
                        train=False,
                        transforms=train_transform,
                        meta_features = meta_features)
        
    else:
        train_data = MelanomaDataset(df=train_d,
                                imfolder=args.train_path,
                                type="train",
                                train=True,
                                transforms=train_transform,
                                meta_features = meta_features)
        val_data = MelanomaDataset(df=val_d,
                        imfolder=args.train_path,
                                type="train",
                        train=True,
                        transforms=test_transform,
                        meta_features = meta_features)
        test_data = MelanomaDataset(df=test_df,
                        imfolder=args.test_path,
                                type="test",
                        train=False,
                        transforms=train_transform,
                        meta_features = meta_features)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.nums_worker)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.nums_worker)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.nums_worker)

    for epoch in range(epochs):
        print(f"starting training in epoch {epoch + 1}")
        train_acc, epoch_loss = train(train_loader, train_d, model, device, optim, criterion, epoch, meta_features)
        print(f"starting validation in epoch {epoch + 1}")
        val_acc, val_roc = val(val_loader, val_d, model, device, scheduler, meta_features)

        # print to the log file
        with open(model_dir + f"logs_{args.version}.txt", 'a+') as f:
            print('Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'. \
                    format(epoch + 1, epochs, epoch_loss, train_acc, val_acc, val_roc),
                    file=f)
        # Print to console
        print('Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'. \
                format(epoch + 1, epochs, epoch_loss, train_acc, val_acc, val_roc))

        scheduler.step(val_roc)

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

    preds = test(best_roc, fold, device,val_loader,test_loader, val_d, test_df, meta_features, oof)
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
    # viz = visdom.Visdom()
    # viz2 = VisdomLinePlotter(env_name="Train Loss")
    # viz3 = VisdomLinePlotter(env_name="Val loss")

    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="TEST")
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--root-path', type=str, default="/home/xinsheng/skinImage/")
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--transform', type=str, default="True")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--TTA', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--no-use-meta-features', action='store_false')
    parser.add_argument('--use-meta-features', action='store_true')

    parser.add_argument('--init', action='store_true')
    parser.add_argument('--nums-worker', type=int, default=4)
    parser.add_argument('--arch', type=int, default=4)
    parser.add_argument('--train-path', type=str,
                        default="/home/xinsheng/skinImage/data/jpeg-melanoma-512x512/train/")
    parser.add_argument('--test-path', type=str,
                        default="/home/xinsheng/PycharmProjects/U-2-Net/segment_test_512/")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--use-diagnosis', action='store_true')

    parser.add_argument('--pos-weight', action='store_false')
    parser.add_argument('--use-pos-weight', action='store_true')
    args, _ = parser.parse_known_args()
    if args.debug:
        args.epochs = 1
        args.TTA = 1
    model_dir = args.model_path + args.version + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # save args into log.txt
    if args.init:
        with open(model_dir + f"logs_{args.version}.txt", 'a+') as f:
            for arg in vars(args):
                print(arg, getattr(args, arg), file=f)

    print(f"model saved in {model_dir}! Version is {args.version}")
    main()
