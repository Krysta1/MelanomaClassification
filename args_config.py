import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--mode',type=str,default="train")

# datasets
parser.add_argument('--dataset_path',type=str,default='/home/xinsheng/skinImage/jpeg/train/',help='the path to save imgs')
parser.add_argument('-dataset_txt_path',type=str,default='./dataset/small_dataset.txt')
parser.add_argument('-train_txt_path',type=str,default='./dataset/Train.txt')
parser.add_argument('-test_txt_path',type=str,default='./dataset/Test.txt')
parser.add_argument('-val_txt_path',type=str,default='./dataset/Validation.txt')

# optimizer
parser.add_argument('--optimizer',default='sgd',choices=['sgd','rmsprop','adam','radam'])
parser.add_argument("--learning_rate",type=float,default=0.0005)
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight_decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# training
parser.add_argument("--checkpoint",type=str,default='./checkpoints')
parser.add_argument("--resume",default='',type=str,metavar='PATH',help='path to save the latest checkpoint')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--start_epoch",default=0,type=int,metavar='N')
parser.add_argument('--epochs',default=3,type=int,metavar='N')
parser.add_argument('--lr_patience', default=1, type=int)
parser.add_argument('--lr_factor', default=0.4, type=float)


parser.add_argument('--image-size',type=int,default=288)
parser.add_argument('--arch',default='resnet50',choices=['resnet34','resnet18','resnet50'])
parser.add_argument('--num_classes',default=2,type=int)

# model path
parser.add_argument('--model_path',default='./checkpoints/model_whole_dataset.tar',type=str)
parser.add_argument('--result_csv',default='./result.csv')

args = parser.parse_args()
