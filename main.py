import argparse
import pickle
import os
import copy
from utils.read_data import DataConverter
from utils.loader_utils import ImageDataset
from torch.utils.data import Dataset, DataLoader
from model import ImageSequentialClassifier
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    with  open('data/train_dataconverter.pickle', 'rb') as handle:
            data = pickle.load(handle)
    
    N = len(data.length)
    train_N = int(N * 0.9) # train : val = 9 : 1
    
    train_data = copy.deepcopy(data)
    train_data.length = train_data.length[:train_N]
    train_data.name_resized = train_data.name_resized[:train_N]
    train_data.label = train_data.label[:train_N]
    train_data.bbox_resized = train_data.bbox_resized[:train_N]

    dev_data = copy.deepcopy(data)
    dev_data.length = dev_data.length[train_N:]
    dev_data.name_resized = dev_data.name_resized[train_N:]
    dev_data.label = dev_data.label[train_N:]
    dev_data.bbox_resized = dev_data.bbox_resized[train_N:]

    with  open('data/test_dataconverter.pickle', 'rb') as handle:
            test_data = pickle.load(handle)

    transform = transforms.Compose([
        # transforms.RandomCrop([54, 54]),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_data, transform=transform)
    dev_dataset = ImageDataset(dev_data, transform=transform)
    test_dataset = ImageDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ImageSequentialClassifier().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)

    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []

    for epoch in range(args.num_epoch):
        model.train()
        train_loss = 0
        correct_num = 0
        total_num = 0
        for batch in tqdm(train_loader):

            optimizer.zero_grad()

            img, length, label, bbox = batch # img: (N, 3, H, W), length: (N,) , label: (N, 5),  bbox: (N, 5, 4)
            img, length, label, bbox = img.cuda(),length.cuda(), label.cuda(), bbox.cuda()

            length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits,\
            bbox1_logits, bbox2_logits, bbox3_logits, bbox4_logits, bbox5_logits = model(img)

            length_loss = F.cross_entropy(length_logits, length)
            digit1_loss = F.cross_entropy(digit1_logits, label[:, 0])
            digit2_loss = F.cross_entropy(digit2_logits, label[:, 1])
            digit3_loss = F.cross_entropy(digit3_logits, label[:, 2])
            digit4_loss = F.cross_entropy(digit4_logits, label[:, 3])
            digit5_loss = F.cross_entropy(digit5_logits, label[:, 4])
            bbox1_loss = F.l1_loss(bbox1_logits, bbox[:, 0 ,:])
            bbox2_loss = F.l1_loss(bbox2_logits, bbox[:, 1 ,:])
            bbox3_loss = F.l1_loss(bbox3_logits, bbox[:, 2 ,:])
            bbox4_loss = F.l1_loss(bbox4_logits, bbox[:, 3 ,:])
            bbox5_loss = F.l1_loss(bbox5_logits, bbox[:, 4 ,:])
            
            loss = length_loss + digit1_loss + digit2_loss + digit3_loss + digit4_loss + digit5_loss \
            + bbox1_loss + bbox2_loss + bbox3_loss + bbox4_loss + bbox5_loss

            train_loss += loss.item()
            pred = torch.stack([torch.argmax(digit1_logits, dim=-1), torch.argmax(digit2_logits, dim=-1), torch.argmax(digit3_logits, dim=-1), torch.argmax(digit4_logits, dim=-1), torch.argmax(digit5_logits, dim=-1)], dim=1)
            
            correct_num += ((pred==label).sum(dim=-1) == 5).sum()
            total_num += len(img)

            loss.backward()
            optimizer.step()
        print(label, pred)
        train_loss = train_loss / len(train_loader)
        train_accuracy = correct_num / total_num

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        scheduler.step()

    

        model.eval()
        dev_loss = 0
        correct_num = 0
        total_num = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader):
                img, length, label, bbox = batch 
                img, length, label, bbox = img.cuda(), length.cuda(), label.cuda(), bbox.cuda()

                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits,\
                bbox1_logits, bbox2_logits, bbox3_logits, bbox4_logits, bbox5_logits = model(img)

                length_loss = F.cross_entropy(length_logits, length)
                digit1_loss = F.cross_entropy(digit1_logits, label[:, 0])
                digit2_loss = F.cross_entropy(digit2_logits, label[:, 1])
                digit3_loss = F.cross_entropy(digit3_logits, label[:, 2])
                digit4_loss = F.cross_entropy(digit4_logits, label[:, 3])
                digit5_loss = F.cross_entropy(digit5_logits, label[:, 4])
                bbox1_loss = F.mse_loss(bbox1_logits, bbox[:, 0 ,:])
                bbox2_loss = F.mse_loss(bbox2_logits, bbox[:, 1 ,:])
                bbox3_loss = F.mse_loss(bbox3_logits, bbox[:, 2 ,:])
                bbox4_loss = F.mse_loss(bbox4_logits, bbox[:, 3 ,:])
                bbox5_loss = F.mse_loss(bbox5_logits, bbox[:, 4 ,:])
                
                loss = length_loss + digit1_loss + digit2_loss + digit3_loss + digit4_loss + digit5_loss \
                + bbox1_loss + bbox2_loss + bbox3_loss + bbox4_loss + bbox5_loss

                dev_loss += loss.item()
                pred = torch.stack([torch.argmax(digit1_logits, dim=-1), torch.argmax(digit2_logits, dim=-1), torch.argmax(digit3_logits, dim=-1), torch.argmax(digit4_logits, dim=-1), torch.argmax(digit5_logits, dim=-1)], dim=1)
                correct_num += ((pred==label).sum(dim=-1) == 5).sum()
                total_num += len(img)
            
            dev_loss = dev_loss / len(dev_loader)
            dev_accuracy = correct_num / total_num

            dev_losses.append(dev_loss)
            dev_accuracies.append(dev_accuracies)

        model.eval()
        test_loss = 0
        correct_num = 0
        total_num = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                img, length, label, bbox = batch 
                img, length, label, bbox = img.cuda(), length.cuda(), label.cuda(), bbox.cuda()

                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits,\
                bbox1_logits, bbox2_logits, bbox3_logits, bbox4_logits, bbox5_logits = model(img)

                length_loss = F.cross_entropy(length_logits, length)
                digit1_loss = F.cross_entropy(digit1_logits, label[:, 0])
                digit2_loss = F.cross_entropy(digit2_logits, label[:, 1])
                digit3_loss = F.cross_entropy(digit3_logits, label[:, 2])
                digit4_loss = F.cross_entropy(digit4_logits, label[:, 3])
                digit5_loss = F.cross_entropy(digit5_logits, label[:, 4])
                bbox1_loss = F.mse_loss(bbox1_logits, bbox[:, 0 ,:])
                bbox2_loss = F.mse_loss(bbox2_logits, bbox[:, 1 ,:])
                bbox3_loss = F.mse_loss(bbox3_logits, bbox[:, 2 ,:])
                bbox4_loss = F.mse_loss(bbox4_logits, bbox[:, 3 ,:])
                bbox5_loss = F.mse_loss(bbox5_logits, bbox[:, 4 ,:])
                
                loss = length_loss + digit1_loss + digit2_loss + digit3_loss + digit4_loss + digit5_loss \
                + bbox1_loss + bbox2_loss + bbox3_loss + bbox4_loss + bbox5_loss

                test_loss += loss.item()
                pred = torch.stack([torch.argmax(digit1_logits, dim=-1), torch.argmax(digit2_logits, dim=-1), torch.argmax(digit3_logits, dim=-1), torch.argmax(digit4_logits, dim=-1), torch.argmax(digit5_logits, dim=-1)], dim=1)
                correct_num += ((pred==label).sum(dim=-1) == 5).sum()
                total_num += len(img)
            
            test_loss = test_loss / len(test_loader)
            test_accuracy = correct_num / total_num

        

            print("epoch: {}    train_loss: {:.6f}    train_accuracy: {:.3f}    dev_loss: {:.6f}    dev_accuracy: {:.3f}    test_loss: {:.6f}    test_accuracy: {:.3f}"\
                  .format(epoch+1, train_loss, train_accuracy, dev_loss, dev_accuracy, test_loss, test_accuracy))
            
        
    if args.draw_learning_curve:
        x = np.arange(1, args.num_epoch+1)
        plt.plot(x, train_losses, label='train_loss')
        plt.plot(x, dev_losses, label='dev_loss')
        plt.legend()
        plt.title("learning curve of the model")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig("learning_curve.png")


            
        



             

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', default=20, type=int,  help='Default 20')
    parser.add_argument('--batch_size', default=32, type=int,  help='Default 32')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Default 1e-2')
    parser.add_argument('--draw_learning_curve', default=True, type=bool, help='Default 1e-2')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument( '--weight_decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')



    args = parser.parse_args()
    
    main(args)