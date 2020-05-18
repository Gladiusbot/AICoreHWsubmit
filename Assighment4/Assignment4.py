# pip install torch torchvision
# pip install tensorflow==1.14.0
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
import numpy as np
from model import TextRNN
from cnews_loader import read_category, read_vocab, process_file
import datetime


class RNN_Model():

    def __init__(self):
        self.train_file = 'cnews.train.txt'
        self.test_file = 'cnews_test.txt'
        self.val_file = 'cnews.val.txt'
        self.vocab_file = 'cnews.vocab.txt'
        # 获取文本的类别及其对应id的字典
        categories, cat_to_id = read_category()
        print(categories)
        # 获取训练文本中所有出现过的字及其所对应的id
        words, word_to_id = read_vocab('cnews.vocab.txt')
        # print(words)
        # print(word_to_id)
        # print(word_to_id)
        # 获取字数
        vocab_size = len(words)

        # 数据加载及分批
        # 获取训练数据每个字的id和对应标签的one-hot形式
        x_train, y_train = process_file(
            'cnews.train.txt',
            word_to_id,
            cat_to_id,
            600)
        print('x_train=', x_train)
        x_val, y_val = process_file(
            'cnews.val.txt',
            word_to_id,
            cat_to_id,
            600)

        x_train, y_train = torch.LongTensor(x_train), torch.LongTensor(y_train)
        x_val, y_val = torch.LongTensor(x_val), torch.LongTensor(y_val)

        train_dataset = Data.TensorDataset(x_train, y_train)
        self.train_loader = Data.DataLoader(
            dataset=train_dataset, batch_size=300, shuffle=True, num_workers=2)

        val_dataset = Data.TensorDataset(x_val, y_val)
        self.val_loader = Data.DataLoader(
            dataset=val_dataset, batch_size=300, shuffle=True, num_workers=2)

    def train(self):
        model = TextRNN().cuda()
        Loss = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_acc = 0
        for epoch in range(10):
            torch.cuda.empty_cache()
            print('epoch=', epoch)
            # t0 = datetime.datetime.now()
            for step, (x_batch, y_batch) in enumerate(self.train_loader):
                x = x_batch.cuda()
                y = y_batch.cuda()
                out = model(x)
                loss = Loss(out, y)
    #             print('loss=', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = np.mean((torch.argmax(out, 1)
                                    == torch.argmax(y, 1)).cpu().numpy())
                print(accuracy)
            if(epoch + 1) % 5 == 0:
                for step, (x_batch, y_batch) in enumerate(self.val_loader):
                    with torch.no_grad():
                        x = x_batch.cuda()
                        y = y_batch.cuda()
                        out = model(x)
                        accuracy = np.mean((torch.argmax(out, 1)
                                            == torch.argmax(y, 1)).cpu().numpy())
    #                     print(accuracy)
                        if accuracy > best_val_acc:
                            torch.save(
                                model.state_dict(),
                                'model_params.pkl')
                            best_val_acc = accuracy
                            print("best acc update:", accuracy)
            # print(datetime.datetime.now() - t0)
            # t0 = datetime.datetime.now()


if __name__ == '__main__':
    new_model = RNN_Model()
    new_model.train()
