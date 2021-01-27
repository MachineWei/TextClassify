# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from models import TextCNN, TextRCNN, TextRNN, TextRNNAttention, DPCNN
from models import TextCNNConfig, TextRCNNConfig, TextRNNConfig, TextRNNAttentionConfig, TextDPCNNConfig
from data.data_utils import LoadData
from data import batch_iter
from tensorboardX import SummaryWriter
from create_log import create_logger

# 可复现性
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.teterministic = True

global logger
logger = create_logger('train.log')

class Platform:

    def __init__(self, model, model_config, device='gpu'):
        self.load_data = LoadData()
        self.device = device
        self.model_config = model_config
        self._device()
        self.model = model(self.model_config)
        self.model.to(self.device)
        self.writer = SummaryWriter("runs/model")

    def _device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        # 载入数据

        x_data, x_labels = self.load_data.load_data('train', self.model_config.sequence_length)
        z_data, z_labels = self.load_data.load_data('val', self.model_config.sequence_length)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config.lr)
        total_batch = 0
        val_best_loss = float("inf")
        last_improve = 0

        flag = False
        for epoch in range(self.model_config.num_epochs):
            print("Epoch [{}/{}]".format(epoch + 1, self.model_config.num_epochs))
            for i, (train_data, labels) in enumerate(batch_iter(x_data, x_labels, self.model_config.batch_size)):
                batch_input = torch.LongTensor(train_data).to(self.device)
                outputs = self.model(batch_input)
                self.model.zero_grad()
                batch_label = torch.LongTensor(labels).to(self.device)
                loss = F.cross_entropy(outputs, batch_label)
                loss.backward()
                optimizer.step()
                total_batch += 1
                if total_batch % 100 == 0:
                    true = torch.LongTensor(labels).data
                    predict = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predict)   # 100批次中最后一批次准确率
                    val_acc, val_loss = self.evaluate(z_data, z_labels)
                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                        torch.save(self.model.state_dict(), self.model_config.model_save_path)
                        improve = " *"
                        last_improve = total_batch
                    else:
                        improve = ""
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}' + improve
                    logger.info(msg.format(total_batch, loss.item(), train_acc, val_loss, val_acc))
                    self.writer.add_scalar("loss/train", loss.item(), total_batch)
                    self.writer.add_scalar("loss/dev", val_loss, total_batch)
                    self.writer.add_scalar("acc/train", train_acc, total_batch)
                    self.writer.add_scalar("acc/dev", val_acc, total_batch)
                    self.model.train()
                if total_batch - last_improve > self.model_config.require_improvement:
                    logger.info("No improvement for {} batches, I quit!!!".format(self.model_config.require_improvement))
                    flag = True
                    break
            if flag:
                break
        self.test()

    def evaluate(self, z_data, z_labels, test_data=False):
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            dataset_batch_generator = batch_iter(z_data, z_labels, self.model_config.batch_size)
            batch_count = 0
            for (data, labels) in dataset_batch_generator:
                batch_input = torch.LongTensor(data).to(self.device)
                outputs = self.model(batch_input)
                batch_label = torch.LongTensor(labels).to(self.device)
                val_loss = F.cross_entropy(outputs, batch_label)
                loss_total += val_loss
                labels = torch.LongTensor(labels).numpy()
                max_value, max_index = torch.max(outputs, axis=1)
                predict = max_index.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predict)
                batch_count += 1
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test_data:
            report = metrics.classification_report(labels_all, predict_all)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / batch_count, report, confusion
        return acc, loss_total / batch_count

    def test(self):
        self.model.eval()
        y_data, y_labels= self.load_data.load_data('test', self.model_config.sequence_length)
        test_acc, test_loss, test_report, test_confusion = self.evaluate(y_data, y_labels, test_data=True)
        logger.info("Test loss: {0:>5.2},  Test acc: {1:>6.2%}".format(test_loss, test_acc))
        print("Precision, Recall, and F1-score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)


if __name__ == '__main__':
    new_model = Platform(TextCNN, TextCNNConfig)
    new_model.train()

