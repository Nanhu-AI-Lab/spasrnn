'''
The metrics helper. Note it has some issue when dealing with multi-batch.
Please import torchmetrics and use it.
'''
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Score(object):
    '''
    To calculate the score.
    '''
    def __init__(self, y_true, y_pred, average:str):
        self.y_true = y_true
        self.y_pred = y_pred
        self.average = average
    def cal_acc(self):
        return accuracy_score(self.y_true, self.y_pred)
    def cal_precision(self):
        return precision_score(self.y_true, self.y_pred, average=self.average)
    def cal_recall(self):
        return recall_score(self.y_true, self.y_pred, average=self.average)
    def cal_f1(self):
        return f1_score(self.y_true, self.y_pred, average=self.average)

def get_accuracy_bce(prediction, label):
    batch_size = prediction.shape[0]
    predicted_classes = prediction > 0.5
    correct_predictions = (predicted_classes == label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def metric_acc(score, labels):
    # global acc
    rounded_predictions = torch.round(torch.sigmoid(score))
    correct = (rounded_predictions == labels).float()
    acc = correct.sum() / len(correct)  # get_accuracy_bce(score, labels)
    return acc
