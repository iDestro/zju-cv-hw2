from datetime import timedelta

import torch
import numpy as np
import time
from sklearn import metrics


def train(model, config, train_iter, test_iter):
    epoch_start_time = time.time()
    model.to(config.device)
    total_batch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(config.num_epochs):
        for i, data in enumerate(train_iter):
            x, y = data[0].to(config.device), data[1].to(config.device)
            y_pred = model(x)
            model.zero_grad()
            batch_loss = loss(y_pred, y)
            batch_loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                y_pred_labels = torch.max(y_pred, dim=1)[1].cpu().numpy()
                train_acc = metrics.accuracy_score(y.cpu().numpy(), y_pred_labels)
                train_loss = batch_loss.item()
                msg = 'Time: {},  Train Loss: {},  Train Acc: {}'
                time_diff = timedelta(seconds=int(round(time.time()-epoch_start_time)))
                print(msg.format(time_diff, train_loss, train_acc))
            total_batch += 1
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x)
            loss = torch.nn.functional.cross_entropy(y_pred, y)
            loss_total += loss.item()
            labels = y.data.cpu().numpy()
            y_pred_labels = torch.max(y_pred, dim=1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, y_pred_labels)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
