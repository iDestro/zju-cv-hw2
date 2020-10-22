import torch
import numpy as np
from sklearn import metrics


def train(model, config, train_iter, test_iter):
    model.to(config.device)
    total_batch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(config.num_epochs):
        for i, data in enumerate(train_iter):
            x, y = data[0].to(config.device), data[1].to(config.device)
            y_pred = model(x)
            model.zero_grad()
            loss = loss_func(y, y_pred)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                print(epoch, i, loss.cpu())
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
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
            loss = torch.nn.CrossEntropyLoss(y, y_pred)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(y_pred.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)