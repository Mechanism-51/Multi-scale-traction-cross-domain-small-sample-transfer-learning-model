import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix

def test(model, data_loader, epoch, cuda):
    model.eval()

    test_loss = 0
    correct_class = 0

    true_labels = []
    pred_labels = []

    for data, label in data_loader:
        if cuda:
            data, label = data.cuda(), label.cuda()

        with torch.no_grad():
            data, label = Variable(data), Variable(label)
            output, _ = model(data, data)

            test_loss += torch.nn.functional.cross_entropy(output, label, reduction='sum').item()

            pred = output.data.max(1, keepdim=True)[1]
            correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()

            true_labels.extend(label.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())

    test_loss /= len(data_loader.dataset)
    accuracy = (100. * correct_class / len(data_loader.dataset)).item()

    num_classes = 4
    all_labels = list(range(num_classes))

    f1 = f1_score(true_labels, pred_labels, labels=all_labels, average='weighted')

    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=all_labels)

    return {
        "epoch": epoch + 1,
        "average_loss": test_loss,
        "correct_class": correct_class.item(),
        "total_elems": len(data_loader.dataset),
        "accuracy %": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }
