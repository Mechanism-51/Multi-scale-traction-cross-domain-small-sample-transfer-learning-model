import torch
from sklearn.metrics import confusion_matrix, f1_score


def test(model, dataloader, epoch, cuda=False):
    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0
    all_preds = []
    all_targets = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            if cuda:
                data, labels = data.cuda(), labels.cuda()

            outputs, _, _,_ = model(data)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    cmatrix = confusion_matrix(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')

    results = {
        'epoch': epoch,
        'average_loss': avg_loss,
        'correct_class': correct,
        'total_elems': total,
        'accuracy %': accuracy,
        'confusion_matrix': cmatrix,
        'f1_score': f1
    }

    return results
