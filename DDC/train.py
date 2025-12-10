import torch
from tqdm import trange
from torch.autograd import Variable
from self_loss_function import MMDLoss


def train(model, source_loader, target_loader,
          optimizer, epoch, lambda_factor, cuda=False):
    model.train()
    mmd_loss = MMDLoss(kernel_type='rbf')

    results = []

    source = list(enumerate(source_loader))
    target = list(enumerate(target_loader))
    train_steps = min(len(source), len(target))

    # start batch training
    for batch_idx in trange(train_steps):
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx]

        print("CUDA:", cuda)
        if cuda:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()

        output1, output2 = model(source_data, target_data)

        classification_loss = torch.nn.functional.cross_entropy(output1, source_label)
        ddc_loss = mmd_loss(output1,output2)

        total_loss = classification_loss + lambda_factor*ddc_loss

        total_loss.backward()

        optimizer.step()

        results.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': lambda_factor,
            'ddc_loss': ddc_loss.item(),
            'classification_loss': classification_loss.item(),
            'total_loss': total_loss.item()
        })

        print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda value: {:.4f}, Classification loss: {:.6f}, ddc_loss: {:.6f}, Total_Loss: {:.6f}'.format(
                  epoch,
                  batch_idx + 1,
                  train_steps,
                  lambda_factor,
                  classification_loss.item(),
                  ddc_loss.item(),
                  total_loss.item()
              ))

    return results
