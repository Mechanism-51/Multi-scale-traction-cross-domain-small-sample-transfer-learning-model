import torch
from tqdm import trange
from torch.autograd import Variable
from self_loss_function import MMDLoss


def train(model, source_loader, target_loader,
          optimizer, epoch, lambda_factor, cuda=False):

    model.train()
    mmd_loss = MMDLoss(kernel_mul=2.0, kernel_num=5)

    results = []

    source_batches = list(enumerate(source_loader))
    target_batches = list(enumerate(target_loader))
    train_steps = min(len(source_batches), len(target_batches))

    for batch_idx in trange(train_steps):
        _, (source_data, source_label) = source_batches[batch_idx]
        _, (target_data, _) = target_batches[batch_idx]

        if cuda:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()

        output_src, src_fc6, src_fc7, src_fc8 = model(source_data)
        _, tgt_fc6, tgt_fc7, tgt_fc8 = model(target_data)

        classification_loss = torch.nn.functional.cross_entropy(output_src, source_label)

        # MK-MMD loss on fc6, fc7, and fc8
        mmd_loss1 = mmd_loss(src_fc6, tgt_fc6)
        mmd_loss2 = mmd_loss(src_fc7, tgt_fc7)
        mmd_loss3 = mmd_loss(src_fc8, tgt_fc8)

        transfer_loss = (1 / 3) * mmd_loss1 + (1 / 3) * mmd_loss2 + (1 / 3) * mmd_loss3
        total_loss = classification_loss + lambda_factor * transfer_loss

        total_loss.backward()
        optimizer.step()

        results.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': lambda_factor,
            'classification_loss': classification_loss.item(),
            'dan_loss': transfer_loss.item(),
            'total_loss': total_loss.item()
        })

        print('Train Epoch: {:2d} [{:2d}/{:2d}] | '
              'Lambda: {:.4f} | Class Loss: {:.6f} | MK-MMD: {:.6f} | Total Loss: {:.6f}'.format(
            epoch, batch_idx + 1, train_steps,
            lambda_factor,
            classification_loss.item(),
            transfer_loss.item(),
            total_loss.item()
        ))

    return results

