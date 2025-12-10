import os
import torch
import warnings
warnings.filterwarnings("ignore")
import math
from tqdm import trange
from train import train
from test import test
from model import DANNet
from utils import load_pretrained_AlexNet, save_log, save_model, load_model
from custom_dataloader import get_custom_dataloader, print_samples_per_label_per_domain


# --------------------------------
CUDA = torch.cuda.is_available()
learning_rate = 0.0001
L2_DECAY = 5e-4
MOMENTUM = 0.9
NUM_CLASSES = 4
EPOCHS = 50
LAMBDA_FACTOR = 0.5
BATCH_SIZE_SOURCE = 32
BATCH_SIZE_TARGET = 32
ADAPT_DOMAIN = True
LOAD_MODEL_PATH = None


base_path = "#Data"
source_list = ['aokeng-1', 'aokeng-3', 'hengwen-1', 'hengwen-3', 'shuwen-1', 'shuwen-3', 'wu-1', 'wu-3']
target_list = ['aokeng-2', 'hengwen-2', 'shuwen-2', 'wu-2']

SOURCE_NAME = "defect13"
TARGET_NAME = "defect2"


def step_decay(epoch, learning_rate):
    initial_learning_rate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))


def main():
    print("Loading source & target datasets...")
    source_loader = get_custom_dataloader(base_path, source_list, batch_size=BATCH_SIZE_SOURCE)
    target_loader = get_custom_dataloader(base_path, target_list, batch_size=BATCH_SIZE_TARGET)
    for images, labels in source_loader:
        print("Input image shape from source_loader:", images.shape)
        break
    for images, labels in target_loader:
        print("Input image shape from target_loader:", images.shape)
        break

    print("Source set size:", len(source_loader.dataset))
    print("Target set size:", len(target_loader.dataset))

    print_samples_per_label_per_domain("source_loader", source_loader)
    print_samples_per_label_per_domain("target_loader", target_loader)


    model = DANNet(num_classes=NUM_CLASSES)
    if CUDA:
        model = model.cuda()
        print("Using CUDA...")


    if LOAD_MODEL_PATH is not None:
        load_model(model, LOAD_MODEL_PATH)
    else:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        print("[INFO] Using pretrained AlexNet feature extractor (frozen)")


    training_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    print("Running training for {} epochs...".format(EPOCHS))

    for epoch in trange(EPOCHS):

        cur_lr = step_decay(epoch, learning_rate)
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Learning rate: {cur_lr:.6f}")


        optimizer = torch.optim.SGD([
            {"params": model.parameters()},
        ], lr=cur_lr, momentum=MOMENTUM, weight_decay=L2_DECAY)


        lambda_factor = LAMBDA_FACTOR if ADAPT_DOMAIN else 0


        result_train = train(model, source_loader, target_loader,
                             optimizer, epoch + 1, lambda_factor, CUDA)


        training_statistic.append(result_train)
        print("[EPOCH] {}: Classification loss: {:.6f}, DAN loss: {:.6f}, Total loss: {:.6f}".format(
            epoch + 1,
            sum(r['classification_loss'] / r['total_steps'] for r in result_train),
            sum(r['dan_loss'] / r['total_steps'] for r in result_train),
            sum(r['total_loss'] / r['total_steps'] for r in result_train),
        ))


        test_source = test(model, source_loader, epoch, CUDA)
        test_target = test(model, target_loader, epoch, CUDA)
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)


        print("[Test Source] Epoch {} - Acc: {}/{} ({:.2f}%) | Avg loss: {:.4f} | F1 Score: {:.4f}".format(
            epoch + 1, test_source['correct_class'], test_source['total_elems'],
            test_source['accuracy %'], test_source['average_loss'], test_source['f1_score']
        ))

        print("[Test Target] Epoch {} - Acc: {}/{} ({:.2f}%) | Avg loss: {:.4f} | F1 Score: {:.4f}".format(
            epoch + 1, test_target['correct_class'], test_target['total_elems'],
            test_target['accuracy %'], test_target['average_loss'], test_target['f1_score']
        ))


        print(f"Confusion Matrix (Source Domain) at Epoch {epoch + 1}:\n{test_source['confusion_matrix']}")
        print(f"Confusion Matrix (Target Domain) at Epoch {epoch + 1}:\n{test_target['confusion_matrix']}")


    log_dir = os.path.join("logs",
                           f"{SOURCE_NAME}_to_{TARGET_NAME}",
                           f"{EPOCHS}_epochs_{BATCH_SIZE_SOURCE}_s_{BATCH_SIZE_TARGET}_t_batch_size")
    os.makedirs(log_dir, exist_ok=True)

    if ADAPT_DOMAIN:
        save_log(training_statistic, os.path.join(log_dir, 'adaptation_training_statistic.pkl'))
        save_log(testing_s_statistic, os.path.join(log_dir, 'adaptation_testing_s_statistic.pkl'))
        save_log(testing_t_statistic, os.path.join(log_dir, 'adaptation_testing_t_statistic.pkl'))
        save_model(model, os.path.join(log_dir, 'adaptation_checkpoint.pth'))
    else:
        save_log(training_statistic, os.path.join(log_dir, 'no_adaptation_training_statistic.pkl'))
        save_log(testing_s_statistic, os.path.join(log_dir, 'no_adaptation_testing_s_statistic.pkl'))
        save_log(testing_t_statistic, os.path.join(log_dir, 'no_adaptation_testing_t_statistic.pkl'))
        save_model(model, os.path.join(log_dir, 'no_adaptation_checkpoint.pth'))


if __name__ == '__main__':
    main()
