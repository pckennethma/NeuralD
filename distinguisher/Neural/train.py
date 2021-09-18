import torchtext.data as data
from tqdm import tqdm
import os, sys, torch
try:
    from distinguisher.Neural.util import *
    from distinguisher.Neural.model import *
except:
    from util import *
    from model import *
import torch.nn.functional as F

def train(config: ModelConfig):
    text_field = data.Field()
    label_field = data.Field(sequential=False)
    train_loader, dev_loader, test_loader = get_train_dev_test_loader(text_field, label_field, config.train_file, config.val_file, batch_size=config.batch_size, device = "cuda")
    if config.model_arch == "CNN":
        input_dim = len(text_field.vocab)
        emb_dim = config.model_args["emb_dim"]
        n_filters = config.model_args["n_filters"]
        out_dim = len(label_field.vocab) - 1
        model = CNN(input_dim, emb_dim, n_filters, [2, 3, 4], out_dim).cuda()
        if os.path.exists(config.model_path): load_model(model, config.model_path)
        cnn_train(model, train_loader, dev_loader, test_loader, text_field, label_field)
    elif config.model_arch == "RNN":
        input_dim = len(text_field.vocab)
        emb_dim = config.model_args["emb_dim"]
        hidden_dim = config.model_args["hidden_dim"]
        out_dim = len(label_field.vocab)
        model = RNN(input_dim, emb_dim, hidden_dim, out_dim).cuda()
        rnn_train(model, train_loader, dev_loader, test_loader, text_field, label_field)
    elif config.model_arch == "LSTM":
        input_dim = len(text_field.vocab)
        emb_dim = config.model_args["emb_dim"]
        hidden_dim = config.model_args["hidden_dim"]
        out_dim = len(label_field.vocab)
        model = LSTM(input_dim, emb_dim, hidden_dim, out_dim).cuda()
        rnn_train(model, train_loader, dev_loader, test_loader, text_field, label_field)
    else:
        raise NotImplementedError()
    print("Model Parameter Count:",sum(p.numel() for p in model.parameters() if p.requires_grad))


def cnn_eval(data_loader, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch_idx, batch in enumerate(data_loader):
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_loader.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    # print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy, size

def cnn_train(model, train_loader, dev_loader, test_loader, text_field, label_field):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = config.model_args["epoch_num"]
    for epoch in range(1, epoch_num + 1):
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)
            # feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            model.train()
            pred = model(feature)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            optimizer.step()
        acc, size = cnn_eval(dev_loader, model)
        print("Accuracy: " + str(acc))
        if acc > 80:
            break
    acc, size = cnn_eval(test_loader, model)
    output_result(acc, size, config.result_path)
    save_model_and_fields(model, config.model_path, text_field, label_field, config.data_folder)

def rnn_eval(data_loader, model):
    model.eval()
    corrects = 0
    for batch_idx, batch in enumerate(data_loader):
        feature, target = batch.text, batch.label
        feature, target = feature.cuda(), target.cuda()
        outputs = model(feature)
        _,y_pred = torch.max(outputs, -1)
        corrects += torch.tensor(y_pred == target,dtype=torch.float).sum()

    size = len(data_loader.dataset)
    accuracy = 100.0 * corrects/size
    # print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy, size

def rnn_train(model, train_loader, dev_loader, test_loader, text_field, label_field):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_num = config.model_args["epoch_num"]
    for epoch in range(1, epoch_num + 1):
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            feature, target = batch.text, batch.label
            # feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            model.train()
            outputs = model(feature)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        acc, size = rnn_eval(dev_loader, model)
        print("Accuracy: " + str(acc))
        if acc > 80:
            break
    acc, size = rnn_eval(test_loader, model)
    output_result(acc, size, config.result_path)
    save_model_and_fields(model, config.model_path, text_field, label_field, config.data_folder)

            

if __name__ == "__main__":
    config = ModelConfig(sys.argv[1])
    train(config)