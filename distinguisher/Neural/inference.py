import torchtext.data as data
from tqdm import tqdm

try:
    from distinguisher.Neural.util import *
    from distinguisher.Neural.model import *
except:
    from util import *
    from model import *

def infer(config: ModelConfig):
    text_field, label_field = load_fields()
    if config.model_arch == "CNN":
        input_dim = len(text_field.vocab)
        emb_dim = config.model_args["emb_dim"]
        n_filters = config.model_args["n_filters"]
        out_dim = len(label_field.vocab) - 1
        model = CNN(input_dim, emb_dim, n_filters, [2, 3, 4], out_dim).cuda()
        model = load_model(model, config.model_path)
    else:
        raise NotImplementedError()

    test_iter = get_test_loader(text_field, label_field, config.val_file, config.batch_size * 2, device = "cuda")
    corrects, avg_loss = 0, 0
    for batch_idx, batch in enumerate(test_iter):
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                    [1].view(target.size()).data == target.data).sum()

    size = len(test_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    output_result(accuracy, size, config.result_path)

if __name__ == "__main__":
    config = ModelConfig("/data/config.json")
    infer(config)