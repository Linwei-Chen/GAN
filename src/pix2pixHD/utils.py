import torch


def get_edges(t: torch.Tensor):
    edge = torch.zeros(t.shape).bool().to(t.device)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def label_to_one_hot(targets: torch.Tensor, n_class: int, with_255: bool = False):
    """
    get one-hot tensor from targets, ignore the 255 label
    :param targets: long tensor[bs, 1, h, w]
    :param nlabels: int
    :return: float tensor [bs, nlabel, h, w]
    """
    # batch_size, _, h, w = targets.size()
    # res = torch.zeros([batch_size, nlabels, h, w])
    targets = targets.squeeze(dim=1)
    # print(targets.shape)
    zeros = torch.zeros(targets.shape).long().to(targets.device)

    # del 255.
    targets_ignore = targets >= n_class
    # print(targets_ignore)
    targets = torch.where(targets < n_class, targets, zeros)

    one_hot = torch.nn.functional.one_hot(targets, num_classes=n_class)
    if with_255:
        one_hot[targets_ignore] = 0
    else:
        one_hot[targets_ignore] = 255
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


def get_encode_features(E: torch.nn.Module, imgs: torch.Tensor, instances: torch.Tensor, labels:torch.Tensor):
    """
    get instance-wise pooling feature from encoder, this function is also built in encode
    :param E:
    :param imgs:
    :param instances:
    :param labels:
    :return:
    """
    assert imgs.dim() == 4
    encode_features = E(imgs)
    batch_size = imgs.size(0)
    class_feature_dict = {}
    for b in range(batch_size):
        encode_feature = encode_features[b]
        instance = instances[b]
        label = labels[b]
        for i in instance.unique():
            mask = (instance == i).expand_as(encode_feature)
            cls = int(label[mask].unique())
            mean_feature = encode_feature[mask] / mask.float().sum()
            encode_feature[mask] = mean_feature
            if cls not in class_feature_dict:
                class_feature_dict[cls] = []
            class_feature_dict[cls].append(mean_feature.cpu().numpy())
    return encode_features, class_feature_dict
