import torch
from torch import nn
from ..utils.detection import cxcy_to_xyxy
from ..utils.detection import get_jaccard_overlaps
from ..utils.detection import xyxy_to_gcxgcy

class MultiBoxLoss(nn.Module):

    def __init__(self, priors, variances):
        super(MultiBoxLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # priors in cxcy format
        self.priors = priors.to(self.device)
        self.variances = variances
        self.min_overlap = 0.5
        # Loss functions
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predictions, targets):

        predict_locs, predict_confs = predictions

        batch_size = predict_locs.size(0)
        n_priors = self.priors.size(0)
        n_classes = predict_confs.size(2)

        assert n_priors == predict_locs.size(1) == predict_confs.size(1)

        true_locs = torch.Tensor(batch_size, n_priors, 4).to(self.device)
        true_confs = torch.LongTensor(batch_size, n_priors).to(self.device)

        for i in range(batch_size):
            # Number of objects in the image
            n_objects = targets[i].size(0)
            overlaps = get_jaccard_overlaps(targets[i][:, :-1], cxcy_to_xyxy(self.priors)) # (n_objects, n_priors)

            # The best ground truth for each prior
            prior_overlaps, prior_objects = overlaps.max(dim=0)
            # The best prior for each ground truth
            object_overlaps, object_priors = overlaps.max(dim=1)

            # Assign best ground truth to each prior
            prior_objects[object_priors] = torch.LongTensor(range(n_objects)).to(self.device)
            prior_overlaps[object_priors] = 1

            prior_labels = targets[i][:, -1][prior_objects]
            prior_labels[prior_overlaps < self.min_overlap] = 0

            true_confs[i] = prior_labels
            true_locs[i] = xyxy_to_gcxgcy(targets[i][:, :-1][prior_objects], self.priors, self.variances)

        positive_priors = true_confs != 0

        # Localization loss
        loc_loss = self.smooth_l1(predict_locs[positive_priors], true_locs[positive_priors])

        # Hard negative mining
        n_positives = positive_priors.sum(dim=1)
        n_negatives = 3 * n_positives

        # Confidence loss for all priors
        conf_loss_all = self.cross_entropy(predict_confs.view(-1, n_classes), true_confs.view(-1))
        # (N, n_priors)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)

        # Confidence loss for positive priors
        conf_loss_positive = conf_loss_all[positive_priors]
        # Find hard negatives (highest loss)
        conf_loss_negative = conf_loss_all.clone()
        conf_loss_negative[positive_priors] = 0
        # Sort confidence loss in descending order
        _, loss_idx = conf_loss_negative.sort(dim=1, descending=True) # (N, n_priors)
        _, idx_rank = loss_idx.sort(1)
        hard_negatives = idx_rank < n_negatives.unsqueeze(1)
        conf_loss_hard_negative = conf_loss_negative[hard_negatives]

        conf_loss = (conf_loss_hard_negative.sum() + conf_loss_positive.sum()) / n_positives.sum().float()

        return conf_loss + loc_loss
