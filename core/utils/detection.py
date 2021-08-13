import torch

def xyxy_to_cxcy(xyxy):
    return torch.cat([(xyxy[:, 2:] + xyxy[:, :2]) / 2, xyxy[:, 2:] - xyxy[:, :2]], 1)

def cxcy_to_xyxy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)

def xyxy_to_gcxgcy(xyxy, priors, variances=[0.1, 0.2]):
    gcxcy = (xyxy[:, :2] + xyxy[:, 2:]) / 2 - priors[:, :2]
    gcxcy = gcxcy / (priors[:, 2:] * variances[0])
    gwh = (xyxy[:, 2:] - xyxy[:, :2]) / priors[:, 2:]
    gwh = torch.log(gwh) / variances[1]
    return torch.cat([gcxcy, gwh], 1)

def gcxgcy_to_xyxy(gcxgcy, priors, variances=[0.1, 0.2]):
    xyxy = torch.cat([
        priors[:, :2] + gcxgcy[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(gcxgcy[:, 2:] * variances[1])], 1)
    xyxy[:, :2] -= xyxy[:, 2:] / 2
    xyxy[:, 2:] += xyxy[:, :2]
    return xyxy

def cxcy_to_gcxgcy(cxcy, priors, variances=[0.1, 0.2]):
    gcxcy = (cxcy[:, :2] - priors[:, :2]) / (priors[:, 2:] * variances[0])
    gwh = torch.log(cxcy[:, 2:] / priors[:, 2:]) / variances[1]
    return torch.cat([gcxcy, gwh], 1)

def get_intersections(xyxy1, xyxy2):
    # set_1: (n_xyxy1, 4) set_2: (n_xyxy2, 4)
    max_xy = torch.min(xyxy1[:, 2:].unsqueeze(1), xyxy2[:, 2:].unsqueeze(0)) # (n_xyxy1, n_xyxy2, 2)
    min_xy = torch.max(xyxy1[:, :2].unsqueeze(1), xyxy2[:, :2].unsqueeze(0)) # (n_xyxy1, n_xyxy2, 2)
    intersections = torch.clamp(max_xy - min_xy, min=0) # (n_xyxy1, n_xyxy2, 2)
    # Return A intersect B
    return intersections[..., 0] * intersections[..., 1]

def get_jaccard_overlaps(xyxy1, xyxy2):
    # set_1: (n_xyxy1, 4) set_2: (n_xyxy2, 4)
    intersections = get_intersections(xyxy1, xyxy2)
    set1_areas = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
    set2_areas = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])
    unions = set1_areas.unsqueeze(1) + set2_areas.unsqueeze(0) - intersections
    return intersections / unions

def nms(xyxy, confs, max_overlap=0.2):
    overlaps = get_jaccard_overlaps(xyxy, xyxy)
    suppress = torch.zeros_like(confs, dtype=torch.bool)
    for bbox in range(xyxy.size(0)):
        # Skip boxes already marked for suppression
        if suppress[bbox]:
            continue
        suppress = torch.logical_or(suppress, overlaps[bbox] > max_overlap)
        suppress[bbox] = False
    return xyxy[~suppress], confs[~suppress]
