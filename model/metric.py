import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def MAE(output, target, device='cpu'):
    with torch.no_grad():
        pred = torch.sigmoid(output)
        assert pred.shape == target.shape, "Predictions and ground-truths should be of the same sizes."
        mae = torch.abs(pred - target).mean(dim=[2,3])
        mae[mae != mae] = 0 # for Nan
    return mae.mean().item()
    
# def Fmeasure(output, target, beta2 = 0.3, device='cpu'):
#     avg_f, img_num = 0.0, 0.0

#     with torch.no_grad():
#         pred = torch.sigmoid(output)
#         assert pred.shape == target.shape, "Predictions and ground-truths should be of the same sizes."
#         for pred_, target_ in zip(pred, target):
#             pred_, target_ = pred_.unsqueeze(0), target_.unsqueeze(0)
#             prec, recall = _eval_pr(pred_, target_, 255, device)
#             f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
#             f_score[f_score != f_score] = 0 # for Nan
#             avg_f += f_score
#             img_num += 1.0
#             score = avg_f / img_num
#         return score.max().item()

def Smeasure(output, target, alpha=0.5, device='cpu'):
    with torch.no_grad():
        pred = torch.sigmoid(output)
        assert pred.shape == target.shape, "Predictions and ground-truths should be of the same sizes."
        avg_q = 0.0
        for pred_, target_ in zip(pred, target):
            pred_, target_ = pred_.unsqueeze(0), target_.unsqueeze(0)
            y = target_.mean()
            if y == 0:
                x = pred_.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred_.mean()
                Q = x
            else:
                target_[target_ >= 0.5] = 1
                target_[target_ < 0.5] = 0
                Q = alpha * _S_object(pred_, target_) + (1 - alpha) * _S_region(pred_, target_, device)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            avg_q += Q.item()
        return avg_q / len(target)
 
# def _eval_pr(y_pred, y, num, device):
#     prec, recall = torch.zeros(num).to(device), torch.zeros(num).to(device)
#     thlist = torch.linspace(0, 1 - 1e-10, num).to(device)
#     for i in range(num):
#         y_temp = (y_pred >= thlist[i]).float()
#         tp = (y_temp * y).sum()
#         prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
#     return prec, recall

def _S_object(pred, gt):
    fg = torch.where(gt==0, torch.zeros_like(pred), pred)
    bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1-gt)
    u = gt.mean()
    Q = u * o_fg + (1-u) * o_bg
    return Q

def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    
    return score

def _S_region(pred, gt, device):
    X, Y = _centroid(gt, device)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
    return Q

def _centroid(gt, device):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1).to(device) * round(cols / 2)
        Y = torch.eye(1).to(device) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.arange(0,cols).to(device).float()
        j = torch.arange(0,rows).to(device).float()
        X = torch.round((gt.sum(dim=0)*i).sum() / total)
        Y = torch.round((gt.sum(dim=1)*j).sum() / total)
    return X.long(), Y.long()

def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h*w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h*w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
    
    aplha = 4 * x * y *sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q
