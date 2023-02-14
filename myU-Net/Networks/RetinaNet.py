import Networks.utilsmrcnn.model_utils as mutils
import sys
sys.path.append('../')
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

def nms_3D(boxes, scores, overlap=0.5):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,6].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.zeros(scores.size()[0],dtype=torch.long).to("cuda")
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    z1 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = torch.mul(x2 - x1 + 1, y2 - y1 + 1)
    volume = torch.mul(area,z2-z1 + 1)
    idx = (scores.sort(0, descending=True)[1])
    count = torch.LongTensor(1).to("cuda")
    #keep = torch.arange(scores.size()[0]).to("cuda")
    while idx.numel() > 0:
        i = idx[0]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[1:]  # remove kept element from
        # load bboxes of next highest vals
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        zz1 = torch.index_select(z1, 0, idx)
        zz2 = torch.index_select(z2, 0, idx)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        zz1 = torch.clamp(zz1, min=z1[i])
        zz2 = torch.clamp(zz2, max=z2[i])
        w = xx2 - xx1
        h = yy2 - yy1
        z = zz2 - zz1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        z = torch.clamp(z, min=0.0)
        print(w,h,z)
        inter = w*h*z
        print(inter)
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(volume, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + volume[i]
        IoU = (inter.float() + 1e-10)/(union.float() + 1e-10)  # store result in iou
        print(IoU)
        exit()
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    print(idx[keep[:count[0]]])

    return idx[keep[:count[0]].cuda()].contiguous() #torch.unique(keep)

############################################################
# Networks on top of backbone
############################################################
class ResBlock(nn.Module):

    def __init__(self, start_filts, planes, conv, stride=1, downsample=None, norm=None, relu='relu'):
        super(ResBlock, self).__init__()
        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, planes * 4, ks=1, norm=norm, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], norm=norm, relu=None)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, conv, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts = 18
        start_filts = self.start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}['resnet50'], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = False
        self.dim = conv.dim

        if operate_stride1:
            self.C0 = nn.Sequential(conv(1, start_filts, ks=3, pad=1, norm=None, relu='relu'),
                                    conv(start_filts, start_filts, ks=3, pad=1, norm=None, relu='relu'))

            self.C1 = conv(start_filts, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=None, relu='relu')

        else:
            self.C1 = conv(1, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=None, relu='relu')

        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                         if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, norm=None, relu='relu',
                                    downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=None, relu='relu'))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=None, relu='relu',
                                    downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=None, relu='relu'))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=None, relu='relu', downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=None, relu='relu'))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=None, relu='relu', downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=None, relu='relu'))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(
                start_filts_exp * 8, start_filts * 16, conv=conv, stride=2, norm=None, relu='relu', downsample=(start_filts_exp * 8, 2, 2)))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(start_filts_exp * 16, start_filts * 16, conv=conv, norm=None, relu='relu'))
            self.C6 = nn.Sequential(*C6_layers)

        if conv.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        self.out_channels = 36
        self.P5_conv1 = conv(start_filts*32 + 0, self.out_channels, ks=1, stride=1, relu=None) #
        self.P4_conv1 = conv(start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
        #     print ("encoder shapes:", ii.shape)
        #
        # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
        #     print("decoder shapes:", ii.shape)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list

class RPN(nn.Module):
    """
    Region Proposal Network.
    """

    def __init__(self, conv):

        super(RPN, self).__init__()
        self.dim = conv.dim

        self.conv_shared = conv(36, 128, ks=3, stride={'xy': [[8], [16], [32], [64]], 'z': [[4], [8], [16], [32]]}, pad=1, relu='relu')
        self.conv_class = conv(128, 2 * len([0.5, 1, 2]), ks=1, stride=1, relu=None)
        self.conv_bbox = conv(128, 2 * self.dim * len([0.5, 1, 2]), ks=1, stride=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :return: rpn_class_logits (b, 2, n_anchors)
        :return: rpn_probs_logits (b, 2, n_anchors)
        :return: rpn_bbox (b, 2 * dim, n_anchors)
        """

        # Shared convolutional base of the RPN.
        x = self.conv_shared(x)

        # Anchor Score. (batch, anchors per location * 2, y, x, (z)).
        rpn_class_logits = self.conv_class(x)
        # Reshape to (batch, 2, anchors)
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        rpn_class_logits = rpn_class_logits.permute(*axes)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension (fg vs. bg).
        rpn_probs = F.softmax(rpn_class_logits, dim=2)

        # Bounding box refinement. (batch, anchors_per_location * (y, x, (z), log(h), log(w), (log(d)), y, x, (z))
        rpn_bbox = self.conv_bbox(x)

        # Reshape to (batch, 2*dim, anchors)
        rpn_bbox = rpn_bbox.permute(*axes)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, self.dim * 2)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Network Heads
############################################################

class Classifier(nn.Module):


    def __init__(self, conv):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.n_classes = 2
        n_input_channels = 36
        n_features = 64
        n_output_channels = 18
        anchor_stride = 1

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)

        return [class_logits]



class BBRegressor(nn.Module):


    def __init__(self, conv):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = 36
        n_features = 64
        n_output_channels = 9 * self.dim * 2
        anchor_stride = 1

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu='relu')
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride,
                               pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        return [bb_logits]

############################################################
#  Loss Functions
############################################################

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices]
        print(roi_logits_pos, targets_pos)
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).to("cuda")

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.size()[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
        print(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]))
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).to("cuda"))
        # return the indices of negative samples, which contributed to the loss (for monitoring plots).
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).to("cuda")
        np_neg_ix = np.array([]).astype('int32')
    print(pos_loss, neg_loss)
    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        print(pred_deltas,target_deltas)
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).to("cuda")

    return loss

############################################################
#  Output Handler
############################################################

def refine_detections(anchors, probs, deltas, batch_ixs, patch_size):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    anchors = anchors.repeat(len(torch.unique(batch_ixs)), 1)
    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:6000]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat((torch.div(keep_ix,fg_probs.shape[1],rounding_mode='floor').unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)
    pre_nms_scores = flat_probs[:6000]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).type(th.long).to("cuda")

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]), [1, 3 * 2])).type(th.float).to("cuda")
    scale = torch.from_numpy(np.array([patch_size[0], patch_size[1], patch_size[0], patch_size[1],patch_size[2],patch_size[2]])).type(th.float).to("cuda")
    print(pre_nms_deltas[0])
    refined_rois = mutils.apply_box_deltas_3D(torch.div(pre_nms_anchors,scale,rounding_mode='floor'), pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = mutils.clip_to_window(np.array([0, 0, patch_size[0], patch_size[1],0,patch_size[2]]), refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    print(pre_nms_rois[0])
    for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]
            ix_scores = ix_scores

            class_keep = order[:2].contiguous() #nms_3D(ix_rois, ix_scores, 1e-5) #in case change with 0.7
            # map indices back.

            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:30]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep
    print(pre_nms_rois[keep][0])
    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result



def get_results(img_shape, detections, seg_logits, box_results_list=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                          retina_unet and dummy array for retina_net.
    """
    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, 6]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    #for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:

            boxes = detections[ix][:, :2 * 3].astype(np.int32)
            class_ids = detections[ix][:, 2 * 3 + 1].astype(np.int32)
            scores = detections[ix][:, 2 * 3 + 2]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)

            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    if score >= 0.1:
                        box_results_list[ix].append({'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_type': 'det',
                                                     'box_pred_class_id': class_ids[ix2]})

    results_dict = {'boxes': box_results_list}
    if seg_logits is None:
        # output dummy segmentation for retina_net.
        results_dict['seg_preds'] = np.zeros(img_shape)[:, 0][:, np.newaxis]
    else:
        # output label maps for retina_unet.
        results_dict['seg_preds'] = F.softmax(seg_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis].astype('uint8')

    return results_dict

############################################################
#  Retina U-Net Class
############################################################

class net(nn.Module):


    def __init__(self, patch_size):

        super(net, self).__init__()
        self.patch_size = patch_size
        self.build()

    def build(self):
        """
        Build Retina Net architecture.
        """

        # Image size must be dividable by 2 multiple times.
        h, w = self.patch_size[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # instanciate abstract multi dimensional conv class and backbone model.
        conv = mutils.NDConvGenerator(3)

        # build Anchors, FPN, Classifier / Bbox-Regressor -head
        self.np_anchors = mutils.generate_pyramid_anchors(self.patch_size)
        self.anchors = torch.from_numpy(self.np_anchors).float().to("cuda")
        self.Fpn = FPN(conv, operate_stride1=True)
        self.Classifier = Classifier(conv)
        self.BBRegressor = BBRegressor(conv)
        self.final_conv = conv(36, 2, ks=1, pad=0, norm=None, relu=None)


    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        axes = (0, 1, 3, 4, 2)
        img = img.permute(*axes)
        gt_class_ids = batch['roi_labels']
        gt_boxes = batch['bb_target']
        var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(batch['seg'], 2)).to("cuda")
        var_seg = torch.LongTensor(batch['seg']).to("cuda")

        batch_class_loss = torch.FloatTensor([0]).to("cuda")
        batch_bbox_loss = torch.FloatTensor([0]).to("cuda")

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        self.Fpn.train(),self.final_conv.train(),self.Classifier.train(),self.BBRegressor.train()
        detections, class_logits, pred_deltas, seg_logits = self.forward(img)
        # loop over batch
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': batch['bb_target'][b][ix],
                                                'box_label': batch['roi_labels'][b][ix], 'box_type': 'gt'})
                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas = mutils.gt_anchor_matching(self.np_anchors, gt_boxes[b], gt_class_ids[b])
                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            anchor_class_match = torch.from_numpy(anchor_class_match).to("cuda")
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().to("cuda")
            # compute losses.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss / img.shape[0]
        seg_loss_dice = 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1),var_seg_ohe)
        seg_loss_ce = F.cross_entropy(seg_logits, var_seg[:, 0])
        loss = batch_class_loss + batch_bbox_loss + (seg_loss_dice + seg_loss_ce) / 2
        print('loss', loss.item(), 'class_loss', batch_class_loss.item())
        print(batch_bbox_loss.item(), seg_loss_dice.item(),seg_loss_ce.item())

        return loss


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                            retina_unet and dummy array for retina_net.
        """
        img = batch['data']
        self.Fpn.eval(),self.final_conv.eval(),self.Classifier.eval(),self.BBRegressor.eval()
        std_dev = torch.from_numpy(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])).type(th.float).to(
            "cuda")
        scale = torch.from_numpy(
            np.array(
                [self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1], self.patch_size[2],
                 self.patch_size[2]])).type(
            th.float).to("cuda")
        refined_rois = torch.zeros((img.size()[0],6)).to("cuda")
        classes = torch.zeros(img.size()[0]).to("cuda")
        with torch.no_grad():
            _, class_logits, bb_outputs, seg_logits = self.forward(img)#,test=True)
            class_ = F.softmax(class_logits,img.size()[0])
            print(class_logits.max())
            for b in range(img.size()[0]):
                class__ = torch.argwhere(class_[b,:,1]>class_[b,:,0])
                if class__.size()[0]!=0:
                    classe = torch.index_select(class_,1,class__.view(-1))
                    bb = torch.index_select(bb_outputs,1,class__.view(-1))
                    anc = torch.index_select(self.anchors,0,class__.view(-1))
                    print(self.anchors.size())
                    classe, order = classe[b,:,1].sort(0, descending=True)
                    bb_final =bb[b,order[0]]
                    anc_final = anc[order[0]]
                    classes[b] = classe[0]
                    print(classes)
                    refined_rois[b] = mutils.apply_box_deltas_results_3D(torch.div(anc_final,scale,rounding_mode='floor'),bb_final * std_dev,self.patch_size) * scale
                    print(refined_rois)
                else:
                    classes[b] = 0
                    refined_rois[b] = torch.tensor([0,0,128,128,0,128]).to("cuda")
        #results_dict = get_results(img.shape, detections, seg_logits)
        return classes, refined_rois


    def forward(self, img, test=False):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # Feature extraction
        fpn_outs = self.Fpn(img)
        seg_logits = self.final_conv(fpn_outs[0])
        selected_fmaps = [fpn_outs[i + 1] for i in [0, 1, 2, 3]]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.Classifier(p))
            bb_reg_layer_outputs.append(self.BBRegressor(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]
        print(class_logits.size(),bb_outputs.size())
        # merge batch_dimension and store info in batch_ixs for re-allocation.
        if test:
            batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).to(
                "cuda")
            flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
            flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
            detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, self.patch_size)
        else:
            detections = None

        return detections, class_logits, bb_outputs, seg_logits