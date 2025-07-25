import sys
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from carla import log
from carla.models.api import MLModel
from carla.recourse_methods.autoencoder import CSVAE
from carla.recourse_methods.processing import reconstruct_encoding_constraints

import os
from carla.gpu import GPU_N
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_N


def compute_loss(cf_initialize, query_instance, target, i, lambda_param, mlmodel):
    loss_function = nn.BCELoss()
    output = mlmodel.predict_proba(cf_initialize)
    loss1 = loss_function(output, target)  # classification loss
    loss2 = torch.sum((cf_initialize - query_instance) ** 2)  # distance loss
    total_loss = loss1 + lambda_param * loss2
    return total_loss


def counterfactual_search(
    mlmodel: MLModel,
    csvae: CSVAE,
    factual: np.ndarray,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    target_class: Optional[List] = None,
    lambda_param: float = 0.001,
    optimizer: str = "RMSprop",
    lr: float = 0.008,
    max_iter: int = 2000,
) -> np.ndarray:
    if torch.backends.mps.is_available():
        device = torch.device("mps")  
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")
    if target_class is None:
        target_class = [0, 1]

    x_train = factual[:, :-1]
    y_train = np.zeros((factual.shape[0], 2))
    y_train[:, 0] = 1 - factual[:, -1]
    y_train[:, 1] = factual[:, -1]

    x, _, zw, _, w, _, _, _, z, _ = csvae.forward(
        torch.from_numpy(x_train).float().to(device),
        torch.from_numpy(y_train).float().to(device),
    )

    # only w is trainable
    w = torch.rand(2, requires_grad=True, dtype=torch.float, device=device)

    target = torch.FloatTensor(np.array(target_class)).to(device)
    target_prediction = torch.argmax(target).to(device)

    if optimizer == "RMSprop":
        optim = torch.optim.RMSprop([w], lr)
    else:
        optim = torch.optim.Adam([w], lr)

    counterfactuals = []  # all possible counterfactuals
    distances = (
        []
    )  # distance of the possible counterfactuals from the intial value - considering distance as the loss function (can even change it just the distance)
    all_loss = []

    query_instance = torch.FloatTensor(x_train).to(device)

    # add the immutable features to the latents
    z = torch.cat([z, query_instance[:, ~csvae.mutable_mask]], dim=-1)

    for j in range(max_iter):
        cf, _ = csvae.p_x(z, w.unsqueeze(0))

        # add the immutable features to the reconstruction
        temp = query_instance.clone()
        temp[:, csvae.mutable_mask] = cf
        cf = temp

        cf = reconstruct_encoding_constraints(
            cf, cat_feature_indices, binary_cat_features
        ).to(device)
        output = mlmodel.predict_proba(cf)
        _, predicted = torch.max(output[0], 0)

        if predicted == target_prediction:
            counterfactuals.append(torch.cat((cf, predicted.reshape((-1, 1))), dim=-1))

        loss = compute_loss(
            cf, query_instance, target.unsqueeze(0), j, lambda_param, mlmodel
        )
        all_loss.append(loss)
        if predicted == target_prediction:
            distances.append(loss)
        loss.backward(retain_graph=True)
        optim.step()
        optim.zero_grad()
        cf.detach_()

    if not len(counterfactuals):
        # if no counterfactual is present, returing the last candidate
        log.debug("No counterfactual found")
        output = mlmodel.predict_proba(cf)
        _, predicted = torch.max(output[0], 0)
        return (
            torch.cat((cf, predicted.reshape((-1, 1))), dim=-1)
            .cpu()
            .detach()
            .numpy()
            .squeeze(axis=0)
        )

    # Choose the nearest counterfactual
    torch_counterfactuals = torch.stack(counterfactuals)
    torch_distances = torch.stack(distances)
    np_counterfactuals = torch_counterfactuals.cpu().detach().numpy()
    np_distances = torch_distances.cpu().detach().numpy()
    index = np.argmin(np_distances)
    log.debug("Counterfactual found")

    return np_counterfactuals[index].squeeze(axis=0)
