import numpy as np


def is_go_or_nogo_loss(_loss_samples, _threshold_loss):
    """
    For a given list of losses associated with experiments return a go/no go
    status depending on whether the loss of an experiment is below/above a
    threshold loss
    :param _loss_samples: loss associated with multiple different experiments (list)
    :param _threshold_loss: threshold below which the loss must be in order for a feature to be rolled out
    :return: list of booleans
    """
    return [
        loss < _threshold_loss for loss in _loss_samples
    ]


def calculate_accuracy(_threshold_loss_go_or_nogo, _uplift_go_or_nogo):
    """
    Calculate the accuracy associateed with rolling out features based on their loss
    vs the true uplift
    :param _threshold_loss_go_or_nogo: list of booleans representing whether an experiment had a loss below
        a set threshold
    :param _uplift_go_or_nogo: list of booleans representing whether an experiment had an uplift above 0
    :return: float
    """
    return np.mean([
        loss_go == uplift_go
        for loss_go, uplift_go in zip(_threshold_loss_go_or_nogo, _uplift_go_or_nogo)
    ])


def calculate_precision(_threshold_loss_go_or_nogo, _uplift_go_or_nogo):
    """
    Calculate the precision associateed with rolling out features based on their loss
    vs the true uplift
    :param _threshold_loss_go_or_nogo: list of booleans representing whether an experiment had a loss below
        a set threshold
    :param _uplift_go_or_nogo: list of booleans representing whether an experiment had an uplift above 0
    :return: float
    """
    return np.mean([
        uplift_go
        for loss_go, uplift_go in zip(_threshold_loss_go_or_nogo, _uplift_go_or_nogo)
        if loss_go
    ])


def calculate_recall(_threshold_loss_go_or_nogo, _uplift_go_or_nogo):
    """
    Calculate the recall associateed with rolling out features based on their loss
    vs the true uplift
    :param _threshold_loss_go_or_nogo: list of booleans representing whether an experiment had a loss below
        a set threshold
    :param _uplift_go_or_nogo: list of booleans representing whether an experiment had an uplift above 0
    :return: float
    """
    return np.mean([
        loss_go
        for loss_go, uplift_go in zip(_threshold_loss_go_or_nogo, _uplift_go_or_nogo)
        if uplift_go
    ])


def calculate_total_uplift(_go_or_nogo, _uplift):
    """
    Calculate total uplift across all experiments that we roll out
    :param _go_or_nogo: list of go/no go statuses
    :param _uplift: true uplift of variant
    :return: float
    """
    return np.sum([
        uplift_go
        for loss_go, uplift_go in zip(_go_or_nogo, _uplift)
        if loss_go
    ])