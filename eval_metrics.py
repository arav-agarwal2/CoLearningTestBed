import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(truth, pred):
    return accuracy_score(truth.cpu().numpy(), pred.cpu().numpy())


def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return accuracy_score(binary_truth, binary_preds)


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_mosi(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    # Average L1 distance between preds and truths
    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0),
                       (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return mae, corr, mult_a7, f_score, accuracy_score(binary_truth, binary_preds)
