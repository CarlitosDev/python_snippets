'''
	ml_simons_regression_trees.py
	taken from https://github.com/simonwardjones/machine_learning/tree/master/machine_learning

'''


groups = self.split(feature_index, feature_split_val)


def split(self, feature_index, feature_split_val, only_y=True):
    """  
    Splits self.data on feature with index feature_index using
    feature_split_val.
    Each sample is included in left output if the feature value for
    the sample is less than or equal to the feature_split_val else 
    it is included in the right output
    Parameters:
    ----------
    feature_index: int
        Index of the feature (column) in self.data
    feature_split_val: float
        Feature value to use when splitting data
    only_y: bool, optional, default True
        Return only the y values in left and right - this is used 
        when checking candidate split purity increase
    Returns:
    -------
    (numpy.ndarray, numpy.ndarray):
        left and right splits of self.data
    """
    assert feature_index in range(self.data.shape[1])
    if only_y:
        select = -1
    else:
        select = slice(None)
    left_mask = self.data[:, feature_index] <= feature_split_val
    right_mask = ~ left_mask
    left = self.data[left_mask, select]
    right = self.data[right_mask, select]
    logger.debug(
        f'Splitting on feature_index {feature_index} with '
        f'feature_split_val = {feature_split_val} creates left '
        f'with shape {left.shape} and right with '
        f'shape {right.shape}')
    return left, right


def mean_square_impurity(self, groups):
    """  
    Calculates the mean square error impurity
    The mse impurity is the weighted average of the group variances
    Parameters:
    ----------
    groups: tuple
        The groups tuple is made up of arrays of values. It is 
        often called with groups = (left, right) to find the purity
        of the candidate split
    Returns:
    -------
    float:
        Mean square error impurity
    """
    mean_square_error = 0
    total_samples = sum(group.shape[0] for group in groups)
    for i, group in enumerate(groups):
        group_size = group.shape[0]
        group_mean = np.mean(group)
        group_mean_square_error = np.mean((group - group_mean) ** 2)
        mean_square_error += group_mean_square_error * \
            (group_size / total_samples)
        logger.debug(
            f'Group {i} has size {group.shape[0]} with '
            f'with MSE impurity {group_mean_square_error:.3}')
    logger.debug(f'MSE candidate {mean_square_error}')
    return mean_square_error