mutual_information.py

from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi



# Also https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429


# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html




https://www.wikiwand.com/en/Adjusted_mutual_information