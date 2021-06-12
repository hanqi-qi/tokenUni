from scipy.stats import ttest_ind
a = np.array([1, 3, 4, 6, 11, 13, 15, 19, 22, 24, 25, 26, 26])
b = np.array([2, 4, 6, 9, 11, 13, 14, 15, 18, 19, 21])
ttest_ind(a, b)