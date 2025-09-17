from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

winsorLower = 2
winsorUpper = 98

def setup():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # Check if data was loaded correctly
    if heart_disease.data is None:
        raise ValueError("Failed to fetch dataset. Please check your internet connection or dataset ID.")

    # data (as pandas dataframes)
    X = heart_disease.data.features
    Y = heart_disease.data.targets

    return X, Y

# impute missing values with mean
def impute_mean(x):
    for col in x.columns:
        if x[col].isnull().sum() > 0:
            mean_value = x[col].mean()
            x[col] = x[col].fillna(mean_value)
    return x

# impute missing values with median 
def impute_median(x):
    for col in x.columns:
        if x[col].isnull().sum() > 0:
            median_value = x[col].median()
            x[col] = x[col].fillna(median_value)
    return x

# winsorization for outlier removal
# basically sets all outliers to the nearest non-outlier value
def outlier_removal(x):
    for col in x.columns:
        lower_bound = x[col].quantile(winsorLower/100)
        upper_bound = x[col].quantile(winsorUpper/100)
        x[col] = np.where(x[col] < lower_bound, lower_bound, x[col])
        x[col] = np.where(x[col] > upper_bound, upper_bound, x[col])
    return x

# boxplots for all the features
def plot_boxplots(x):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=x)
    plt.title("Boxplots for all features")
    plt.xticks(rotation=45)

# histograms for all the features
def plot_histograms(x):
    x.hist(bins=15, figsize=(8, 6), layout=(4, 4))
    plt.suptitle("Histograms for all features")
    plt.tight_layout()

# kernel density estimates for all the features
def plot_kernel_density(x):
    num_features = len(x.columns)
    ncols = 4
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2*nrows))
    axes = axes.flatten()
    for idx, col in enumerate(x.columns):
        sns.kdeplot(x[col], ax=axes[idx])
        axes[idx].set_title(col)
    # Hide unused axes
    for idx in range(num_features, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("Kernel Density Estimates for all features")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

# Q-Q plots for all the features
def plot_qqplots(x):
    num_features = len(x.columns)
    ncols = 4
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2*nrows))
    axes = axes.flatten()
    for idx, col in enumerate(x.columns):
        stats.probplot(x[col], dist="norm", plot=axes[idx])
        axes[idx].set_title(col)
    # Hide unused axes
    for idx in range(num_features, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("Q-Q Plots for all features")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

# correlation heatmap for all features
def plot_correlation_heatmap(x):
    plt.figure(figsize=(10, 8))
    corr = x.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap")
    plt.tight_layout()

# calculate R² score based on a linear fit
def r2_score(x, y):
    coeffs = np.polyfit(x, y, 1)

    p = np.poly1d(coeffs)
    yhat = p(x)                         
    ybar = np.sum(y)/len(y)          
    ssreg = np.sum((yhat - ybar)**2)  
    sstot = np.sum((y - ybar)**2)     
    r2 = ssreg / sstot

    return r2

# R² score between each feature and the target
def r2_score_target(x, y):
    r2 = {}
    for i in x.columns:
        r2[i] = r2_score(x[i], y["quality"])

    return r2

# R² score between each pair of features
def r2_score_features(x):
    r2 = {}
    cols = list(x.columns)
    for idx_i, i in enumerate(cols):
        for idx_j in range(idx_i + 1, len(cols)):
            j = cols[idx_j]
            r2[(i, j)] = r2_score(x[i], x[j])
    return r2

def main():
    x_raw, y_raw = setup()
    x_imputed_median = impute_median(x_raw.copy())
    x_cleaned = outlier_removal(x_imputed_median.copy())

    plot_correlation_heatmap(x_raw)
    plt.show(block=False)
    plot_correlation_heatmap(x_cleaned)
    plt.show()

if __name__ == "__main__":
    main()
