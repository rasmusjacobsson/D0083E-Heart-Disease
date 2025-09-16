from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt

def setup():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    Y = heart_disease.data.targets

    return X, Y

# Function to calculate R-squared values
def R_calc(X, Y):

    coeffs = np.polyfit(X, Y, 1)

    p = np.poly1d(coeffs)
    yhat = p(X)                         # or [p(z) for z in x]
    ybar = np.sum(Y)/len(Y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((Y - ybar)**2)     # or sum([ (yi - ybar)**2 for yi in y])
    r2 = ssreg / sstot

    return r2

def impute_missing(X):
    for column in X.columns:
        if X[column].isnull().any():
            mean_value = X[column].mean()
            X[column] = X[column].fillna(mean_value)    
    return X

def feature_correlation(X):
    r2 = {}
    columns = list(X.columns)
    for idx_i, i in enumerate(columns):
        for idx_j in range(idx_i + 1, len(columns)):
            j = columns[idx_j]
            r2[(i, j)] = R_calc(X[i], X[j])

    top_pairs = sorted(r2.items(), key=lambda item: item[1], reverse=True)[:3]

    for (i, j), r2_value in top_pairs:
        print(f"Top pair: {i} and {j} with R-squared: {r2_value}")

    for (i, j), r2_value in top_pairs:
        plt.figure()
        plt.scatter(X[i], X[j], alpha=0.5, label='Data')
        # Fit regression line
        coeffs = np.polyfit(X[i], X[j], 1)
        p = np.poly1d(coeffs)
        plt.plot(X[i], p(X[i]), color='red', label='Fit')
        plt.xlabel(i)
        plt.ylabel(j)
        plt.title(f'{i} vs {j} (R²={r2_value:.2f})')
        plt.legend()
    plt.show()


def main():
    X, Y = setup()
    X = impute_missing(X)
    Y = impute_missing(Y)

    r2 = {}
    for i in X.columns:
        r2[i] = R_calc(X[i], Y["num"])

    # Get top 3 features by R-squared
    top_features = [k for k, v in sorted(r2.items(), key=lambda item: item[1], reverse=True)[:3]]
    print("\nTop 3 features by R-squared:", top_features)

    # Plot each top feature vs target
    for feature in top_features:
        plt.figure()
        plt.scatter(X[feature], Y["num"], alpha=0.5, label='Data')
        # Fit regression line
        coeffs = np.polyfit(X[feature], Y["num"], 1)
        p = np.poly1d(coeffs)
        plt.plot(X[feature], p(X[feature]), color='red', label='Fit')
        plt.xlabel(feature)
        plt.ylabel('num')
        plt.title(f'{feature} vs num (R²={r2[feature]:.2f})')
        plt.legend()
    plt.show()

    feature_correlation(X)


if __name__ == "__main__":
    main()





