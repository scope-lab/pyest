import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pyest.gm as pygm
from sklearn import datasets

# PLOTTING FUNCTIONS
def plot_iris(ax, iris, dimensions):
    """
    Plot the iris dataset on a 2D scatter plot given an existing figure axis.
    """
    d1 = dimensions[0]
    d2 = dimensions[1]
    data = iris.data
    scatter = ax.scatter(data[:, d1], data[:, d2], c=iris.target)
    ax.set(xlabel=iris.feature_names[d1], ylabel=iris.feature_names[d2])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )
    return scatter

def plot_contour(ax, p, X, Y, data, dimensions, xlabel, ylabel, legend, title):
    """
    Plot a contour plot of a 2D Gaussian mixture model on top of the iris datasetgiven an existing figure axis.
    """
    ax.contour(X, Y, p)
    scatter = plot_iris(ax, data, dimensions)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    ax.set(xlabel=xlabel, ylabel = ylabel)
    _ = ax.legend(
        scatter.legend_elements()[0], legend, loc="lower right", title="Classes"
    )

def plot_PyestGMM(pyest_gmm, data, dimensions, xlabel, ylabel, legend, title):
    """
    Plot a PyEst-type Gaussian mixture model on top of specified dimensions of the iris dataset.
    """
    # Create figure and axis
    _, ax = plt.subplots()
    # Loop over GMMs and plot each mixand contour onto the axis
    for i in range(len(pyest_gmm)):
        gmm = pygm.GaussianMixture(pyest_gmm[i][0], pyest_gmm[i][1], pyest_gmm[i][2])
        p, X, Y = gmm.pdf_2d(dimensions=dimensions, res=100)
        plot_contour(ax, p, X, Y, data, dimensions, xlabel, ylabel, legend, title)


def main():
    # Load iris dataset
    iris = datasets.load_iris()

    # Setting parameters
    dimensions = [0, 1]     # We will only look at the first 2 dimensions of the Iris dataset.
    seed = 1
    np.random.seed(seed)  # for reproducibility

    # Pretty formatting of printing matrices
    np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"})

    # Use scikit-learn's GMM to fit the data to a GMM
    p_sk = sk.mixture.GaussianMixture(n_components=1, covariance_type='full').fit(iris.data)

    # Use the results from the sklearn GMM to create a PyEst GMM
    pyest_gmm = pygm.GaussianMixture(p_sk.weights_, p_sk.means_, p_sk.covariances_)

    # Plot first GMM fit with 1 mixand over the iris dataset
    plot_PyestGMM(pyest_gmm, iris, dimensions,
        xlabel = iris.feature_names[0],
        ylabel = iris.feature_names[1],
        legend = iris.target_names[0],
        title = "PyEst GMM Iteration 1")
    plt.show()

    # Use PyEst split_gaussian to split the Gaussian mixture into two components
    split_options = pygm.GaussSplitOptions(L=2, lam=0.9)    # Set number of components and lambda parameter
    split_comp = pygm.split_gaussian(*pyest_gmm.pop(0), split_options)  # Perform split and remove the old GMM component
    pyest_gmm = split_comp # Update PyEst GMM to contain the new split mixands

    ''' NOTE: PyEst GMMs do NOT support GMMs with 0 mixands.
    To avoid this error, be sure to add a mixand before removing one from the mixture. '''

    # Plot PyEst GMM over iris dataset
    plot_PyestGMM(pyest_gmm, iris, dimensions,
        xlabel = iris.feature_names[0],
        ylabel = iris.feature_names[1],
        legend = iris.target_names[0],
        title = "PyEst GMM Iteration 2")
    plt.show()

if __name__ == "__main__":
    main()
