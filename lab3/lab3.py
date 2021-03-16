from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "image"

class MNIST5:
    def __init__(self):
        self.X, self.y = self.load_and_sort()
        self.X_train, self.y_train = self.X[:60000], self.y[:60000]
        self.X_test, self.y_test = self.X[60000:], self.y[60000:]
        self.sgd_clsf = None
        self.y_train_5 = None
        self.y_test_5 = None
        self.id = 1

    def save_fig(self, fig_id, tight_layout=True):
        path = os.path.join(PROJECT_ROOT_DIR, IMAGE_DIR, str(fig_id) + ".png")
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format='png', dpi=300)


    def random_digit(self, index):
        some_digit = self.X[index]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap=mpl.cm.binary,
               interpolation="nearest")
        plt.axis("off")
        
        self.save_fig(self.id)
        self.id = self.id + 1
        plt.show()
        return some_digit


    def load_and_sort(self):
        try:
            from sklearn.datasets import fetch_openml
            # fetch_openml() returns targets as strings
            mnist = fetch_openml('mnist_784', version=1,as_frame=False, cache=True)
            mnist.target = mnist.target.astype(np.int8)
            self.sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
        except ImportError:
            from sklearn.datasets import fetch_mldata
            mnist = fetch_mldata('MNIST original')
        return mnist["data"], mnist["target"]


    def sort_by_target(self, mnist):
        reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
        reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    
        mnist.data[:60000] = mnist.data[reorder_train]
        mnist.target[:60000] = mnist.target[reorder_train]
        mnist.data[60000:] = mnist.data[reorder_test + 60000]
        mnist.target[60000:] = mnist.target[reorder_test + 60000]


    def train_predict(self, some_digit):
        shuffle_index = np.random.permutation(60000)
        self.X_train, self.y_train = self.X_train[shuffle_index], self.y_train[shuffle_index]

        # Binary number 5 Classifier
        self.y_train_5 = (self.y_train == 5)
        self.y_test_5 = (self.y_test == 5)
    

        # print prediction result of the given input some_digit
        self.sgd_clsf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
        self.sgd_clsf.fit(self.X_train, self.y_train_5)
        print("Prediction result (is 5): ", self.sgd_clsf.predict([some_digit]))
        self.calculate_cross_val_score(self.X_train, self.y_train_5)


    def calculate_cross_val_score(self, X, Y):
        score = cross_val_score(self.sgd_clsf, X, Y, cv=4, scoring="accuracy")
        print("Cross valdation score: ", score)
        
    def predict(self, v):
        return self.sgd_clsf.predict([v])
    


if __name__ == "__main__":
    clsf5 = MNIST5()
    some_digit = clsf5.random_digit(36000)
    clsf5.train_predict(some_digit)
    
    some_digit = clsf5.random_digit(65000)
    print("Actual result: ", clsf5.y[65000])
    print("Prediction result (is 5): ", clsf5.predict(some_digit))
    
    some_digit = clsf5.random_digit(61000)
    print("Actual result: ", clsf5.y[61000])
    print("Prediction result (is 5): ", clsf5.predict(some_digit))
    
    some_digit = clsf5.random_digit(35000)
    print("Actual result: ", clsf5.y[35000])
    print("Prediction result (is 5): ", clsf5.predict(some_digit))
