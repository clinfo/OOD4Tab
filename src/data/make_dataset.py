# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def main():
    # make dataset
    X,y = make_moons(n_samples=1000,random_state=0,noise=0.5)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)
    
    np.save("../../data/external/X_train",X_train)
    np.save("../../data/external/y_train",y_train)
    np.save("../../data/external/X_test",X_test)
    np.save("../../data/external/y_test",y_test)
    
    # visualize the data
    X_train_0 = X_train[y_train==0]
    X_train_1 = X_train[y_train==1]

    plt.scatter(X_train_0[:,0], X_train_0[:,1], c='r', label='0')
    plt.scatter(X_train_1[:,0], X_train_1[:,1], c='b', label='1')
    plt.grid()
    plt.title('moon sample training data')
    plt.savefig("../../reports/figures/Sample Training data.png")
    
if __name__ == "__main__":
    main()
