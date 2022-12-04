import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy import special as sp
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from scipy.misc import derivative

'''
Q function expressed in terms of the error function (https://en.wikipedia.org/wiki/Q-function).
'''
def _qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

"""
Theoretical symbol error probability
"""

p = lambda x,Es: np.exp(-Es)/(2*np.pi)*(1+np.exp(Es*(np.cos(x)**2))*np.sqrt(4*np.pi*Es)*np.cos(x)*(1-_qfunc(np.sqrt(2*Es)*np.cos(x))))

def theoretical_ser(mod, M, SNR_db, channel, Es = 1):
    if channel == 'awgn':
        if mod == 'PSK':
            Pe = 1 - quad(p, -np.pi/M, np.pi/M, args=(10**(SNR_db/10),))[0]
        else:
            SNR_l = 10**(SNR_db/10) #from dB to linear scale
            Pe = 4*(1-(1/np.sqrt(M)))*_qfunc(np.sqrt(3*SNR_l/(M-1))) \
                 - 4*(1-(1/np.sqrt(M)))**2 * _qfunc(np.sqrt(3*SNR_l/(M-1)))**2
    elif channel == 'rayleigh':
        def Prob_e(C, D):
            return (C / 2) * (1 - np.sqrt((D * Es / 2) / (1 + D * Es / 2)))
        if mod == 'PSK':
            Pe = Prob_e(2, 2 * np.log2(M) * np.sin(np.pi / M)**2)
        else:
            Pe = Prob_e(4 * (1 - 1 / np.sqrt(M)), 3 * np.log2(M) / (M - 1))
    
    return Pe
"""     
    if mod == 'PSK':
        if channel == 'awgn':
            Pe = 1 - quad(p, -np.pi/M, np.pi/M, args=(10**(SNR_db/10),))[0]
        elif channel == 'rayleigh':
            SNR_l = 10**(SNR_db/10)
            u = np.sqrt(SNR_l/(SNR_l + 1))
            def f(b):
                return (1 / (b - u**2))*(np.pi / M * (M - 1) - u * np.sin(np.pi / M) / np.sqrt(b - u**2 * np.cos(np.pi / M)**2) * (1 / np.arctan(-u * np.cos(np.pi / M) / np.sqrt(b - u**2 * np.cos(np.pi / M)**2))))
            
            Pe = ((-1)**(L - 1) * (1 - u**2)**L) / (np.pi * sp.factorial(L - 1)) * derivative(f, 1.0, n=(L-1), dx=1e-6)
            # Pe = (M - 1) / (M * np.log2(M) * np.sin(np.pi / M)**2 * 10**(SNR_db/10)/np.log2(M))
    else:
        if channel == 'awgn':

        elif channel == 'rayleigh':
            k = np.log2(M)
            SNR_l = 10**(SNR_db/10)
            c1 = 1.5 * SNR_l / (M - 1)
            n1 = np.sqrt(c1 / (c1 + 1))
            Pe = 2 * (np.sqrt(M) - 1) / np.sqrt(M) * (1 - n1) - ((np.sqrt(M) - 1) / np.sqrt(M)) * ((np.sqrt(M) - 1) / np.sqrt(M)) * (1 - 4 * n1 / np.pi * np.arctan(1 / n1))
"""

def ser(clf, X, y):
    """ Calculate the misclassification rate, which
        coincides with the symbol error rate (SER) for PSK transmission.
    """
    y_pred = clf.predict(X)
    ser    = np.sum(y != y_pred)/len(y)

    return ser

def plot_confusion_matrix(clf, X, y, num_classes):
    """ Plot the confusion matrix
    """
    y_pred   = clf.predict(X)
    conf_mtx = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10,6))
    sns.heatmap(conf_mtx, cmap=sns.cm.rocket_r, square=True, linewidths=0.1,
                annot=True, fmt='d', annot_kws={"fontsize": 8})
    plt.tick_params(axis='both', which='major', labelsize=10,
                    bottom=False, top=False, left=False,
                    labelbottom=False, labeltop=True)
    plt.yticks(rotation=0)


    plt.show()


def plot_decision_boundary(classifier, X, y, legend=False, plot_training=True):
    """ Plot the classifier decision regions
    """
    num_classes = int(np.max(y))+1 #e.g. 16 for PSK-16
    axes = [np.min(X[:,0]), np.max(X[:,0]),np.min(X[:,1]), np.max(X[:,1])]
    #print(axes)
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = classifier.predict(X_new).reshape(x1.shape)

    # Set different color for each class
    custom_cmap = cm.get_cmap('tab20')
    colors = custom_cmap.colors[:num_classes]
    levels = np.arange(num_classes + 2) - 0.5

    plt.contourf(x1, x2, y_pred, levels=levels, colors=colors, alpha=0.3)

    if plot_training:
        for ii in range(num_classes):
            selected_indices = np.argwhere(y==ii)
            selected_indices = selected_indices.reshape((-1,))
            plt.plot(X[selected_indices, 0], X[selected_indices, 1], "o",
                     c=colors[ii], label=f'{ii}')
        #plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        #plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        #plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        #plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(title='Classes', bbox_to_anchor=(1, 1), loc='upper left',
                   ncol=2, handleheight=2, labelspacing=0.05, frameon=False)
    plt.show()

    
def plot_symbols(X_train, y_train, M, symbs):
    custom_cmap = cm.get_cmap('tab20')
    num_classes = M
    colors = custom_cmap.colors[:num_classes]
    levels = np.arange(num_classes + 2) - 0.5

    for ii in range(num_classes):
        selected_indices = np.argwhere(y_train==ii)
        selected_indices = selected_indices.reshape((-1))
        plt.plot(X_train[selected_indices, 0], X_train[selected_indices, 1], 'o', color=colors[ii], label=f'{ii}')
        
    plt.plot(np.real(symbs), np.imag(symbs), 'rx')
    plt.legend(title='Classes', bbox_to_anchor=(1, 1), loc='upper left', ncol=2, handleheight=2, labelspacing=0.05, frameon=False)

    plt.show()  

def main():
    # file_name = 'psk_crazy.csv'
    file_name = 'psk_awgn.csv'
    with open(file_name, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    data = np.array(data, dtype=float)
    #print(data.shape)

    X = data[:,:-1] # petal length and width
    y = data[:,-1]  # Labels

    classifier = DecisionTreeClassifier(max_depth=20, random_state=42)
    classifier.fit(X, y)

    classifier_ser = ser(classifier, X, y)
    print(f'SER: {classifier_ser*100:.3f} %')

    plot_decision_boundary(classifier, X, y, legend=False, plot_training=True)

if __name__ == '__main__':
    main()

