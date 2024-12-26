import numpy as np


def plot_confusion_mat(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    fontsize = 20
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    acsa = (cm[0,0]/sum(cm[0,:])+cm[1,1]/sum(cm[1,:])+cm[2,2]/sum(cm[2,:]))/3
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize= fontsize)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0,fontsize= fontsize)
        plt.yticks(tick_marks, target_names,fontsize= fontsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:d}".format(int(cm[i, j])),fontsize= 20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:d}".format(int(cm[i, j])),fontsize= 20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    class_acc = [cm[i,i]/np.sum(cm[i,:]) if np.sum(cm[i,:]) else 0 for i in range(len(cm))]
    ppvs = [cm[i,i]/np.sum(cm[:,i]) if np.sum(cm[:,i]) else 0 for i in range(len(cm))]

   # plt.tight_layout()
    plt.ylabel('True label',fontsize= fontsize)
    plt.xlabel('Predicted label\n\naccuracy={:0.4f}; misclass={:0.4f}: acsa={:0.4f}\nSens Normal: {:0.4f}, Pneumonia {:0.4f}, COVID-19: {:0.4f}\nPPV Normal: {:0.4f}, Pneumonia {:0.4f}, COVID-19: {:0.4f}'.format(accuracy, misclass,acsa,class_acc[0],class_acc[1],class_acc[2],ppvs[0], ppvs[1], ppvs[2]),fontsize= fontsize)

  #  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\nSensitivity Normal: {0:.3f}, Pneumonia {:.3f}, COVID-19: {:.3f};\nPPV Normal: {:.3f}, Pneumonia {:.3f}, COVID-19: {:.3f}'.format(accuracy, misclass,class_acc[0],class_acc[1],class_acc[2],ppvs[0], ppvs[1], ppvs[2]),fontsize= 'x-large')
    plt.show()
    return fig
    