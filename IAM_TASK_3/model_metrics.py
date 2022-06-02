import numpy as np

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

def plot_AUC(clf, name, y_test, x_test):
    score = cross_val_score(clf, x_test, y_test, cv=5)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(x_test), pos_label=2)

    y_predict = clf.predict(x_test)

    y_error = 0
    for i in range(len(y_predict)):
        y_error += (y_test[i] - y_predict[i]) ** 2

    y_error /= len(y_predict)

    y_noise = np.var(y_test)
    y_var = np.var(y_predict)

    # Plot AUC
    AUC = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10,8), dpi=150)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(name + "_AUC" + ".png", bbox_inches='tight')
    plt.clf()

    with open(name + "results.txt", 'w') as f:
        f.write("-- Score --\n" + str(score.mean()) + "\n\n" + "X Test ----------------------------\n" + str(x_test) + "\n\n" + "Prediction -------------------\n" + str(y_predict) + "\n\n" + "Y Test ----------------------------\n" + str(y_test) + "\n\n" + "-- AUC --\n" + str(AUC) + "\n\n" + "-- Error --\n" + str(y_error) + "\n\n" + "-- Noise --\n" + str(y_noise) + "\n\n" + "-- Variance --\n" + str(y_var))

def model_comparision(models, x_test, y_test):
    
    best_auc = 0.0
    models_to_print = []

    with open("./results.txt", 'w') as f:
        for model in models:
            clf = model[0]
            name = model[1]

            print("--- Testing ---")
            print("clf=", clf)
            print("name=", name)
            print("-------")
            print()

            y_prediction = clf.predict(x_test)

            fpr, tpr, _ = metrics.roc_curve(y_test, clf.predict(x_test), pos_label=2)
            running_auc = metrics.auc(fpr, tpr)

            r2_score = clf.score(x_test,y_test)

            model.append(running_auc)
            model.append(r2_score)

            if round(running_auc,2) > best_auc:
                best_auc = round(running_auc,2)
                best_model = name


            f.write("#### " + " On the prediction accuracy of " + name + " ####\n" + "-" * 10 + "\n" + "R2 score (variance explained by the independent variable, ie. how correlated they are) = " + str(round(r2_score, 4)) + "\n" + "Mean sqrd error (the smaller the better)= " + str(round(mean_squared_error(y_test,y_prediction),4)) + "\n" + "Root mean squared error = " + str(round(np.sqrt(mean_squared_error(y_test,y_prediction)),4)) + "\n" + "AUC Score (the largest the better)= " + str(round(running_auc,2)) + "\n" + "-" * 10 + "\n\n\n")
            
            print("R2 score (variance explained by the independent variable, ie. how correlated they are. ~50 no variance) = " + str(round(r2_score, 4)) + "\n" + "Mean sqrd error (the smaller the better)= " + str(round(mean_squared_error(y_test,y_prediction),4)) + "\n" + "Root mean squared error = " + str(round(np.sqrt(mean_squared_error(y_test,y_prediction)),4)) + "\n" + "AUC Score (the larger the better)= " + str(round(running_auc,2)) + "\n" + "-" * 100 + "\n")
            print()

            models_to_print.append(model)

        f.write("--- Best model is " + best_model + " ---\n")
        print("--- Best model is " + best_model + " ---")
        if best_auc > 0.7:
            f.write("From the AUC score of " + str(best_auc) + " the model is predicting above random chance - good prediction.")

            print("From the AUC score of " + str(best_auc) + " the model is predicting above random chance - good prediction.")
        else:
            f.write("From the AUC score of " + str(best_auc) + " the model is predicting at random chance - bad prediction.\n\n\n")
            
            print("From the AUC score of " + str(best_auc) + " the model is predicting at random chance - bad prediction.")
            print()

        f.write(tabulate(models_to_print, headers=["CLF", "Name", "AUC", "R2"]))
        print(tabulate(models_to_print, headers=["CLF", "Name", "AUC", "R2"]))
        print()