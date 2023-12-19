import pandas as pd
import numpy as np
from tkinter import messagebox
from GUI import *
from sklearn.preprocessing import LabelEncoder
import numpy as np

import warnings
warnings.filterwarnings("ignore")



def net_input(x1, x2, w1, w2, bias):
    if (bias_bool == True):
        # print("ashta")
        return w1 * x1 + w2 * x2 + bias
    else:
        return w1 * x1 + w2 * x2


def activation(x):
    return x


def activation2(net):
    return np.where(net > 0, 1, -1)


def fit(X, Y, learning_rate, num_epochs, MSE_threshold):
    w1, w2 = np.random.rand(2)
    bias = np.random.rand()

    # costs = []

    for epoch in range(num_epochs):
        mse_ = 0.0

        for i in range(len(X)):
            x1, x2 = X[i]
            d = Y[i]

            net = net_input(x1, x2, w1, w2, bias)
            y = activation(net)
            error = d - y

            if bias_bool:
                bias += learning_rate * error
            # Update the weights
            w1 += learning_rate * error * x1
            w2 += learning_rate * error * x2
            # Calculate the MSE for this example
            mse_ += (error ** 2)
        # Calculate the MSE for the epoch
        mse_ = (0.5 * mse_) / len(X)
        # costs.append(MSE)


        # Check if MSE has reached the threshold
        if mse_ <= MSE_threshold:
            print(f"Stopped training after {epoch + 1} epochs. MSE reached the threshold.")
            break
        else:
            print(f"Stopped training after {epoch + 1} epochs.")
    if not bias_bool:
        bias = 0

    return w1, w2, bias


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def test(X, w1, w2, bias):
    net = w1 * X[:, 0] + w2 * X[:, 1] + bias
    return activation2(net)



# Read the data
dataset = pd.read_excel("Dry_Bean_Dataset.xlsx")
mean_minor_axis_length = dataset['MinorAxisLength'].mean()
dataset['MinorAxisLength'].fillna(mean_minor_axis_length, inplace=True)
numeric_columns = dataset.select_dtypes(include=[np.number])  # Select only numerical columns
# Normalize only the numerical columns
dataset[numeric_columns.columns] = (numeric_columns - numeric_columns.min()) / (
            numeric_columns.max() - numeric_columns.min())

# Inputs
selected_feature1 = selected_feature_one.get()
selected_feature2 = selected_feature_two.get()
selected_class1 = selected_class_one.get()
selected_class2 = selected_class_two.get()
eta = eta.get()
epochs = epochs.get()
bias_bool = biias.get()
MSE_threshold = MSE_threshold.get()


if any(val is None or val == '' for val in
       [selected_feature1, selected_feature2, selected_class1, selected_class2, eta, epochs]):
    messagebox.showerror("Error", "Please fill in all required fields.")
else:
    if selected_feature1 == selected_feature2 or selected_class1 == selected_class2:
        messagebox.showerror("Error", "Please make sure that the entered features and classes are different.")
    else:
        c1 = dataset[dataset['Class'] == selected_class1]
        c2 = dataset[dataset['Class'] == selected_class2]

        c1 = c1.sample(frac=1, random_state=1).reset_index(drop=True)
        c2 = c2.sample(frac=1, random_state=1).reset_index(drop=True)

        # Create training and testing datasets
        training_c1 = c1.iloc[:30, :]
        testing_c1 = c1.iloc[30:, :]
        training_c2 = c2.iloc[:30, :]
        testing_c2 = c2.iloc[30:, :]

        training_c1 = training_c1.sample(frac=1, random_state=1)
        training_c2 = training_c2.sample(frac=1, random_state=1)

        training_dataset = pd.concat([training_c1, training_c2])
        testing_dataset = pd.concat([testing_c1, testing_c2])

        label_encoder = LabelEncoder()
        training_dataset['Class'] = label_encoder.fit_transform(training_dataset['Class'])
        testing_dataset['Class'] = label_encoder.transform(testing_dataset['Class'])
        training_dataset['Class'] = training_dataset['Class'].apply(lambda x: 1 if x == 0 else -1)
        testing_dataset['Class'] = testing_dataset['Class'].apply(lambda x: 1 if x == 0 else -1)

        print(training_dataset['Class'] , testing_dataset['Class'])

        # Training Dataset
        x1_train = np.array(training_dataset[selected_feature1])
        x2_train = np.array(training_dataset[selected_feature2])
        x_train = np.column_stack((x1_train, x2_train))
        y_train = np.array(training_dataset['Class'])

        # Testing Dataset
        x1_test = np.array(testing_dataset[selected_feature1])
        x2_test = np.array(testing_dataset[selected_feature2])
        x_test = np.column_stack((x1_test, x2_test))
        y_test = np.array(testing_dataset['Class'])

        print(x_train.shape, x_test.shape , y_train.shape ,y_test.shape )

        w1, w2, bias = fit(x_train, y_train, eta, epochs , MSE_threshold)

        # mse_value = mse(x_test, y_test, w1, w2, bias)
        # print("Mean Squared Error on Testing Data:", mse_value)

        y_pred = test(x_test, w1, w2, bias)
        # y_pred = np.where((w1 * x1_test + w2 * x2_test + bias) >= 0, 1, -1)


        accuracy_value = accuracy(y_test, y_pred)
        # print("Accuracy on Testing Data:", accuracy_value)

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, accuracy_score

        # Plotting the points of both classes
        plt.figure(figsize=(8, 6))
        plt.scatter(x1_test[y_test == 1], x2_test[y_test == 1], marker='o', label='Class 1', color='blue')
        plt.scatter(x1_test[y_test == -1], x2_test[y_test == -1], marker='x', label='Class -1', color='red')

        # Determine the range for the decision boundary plot
        x_min, x_max = x1_test.min() - 0.1, x1_test.max() + 0.1
        y_min, y_max = x2_test.min() - 0.1, x2_test.max() + 0.1

        # Create a mesh grid for the decision boundary
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Calculate the decision boundary values
        Z = w1 * xx + w2 * yy + bias

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, levels=[-1, 0 ,1], colors=['red', 'blue'], alpha=0)

        contour = plt.contour(xx, yy, Z, levels=[0], colors='black')

        plt.xlabel(selected_feature1)
        plt.ylabel(selected_feature2)
        plt.legend(loc='upper right')
        plt.title('Decision Boundary and Scatter Plot For Adaline Model')
        plt.show()


        tp = 0
        tn = 0
        fp = 0
        fn = 0


        # calculate the confusion matrix
        for i in range(len(y_test)):

            if y_test[i] == y_pred[i] :
                if y_test[i]==1:
                 tp += 1
                else :
                  tn+=1
            else:
                if y_test[i]==1:
                 fn += 1
                else :
                 fp+=1



        # calculate the precision and recall
        if tp + fp != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0
            recall = 0

        # print the results
        print('--------------------------------')
        print(f'Testing MSE: {mse_value}')
        print(f'Testing Accuracy: {accuracy_value:.2f}%')
        print(f'Testing Precision: {precision * 100:.2f}%')
        print(f'Testing Recall: {recall * 100:.2f}%')
        print('--------------------------------')
        print(f'Confusion Matrix:')
        print(f'\t\t\tNegative\t|\tPositive\t')
        print(f'Negative:\t\t{tn}\t\t|\t{fp}\t')
        print(f'Positive:\t\t{fn}\t\t|\t{tp}')
        print('--------------------------------')








