import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Data.Dataset import LatentFMRIDataset
import matplotlib.pyplot as plt
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

data_directories = ['/Users/balazsmorvay/PycharmProjects/VAE/Assets/UM_1_all',
                    '/Users/balazsmorvay/PycharmProjects/VAE/Assets/NYU_all',
                    '/Users/balazsmorvay/PycharmProjects/VAE/Assets/NYU_UM1_merged']
dataset_names = ['UM_1', 'NYU', 'UM1+NYU']
test_ratio = 0.15


def perform_experiment():
    for dataset_name, data_directory in zip(dataset_names, data_directories):

        dataset = LatentFMRIDataset(data_dir=data_directory)
        all_data_items = dataset.get_all_items()
        X = all_data_items['X']
        y = all_data_items['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

        mlflow.set_experiment(f"SVC on {dataset_name}")

        for c in np.arange(start=0.01, stop=5, step=0.5):
            with mlflow.start_run():
                experiment_name = f"SVC on {dataset_name} with c={c}"
                mlflow.set_tag("mlflow.runName", experiment_name)
                experiment_path = os.path.join('/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments',
                                               experiment_name)
                os.mkdir(path=experiment_path)

                mlflow_tracking_params = {
                    'Site': dataset_name,
                    'Data directory': data_directory,
                    'Test_ratio': test_ratio,
                    'Train_labels_mean': np.mean(y_train),
                    'Test_labels_mean': np.mean(y_test),
                    'C': c
                }
                mlflow.log_params(params=mlflow_tracking_params, synchronous=True)

                X_train = X_train.reshape((X_train.shape[0], -1))
                X_test = X_test.reshape((X_test.shape[0], -1))

                model = SVC(kernel='rbf', C=c, class_weight='balanced', random_state=42, verbose=True)
                model.fit(X=X_train, y=y_train)
                mlflow.sklearn.log_model(model, experiment_name)

                test_predictions = model.predict(X_test)
                cm = confusion_matrix(y_test, test_predictions, labels=model.classes_)
                display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                display.plot().figure_.savefig(os.path.join(experiment_path, 'confusion_matrix.png'))
                plt.close('all')
                mlflow.log_artifact(os.path.join(experiment_path, 'confusion_matrix.png'))

                tn, fp, fn, tp = cm.ravel()
                metrics = {
                    'accuracy': ((tp + tn) / (tp + tn + fp + fn + 1e-6)),
                    'recall': (tp / (tp + fn + 1e-6)),
                    'precision': (tp / (tp + fp + 1e-6))
                }
                mlflow.log_metrics(metrics, synchronous=True)


if __name__ == '__main__':
    perform_experiment()
