import os

from CSVtoNPZ import process_single_csv_to_npz
from TCLtoCSV import parse_tcl_to_csv
from TrainGRU import train_and_save, load_data
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == '__main__':
    tcl_folder = './data/tcl_files'
    csv_folder = './data/csv_files'
    npz_folder = './data/npz_files'
    sequence_length = 10
    train_file = './data/npz_files/mobility_120.npz'
    test_file = './data/npz_files/mobility_60.npz'
    model_file = './models/gru_position_predictor.h5'

    #--------------------- First make csv files from tcl for using in GRU------------------
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder, exist_ok=True)

        for filename in os.listdir(tcl_folder):
            if filename.endswith('.tcl'):
                parse_tcl_to_csv(filename, tcl_folder, csv_folder)
                print(f"Processed {filename}")

    #---------------------- Second change csv to npz-------------------------------
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder, exist_ok=True)

        for filename in os.listdir(csv_folder):
            if filename.endswith('.csv'):
                csv_path = os.path.join(csv_folder, filename)
                base_name = os.path.splitext(filename)[0]
                output_npz_path = os.path.join(npz_folder, f"{base_name}.npz")

                process_single_csv_to_npz(csv_path, output_npz_path, sequence_length)


    #---------------------- Third train GRU with data------------------------------------
    if not os.path.exists(model_file):
        os.makedirs(os.path.dirname(model_file), exist_ok=True)

        print("Loading training data...")
        X_train, Y_train = load_data(train_file)
        print(f"Train: X={X_train.shape}, Y={Y_train.shape}")

        print("Training model...")
        train_and_save(X_train, Y_train, model_file)
    else:
        print("Model exists. Skipping training.")


    # --------------------------Last test the model--------------------------------------
    print("Loading test data...")
    X_test, Y_test = load_data(test_file)
    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

    print("Evaluating model...")
    model = load_model(model_file, compile=False)
    Y_pred = model.predict(X_test)

    # ------------------------ Evaluation Metrics ----------------------------------------
    mae_x = mean_absolute_error(Y_test[:, 0], Y_pred[:, 0])
    mae_y = mean_absolute_error(Y_test[:, 1], Y_pred[:, 1])
    rmse_x = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred[:, 1]))
    distances = np.sqrt(np.sum((Y_test - Y_pred) ** 2, axis=1))
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    p_at_50 = np.mean(distances < 50) * 100
    mape_x = mean_absolute_percentage_error(Y_test[:, 0], Y_pred[:, 0])
    mape_y = mean_absolute_percentage_error(Y_test[:, 1], Y_pred[:, 1])
    r2_x = r2_score(Y_test[:, 0], Y_pred[:, 0])
    r2_y = r2_score(Y_test[:, 1], Y_pred[:, 1])

    print(f"MAE: x={mae_x:.2f}, y={mae_y:.2f}")
    print(f"RMSE: x={rmse_x:.2f}, y={rmse_y:.2f}")
    print(f"Mean Euclidean Distance: {mean_dist:.2f}")
    print(f"Max Euclidean Distance: {max_dist:.2f}")
    print(f"Percentage of predictions < 50 meters error: {p_at_50:.2f}%")
    print(f"MAPE: x={mape_x:.2f}%, y={mape_y:.2f}%")
    print(f"R² Score: x={r2_x:.2f}, y={r2_y:.2f}")

    # ------------------------------ Charts-------------------------------------------
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title("Histogram of Euclidean Errors")
    plt.xlabel("Error (meters)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    ks = np.arange(0, 500, 10)
    accuracies = [np.mean(distances < k) * 100 for k in ks]
    plt.plot(ks, accuracies, marker='o')
    plt.title("Accuracy@K")
    plt.xlabel("Distance Threshold (meters)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test[:100, 0], Y_test[:100, 1], label='Actual', c='blue', alpha=0.6)
    plt.scatter(Y_pred[:100, 0], Y_pred[:100, 1], label='Predicted', c='red', alpha=0.6)
    plt.legend()
    plt.title("Actual vs. Predicted Positions (First 100 samples)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()




