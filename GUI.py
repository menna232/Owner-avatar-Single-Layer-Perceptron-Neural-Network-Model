import tkinter as tk
from tkinter import ttk
import subprocess

def run_selected_algorithm():
    if algorithm_choice.get() == "Perceptron":
        subprocess.run(["python", "perceptron_file.py"])
    elif algorithm_choice.get() == "Adaline":
        subprocess.run(["python", "adaline_file.py"])
    else:
        print("Please select an algorithm.")

def check_changed():
    print(biias.get())


window = tk.Tk()
window.title("User Input")
window.geometry("700x450")

# Create a frame that fills the window
frame = tk.Frame(window)
frame.pack(fill=tk.BOTH, expand=True)

# Saving User Info
user_info_frame = tk.LabelFrame(frame, text="Deep Learning Model Selection")
user_info_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# Initialize variables to store selected features and classes
selected_feature_one = tk.StringVar()
selected_feature_two = tk.StringVar()
selected_class_one = tk.StringVar()
selected_class_two = tk.StringVar()

# Choose Feature One
features_label = tk.Label(user_info_frame, text="Choose Feature One")
features_label.grid(row=0, column=0, padx=10, pady=5)
features_combobox = ttk.Combobox(user_info_frame, textvariable=selected_feature_one, values=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'])
features_combobox.grid(row=0, column=1, padx=10, pady=5)

# Choose Feature Two
features_label = tk.Label(user_info_frame, text="Choose Feature Two")
features_label.grid(row=1, column=0, padx=10, pady=5)
features_combobox = ttk.Combobox(user_info_frame, textvariable=selected_feature_two, values=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'])
features_combobox.grid(row=1, column=1, padx=10, pady=5)

# Choose Class One
classes_label = tk.Label(user_info_frame, text="Choose Class One")
classes_label.grid(row=2, column=0, padx=10, pady=5)
classes_combobox = ttk.Combobox(user_info_frame, textvariable=selected_class_one, values=['BOMBAY', 'CALI', 'SIRA'])
classes_combobox.grid(row=2, column=1, padx=10, pady=5)

# Choose Class Two
classes_label = tk.Label(user_info_frame, text="Choose Class Two")
classes_label.grid(row=3, column=0, padx=10, pady=5)
classes_combobox = ttk.Combobox(user_info_frame, textvariable=selected_class_two, values=['BOMBAY', 'CALI', 'SIRA'])
classes_combobox.grid(row=3, column=1, padx=10, pady=5)


# Enter number of epochs
epochs_label = tk.Label(user_info_frame, text="Enter number of epochs")
epochs_label.grid(row=4, column=0, padx=10, pady=5)
epochs = tk.IntVar()
epochs_entry = tk.Entry(user_info_frame, textvariable=epochs)
epochs_entry.grid(row=4, column=1, padx=10, pady=5)


# Enter Learning Rate
eta_label = tk.Label(user_info_frame, text="Enter Learning Rate")
eta_label.grid(row=5, column=0, padx=10, pady=5)
eta = tk.DoubleVar()
eta_entry = tk.Entry(user_info_frame, textvariable=eta)
eta_entry.grid(row=5, column=1, padx=10, pady=5)

# Enter MSE threshold
MSE_label = tk.Label(user_info_frame, text="Enter MSE threshold")
MSE_label.grid(row=6, column=0, padx=10, pady=5)
MSE_threshold = tk.DoubleVar()
MSE_entry = tk.Entry(user_info_frame, textvariable=MSE_threshold)
MSE_entry.grid(row=6, column=1, padx=10, pady=5)

# Radio buttons for algorithm selection
algorithm_label = tk.Label(user_info_frame, text="Choose the algorithm")
algorithm_label.grid(row=8, column=0, padx=10, pady=5)
algorithm_choice = tk.StringVar()
perceptron_radio = tk.Radiobutton(user_info_frame, text="Perceptron", variable=algorithm_choice, value="Perceptron")
adaline_radio = tk.Radiobutton(user_info_frame, text="Adaline", variable=algorithm_choice, value="Adaline")
perceptron_radio.grid(row=8, column=1, padx=10, pady=5)
adaline_radio.grid(row=8, column=2, padx=10, pady=5)

# Add a button to run the selected algorithm
run_buttonn = tk.Button(user_info_frame, text="Run Algorithm", command=run_selected_algorithm)
run_buttonn.grid(row=9, column=1, columnspan=2, padx=10, pady=5)

# Check for Bias
biias = tk.BooleanVar()
bias_check = tk.Checkbutton(user_info_frame, text="Check for Bias" ,  command=check_changed, variable=biias, onvalue=True, offvalue=False)
bias_check.grid(row=9, column=0, columnspan=2, padx=10, pady=5)


def call_back(e1, e2, e3):
    global feature1, feature2, class1, class2  # Declare them as global
    e1.set(eta.get())
    e2.set(epochs.get())
    e3.set(MSE_threshold.get())
    feature1 = selected_feature_one.get()
    feature2 = selected_feature_two.get()
    class1 = selected_class_one.get()
    class2 = selected_class_two.get()
    window.destroy()


# Run button
run_button = tk.Button(user_info_frame, text="Run Selected Algorithm", command=lambda: call_back(eta, epochs, MSE_threshold))
run_button.grid(row=10, column=0, columnspan=3, padx=10, pady=10)


# Configure frame to expand with the window
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

window.mainloop()