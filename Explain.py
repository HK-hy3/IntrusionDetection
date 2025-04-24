import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
column_names = [ "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate" ]

kernel_names = ["linear", "poly", "rbf", "sigmoid"]
for kernel in kernel_names:
    print(f"\n🔍 SHAP summary for {kernel.upper()} kernel")

    # Load model
    model = joblib.load(f"svm_{kernel}.pkl")

    # Use SHAP KernelExplainer for SVM
    explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Use a small background set
    shap_values = explainer.shap_values(X_test[:100], nsamples=100)  # Limit for speed

    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_test[:100], feature_names=column_names, show=False)
    plt.title(f"SHAP Summary Plot - {kernel.upper()} Kernel")
    plt.savefig(f"Shap_{kernel}.png")
    plt.close()

    print(f"Saved: shap_{kernel}.png")
