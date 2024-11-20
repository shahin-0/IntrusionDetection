import joblib
import numpy as np

model = joblib.load("nids_model.pkl")

def predict_intrusion():

    print("\nEnter network traffic features:")
    feature_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    input_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        input_data.append(value)

    input_data = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_data)
    result = "Intrusion Detected!" if prediction[0] == 1 else "No Intrusion Detected (Normal Traffic)"
    print(f"\nResult: {result}")

if __name__ == "__main__":
    print("=== Network Intrusion Detection System ===")
    predict_intrusion()
