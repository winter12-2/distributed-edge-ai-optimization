import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# create results folder if it does not exist
os.makedirs("results", exist_ok=True)


# estimating total communication cost
# formula: parameters * clients * rounds * bytes per parameter
def estimate_communication_cost(
    parameter_count,           # no. of model parameters getting sent
    clients=10,                # no. of edge devices
    rounds=20,                 # no. of communication rounds
    bytes_per_parameter=4,     # size of each parameter
):

    if parameter_count < 0:
        raise ValueError("parameter_count cannot be negative")

    if clients <= 0:
        raise ValueError("clients must be greater than 0")

    if rounds <= 0:
        raise ValueError("rounds must be greater than 0")

    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be greater than 0")

    # calculating total bytes
    total_bytes = parameter_count * clients * rounds * bytes_per_parameter

    # converting bytes into megabytes so the result is easier to read
    total_mb = total_bytes / (1024 * 1024)

    return {
        "parameter_count": parameter_count,
        "clients": clients,
        "rounds": rounds,
        "bytes_per_parameter": bytes_per_parameter,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
    }


if __name__ == "__main__":

    # model parameter count from our CNN model
    baseline_params = 9098
    quantized_params = 9098

    # simulation setup
    clients = 10
    rounds = 20

    # baseline model uses float32, so each parameter is 4 bytes
    baseline_cost = estimate_communication_cost(
        parameter_count=baseline_params,
        clients=clients,
        rounds=rounds,
        bytes_per_parameter=4,
    )

    # quantized model uses int8, so each parameter is 1 byte
    quantized_cost = estimate_communication_cost(
        parameter_count=quantized_params,
        clients=clients,
        rounds=rounds,
        bytes_per_parameter=1,
    )

    # calculate communication reduction
    reduction_percent = (
        (baseline_cost["total_mb"] - quantized_cost["total_mb"])
        / baseline_cost["total_mb"]
    ) * 100

    print("Baseline communication cost:", round(baseline_cost["total_mb"], 4), "MB")
    print("Quantized communication cost:", round(quantized_cost["total_mb"], 4), "MB")
    print("Communication cost reduction:", round(reduction_percent, 2), "%")

    # create table for comparison
    simulation_data = {
        "Model": ["Baseline CNN", "Quantized CNN"],
        "Parameters": [baseline_params, quantized_params],
        "Clients": [clients, clients],
        "Rounds": [rounds, rounds],
        "Bytes per Parameter": [4, 1],
        "Total Communication Cost (MB)": [
            round(baseline_cost["total_mb"], 4),
            round(quantized_cost["total_mb"], 4),
        ],
    }

    simulation_df = pd.DataFrame(simulation_data)

    print("\n===== FEDERATED SIMULATION RESULTS =====")
    print(simulation_df)

    # save table as csv
    simulation_df.to_csv("results/federated_simulation_results.csv", index=False)

    # save summary as json
    summary = {
        "clients": clients,
        "rounds": rounds,
        "baseline_parameters": baseline_params,
        "quantized_parameters": quantized_params,
        "baseline_total_mb": round(baseline_cost["total_mb"], 4),
        "quantized_total_mb": round(quantized_cost["total_mb"], 4),
        "communication_reduction_percent": round(reduction_percent, 2),
    }

    with open("results/federated_simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # create graph for communication cost comparison
    plt.figure()
    plt.bar(simulation_df["Model"], simulation_df["Total Communication Cost (MB)"])
    plt.ylabel("Total Communication Cost (MB)")
    plt.title("Federated Communication Cost Comparison")
    plt.savefig("results/federated_communication_cost.png")
    plt.show()

    print("\nSaved results to results/federated_simulation_results.csv")
    print("Saved summary to results/federated_simulation_summary.json")
    print("Saved graph to results/federated_communication_cost.png")