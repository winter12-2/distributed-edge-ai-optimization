# estimating total communication cost
# formula: parameters * clients * rounds * bytes per parameter 

def estimate_communication_cost(
    parameter_count,           # no. of model parameters getting sent
    clients=10,                # no. edge devices
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
    # example values for testing
    baseline_params = 106710
    quantized_params = 106710

    # baseline model uses float32 so each parameter is 4 bytes
    baseline_cost = estimate_communication_cost(
        parameter_count=baseline_params,
        clients=10,
        rounds=20,
        bytes_per_parameter=4,
    )

    # quantized model uses int8 so each parameter is 1 byte which lowers communication cost
    quantized_cost = estimate_communication_cost(
        parameter_count=quantized_params,
        clients=10,
        rounds=20,
        bytes_per_parameter=1,
    )

    print("Baseline communication cost:", round(baseline_cost["total_mb"], 2), "MB")
    print("Quantized communication cost:", round(quantized_cost["total_mb"], 2), "MB")