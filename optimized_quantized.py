
import os
import torch


def get_model_size_kb(file_path):

    # return model file size in KB

    return os.path.getsize(file_path) / 1024


def apply_quantization(model):

    # dynamic quantization changes Linear layers to int8 format
    # this helps reduce model size for edge devices

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    return quantized_model


def save_quantized_model(model, output_path="models/quantized_model.pt"):
    # save the quantized model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


if __name__ == "__main__":
    print("This script applies quantization after the baseline model is available.")
    print("Expected input: models/baseline_model.pt")
    print("Expected output: models/quantized_model.pt")