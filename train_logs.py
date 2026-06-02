import csv
import os
import time
from datetime import datetime

import torch


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def _normalize_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _make_dummy_inputs(batch_size, image_size, device):
    h, w = image_size
    rgb = torch.randn(batch_size, 3, h, w, device=device)
    dsm = torch.randn(batch_size, h, w, device=device)
    return rgb, dsm


def _forward_once(model, inputs):
    return _normalize_output(model(*inputs, mode="Test"))


def estimate_flops_g(model, inputs):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if inputs[0].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            _forward_once(model, inputs)

    total_flops = sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages())
    return total_flops / 1e9


def benchmark_inference(model, inputs, warmup=10, repeat=30):
    device = inputs[0].device
    batch_size = inputs[0].shape[0]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _forward_once(model, inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(repeat):
            _forward_once(model, inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    fps = (repeat * batch_size) / elapsed if elapsed > 0 else 0.0
    time_img_ms = (elapsed * 1000.0) / (repeat * batch_size) if elapsed > 0 else 0.0
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0.0
    )

    return fps, time_img_ms, peak_memory_mb


def append_metrics_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "Time",
        "Model",
        "Dataset",
        "Input Size",
        "Batch Size",
        "Params (M)",
        "FLOPS (G)",
        "FPS",
        "Time/img (ms)",
        "Peak Memory",
        "Peak Memory Unit",
        "Device",
        "Note",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def export_train_metrics(
    model,
    csv_path,
    model_name,
    dataset_name,
    image_size,
    batch_size=1,
    warmup=10,
    repeat=30,
):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    note = ""
    params_m = count_params_m(model)
    flops_g = None
    fps = None
    time_img_ms = None
    peak_memory_mb = None

    inputs = _make_dummy_inputs(batch_size, image_size, device)

    try:
        flops_g = estimate_flops_g(model, inputs)
    except Exception as exc:
        note = "FLOPS unavailable: {}".format(type(exc).__name__)

    try:
        fps, time_img_ms, peak_memory_mb = benchmark_inference(
            model, inputs, warmup=warmup, repeat=repeat
        )
    except Exception as exc:
        note = "{}; benchmark unavailable: {}".format(note, type(exc).__name__).strip("; ")

    if was_training:
        model.train()

    row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Dataset": dataset_name,
        "Input Size": "{}x{}".format(image_size[0], image_size[1]),
        "Batch Size": batch_size,
        "Params (M)": round(params_m, 4),
        "FLOPS (G)": "" if flops_g is None else round(flops_g, 4),
        "FPS": "" if fps is None else round(fps, 4),
        "Time/img (ms)": "" if time_img_ms is None else round(time_img_ms, 4),
        "Peak Memory": "" if peak_memory_mb is None else round(peak_memory_mb, 2),
        "Peak Memory Unit": "MB",
        "Device": str(device),
        "Note": note,
    }
    append_metrics_csv(csv_path, row)
    return row
