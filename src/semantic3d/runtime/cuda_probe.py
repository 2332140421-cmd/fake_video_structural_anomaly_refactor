"""Truthful CUDA capability probe including a minimal tensor operation."""

from __future__ import annotations

from typing import Any


def probe_cuda() -> dict[str, Any]:
    """Report CUDA build/runtime facts without treating device count as usability."""

    report: dict[str, Any] = {
        "torch_importable": False,
        "torch_version": "",
        "torch_compiled_cuda_version": "",
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "minimal_tensor_allocation_passed": False,
        "minimal_matrix_operation_passed": False,
        "error": "",
    }
    try:
        import torch

        report["torch_importable"] = True
        report["torch_version"] = str(torch.__version__)
        report["torch_compiled_cuda_version"] = str(torch.version.cuda or "")
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["device_count"] = int(torch.cuda.device_count())
        if report["cuda_available"]:
            devices = []
            for index in range(report["device_count"]):
                properties = torch.cuda.get_device_properties(index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                devices.append(
                    {
                        "index": index,
                        "name": str(properties.name),
                        "total_memory_bytes": int(total_bytes),
                        "free_memory_bytes": int(free_bytes),
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    }
                )
            report["devices"] = devices
            tensor = torch.ones((2, 2), device="cuda")
            report["minimal_tensor_allocation_passed"] = tensor.device.type == "cuda"
            product = tensor @ tensor
            torch.cuda.synchronize()
            report["minimal_matrix_operation_passed"] = bool(
                torch.allclose(product.cpu(), torch.full((2, 2), 2.0))
            )
            del tensor, product
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["ready_for_cuda_batch"] = bool(
        report["cuda_available"]
        and report["minimal_tensor_allocation_passed"]
        and report["minimal_matrix_operation_passed"]
    )
    return report

