import os
import time
from typing import Any, Dict, List

import numpy as np
from prettytable import PrettyTable


def count_parameters(model, module_name: str) -> int:
    """Print and return the number of trainable parameters in a module."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if module_name in name and parameter.requires_grad:
            parameter_count = parameter.numel()
            table.add_row([name, parameter_count])
            total_params += parameter_count
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def generate_performance_table(
    metrics: Dict[str, Dict[str, List[float]]],
    cfg: Any,
    best_round: int,
    best_base_round: int,
    is_test: bool = False,
) -> None:
    """Generate performance tables for the base-to-novel protocol."""
    data_type = cfg.DATASET.SUBSAMPLE_CLASSES

    if is_test:
        print("\n------------Test Performance Metrics------------")
        table = PrettyTable()
        headers = ["Client ID"]
        if data_type != "new":
            headers.append("Local Acc")
        headers.append(f"{data_type} Acc")
        table.field_names = headers

        local_accs = []
        data_type_accs = []
        for client_id in range(cfg.DATASET.USERS):
            row = [client_id]
            if data_type != "new":
                local_acc = metrics["local"]["client_acc"][client_id][0]
                row.append(f"{local_acc:.4f}")
                local_accs.append(local_acc)
            data_type_acc = metrics[data_type]["client_acc"][client_id][0]
            row.append(f"{data_type_acc:.4f}")
            data_type_accs.append(data_type_acc)
            table.add_row(row)

        row = ["Average"]
        if data_type != "new":
            row.append(f"{np.mean(local_accs):.4f}")
        row.append(f"{np.mean(data_type_accs):.4f}")
        table.add_row(row)
        print(table)
        return

    print("\n------------Performance Metrics------------")
    local_table = PrettyTable()
    local_table.field_names = ["Round"] + [
        f"Client {client_id}" for client_id in range(cfg.DATASET.USERS)
    ] + ["Global Avg"]

    data_type_table = PrettyTable()
    data_type_table.field_names = ["Round"] + [
        f"Client {client_id}" for client_id in range(cfg.DATASET.USERS)
    ] + ["Global Avg"]

    for round_idx in range(cfg.OPTIM.ROUND):
        local_table.add_row(
            [round_idx + 1]
            + [f"{metrics['local']['client_acc'][client_id][round_idx]:.4f}" for client_id in range(cfg.DATASET.USERS)]
            + [f"{metrics['local']['acc'][round_idx]:.4f}"]
        )
        data_type_table.add_row(
            [round_idx + 1]
            + [f"{metrics[data_type]['client_acc'][client_id][round_idx]:.4f}" for client_id in range(cfg.DATASET.USERS)]
            + [f"{metrics[data_type]['acc'][round_idx]:.4f}"]
        )

    if cfg.OPTIM.ROUND >= 5:
        local_table.add_row(
            ["Last 5 Avg"]
            + [f"{np.mean(metrics['local']['client_acc'][client_id][-5:]):.4f}" for client_id in range(cfg.DATASET.USERS)]
            + [f"{np.mean(metrics['local']['acc'][-5:]):.4f}"]
        )
        data_type_table.add_row(
            ["Last 5 Avg"]
            + [f"{np.mean(metrics[data_type]['client_acc'][client_id][-5:]):.4f}" for client_id in range(cfg.DATASET.USERS)]
            + [f"{np.mean(metrics[data_type]['acc'][-5:]):.4f}"]
        )

    if best_round >= 0:
        local_table.add_row(
            [f"Best ({best_round + 1})"]
            + [f"{metrics['local']['client_acc'][client_id][best_round]:.4f}" for client_id in range(cfg.DATASET.USERS)]
            + [f"{metrics['local']['acc'][best_round]:.4f}"]
        )

    if best_base_round >= 0:
        data_type_table.add_row(
            [f"Best ({best_base_round + 1})"]
            + [f"{metrics[data_type]['client_acc'][client_id][best_base_round]:.4f}" for client_id in range(cfg.DATASET.USERS)]
            + [f"{metrics[data_type]['acc'][best_base_round]:.4f}"]
        )

    print("\nLocal Accuracy Table:")
    print(local_table)
    print(f"\n{data_type} Accuracy Table:")
    print(data_type_table)


def setup_federated_environment(cfg: Any, args: Any) -> None:
    """Set the random seed and configure file logging."""
    from Dassl.dassl.utils import set_random_seed, setup_logger

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if cfg.DATASET.USEALL:
        setup_logger(os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.SUBSAMPLE_CLASSES))
    else:
        setup_logger(cfg.OUTPUT_DIR)


def initialize_logs(model_save_path: str, model_name: str, args: Any) -> None:
    """Create FedRGP aggregation log files."""
    log_dir = os.path.join(model_save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_files = [
        f"{model_name}_text_prompt_pairs.log",
        f"{model_name}_vision_prompt_pairs.log",
        f"{model_name}_text_aggregate.log",
        f"{model_name}_vision_aggregate.log",
    ]

    for log_file in log_files:
        with open(os.path.join(log_dir, log_file), "w") as file:
            file.write(f"============ {model_name} Training Log ============\n")
            file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Config: {args}\n\n")
