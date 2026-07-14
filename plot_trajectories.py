"""
Скрипт для визуализации истинных и предсказанных траекторий маржи.
Использует как детерминированный генератор траекторий (Trajectory Generator),
так и стохастический CVAE для оценки неопределенности.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Импортируем конфигурацию из предоставленного evaluate_ood.py
try:
    from evaluate_ood import DATASET_CONFIG
except ImportError:
    logging.warning("Не удалось импортировать из 'evaluate_ood'. Проверяем резервные варианты...")
    try:
        from evaluate import DATASET_CONFIG
    except ImportError:
        raise ImportError(
            "Не удалось найти DATASET_CONFIG ни в 'evaluate_ood', ни в 'evaluate'. "
            "Пожалуйста, убедитесь, что ваш файл evaluate_ood (1).py переименован в evaluate_ood.py"
        )

from src.models import TrajectoryGenerator, TrajectoryCVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация истинных и предсказанных траекторий.")
    parser.add_argument(
        "--dataset", type=str, choices=("cifar10n", "cifar100n"), default="cifar100n",
        help="Датасет для визуализации (cifar10n или cifar100n)."
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default=None,
        help="Путь к директории артефактов (переопределяет значение по умолчанию из конфига)."
    )
    parser.add_argument(
        "--num-samples-to-plot", type=int, default=5, 
        help="Количество случайных графиков для построения."
    )
    parser.add_argument(
        "--cvae-samples", type=int, default=50, 
        help="Количество проходов через CVAE для оценки дисперсии/неопределенности."
    )
    parser.add_argument("--latent-dim", type=int, default=128, help="Размерность латентного пространства CVAE.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Размерность скрытого слоя.")
    parser.add_argument("--sequence-length", type=int, default=50, help="Длина траектории (количество эпох).")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def load_models_and_data(args: argparse.Namespace, device: torch.device):
    spec = DATASET_CONFIG[args.dataset]
    artifacts_dir = Path(args.artifacts_dir or spec.artifacts_dir)
    
    # Извлекаем пути к признакам и детерминированному генератору напрямую из DatasetSpec
    features_file = artifacts_dir / spec.features_file
    generator_file = artifacts_dir / spec.trajectory_file

    # Поскольку в DatasetSpec нет явных путей для CVAE и таргетов траекторий, 
    # определяем их динамически на основе настроек стадий обучения
    if args.dataset == "cifar10n":
        targets_file = artifacts_dir / "trajectory_targets_cifar10n_margin.npy"
        if not targets_file.exists():
            targets_file = artifacts_dir / "trajectory_targets_cifar10n.npy"
        cvae_file = artifacts_dir / "trajectory_cvae.pth"
    else:  # cifar100n
        targets_file = artifacts_dir / "trajectory_targets_cifar100n_margin.npy"
        if not targets_file.exists():
            targets_file = artifacts_dir / "trajectory_targets_cifar100n.npy"
        cvae_file = artifacts_dir / "trajectory_cvae_cifar100n.pth"

    logging.info("Используемые пути к файлам:")
    logging.info("  Признаки:    %s", features_file)
    logging.info("  Таргеты:     %s", targets_file)
    logging.info("  D-Generator: %s", generator_file)
    logging.info("  CVAE-Model:  %s", cvae_file)

    if not features_file.exists():
        raise FileNotFoundError(f"Файл признаков не найден: {features_file}")
    if not targets_file.exists():
        raise FileNotFoundError(f"Файл истинных траекторий не найден: {targets_file}")

    features = np.load(features_file)
    targets = np.load(targets_file)

    input_dim = features.shape[1]
    sequence_length = args.sequence_length

    # Инициализация и загрузка весов детерминированного генератора
    generator = TrajectoryGenerator(input_dim=input_dim, output_dim=sequence_length, hidden_dim=args.hidden_dim)
    if generator_file.exists():
        generator.load_state_dict(torch.load(generator_file, map_location=device, weights_only=True))
        logging.info("✅ Успешно загружены веса TrajectoryGenerator.")
    else:
        logging.warning("⚠️ Файл весов TrajectoryGenerator не найден. Используются случайные веса.")
    generator.to(device)
    generator.eval()

    # Инициализация и загрузка весов CVAE
    cvae = TrajectoryCVAE(feature_dim=input_dim, trajectory_dim=sequence_length, latent_dim=args.latent_dim)
    if cvae_file.exists():
        cvae.load_state_dict(torch.load(cvae_file, map_location=device, weights_only=True))
        logging.info("✅ Успешно загружены веса TrajectoryCVAE.")
    else:
        logging.warning("⚠️ Файл весов TrajectoryCVAE не найден. Используются случайные веса.")
    cvae.to(device)
    cvae.eval()

    return features, targets, generator, cvae


def main() -> None:
    configure_logging()
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Используемое устройство: %s", device)

    try:
        features, targets, generator, cvae = load_models_and_data(args, device)
    except Exception as e:
        logging.error("❌ Ошибка при загрузке моделей или данных: %s", e)
        return

    num_samples = len(features)
    num_to_plot = min(args.num_samples_to_plot, num_samples)
    indices = random.sample(range(num_samples), num_to_plot)

    logging.info("Построение графиков для индексов: %s", indices)

    epochs = np.arange(1, args.sequence_length + 1)

    for idx in indices:
        feat_single = torch.tensor(features[idx], dtype=torch.float32).unsqueeze(0).to(device)
        true_traj = targets[idx]

        # 1. Предсказание детерминированного генератора
        with torch.no_grad():
            pred_traj = generator(feat_single).squeeze(0).cpu().numpy()

        # 2. Множественное сэмплирование из стохастического CVAE для оценки неопределенности
        cvae_runs = []
        with torch.no_grad():
            for _ in range(args.cvae_samples):
                sample_traj = cvae.generate(feat_single).squeeze(0).cpu().numpy()
                cvae_runs.append(sample_traj)
        cvae_runs = np.array(cvae_runs)  # Форма: (cvae_samples, sequence_length)
        cvae_mean = np.mean(cvae_runs, axis=0)
        cvae_std = np.std(cvae_runs, axis=0)

        # Построение графика с двойной осью Y
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Левая ось (Истинная маржа и предсказание детерминированного генератора)
        line_true = ax1.plot(epochs, true_traj, label="Реальная траектория", color="dodgerblue", linewidth=2.5)
        line_gen = ax1.plot(
            epochs, pred_traj, label="Прогноз генератора (D-Gen)", color="forestgreen", linestyle="--", linewidth=2
        )
        ax1.set_xlabel("Эпоха обучения", fontsize=11)
        ax1.set_ylabel("Значение маржи (Истинное / D-Gen)", fontsize=11, color="dodgerblue")
        ax1.tick_params(axis='y', labelcolor="dodgerblue")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Правая ось (CVAE предсказание со средней линией и областью неопределенности)
        ax2 = ax1.twinx()
        line_cvae = ax2.plot(epochs, cvae_mean, label="CVAE (Среднее значение)", color="crimson", linewidth=2, zorder=2)
        
        # Область ±1 стандартное отклонение
        ax2.fill_between(
            epochs, 
            cvae_mean - cvae_std, 
            cvae_mean + cvae_std, 
            color="crimson", 
            alpha=0.18, 
            zorder=1
        )
        has_fill = True
        
        ax2.set_ylabel("CVAE оценка траектории", fontsize=11, color="crimson")
        ax2.tick_params(axis='y', labelcolor="crimson")
        
        # Задаем фиксированные границы, если сэмплы лежат в диапазоне вероятностей [0, 1]
        if np.all(cvae_mean >= -0.05) and np.all(cvae_mean <= 1.05):
            ax2.set_ylim(-0.05, 1.05)

        # Динамически собираем легенду без дубликатов
        lines = line_true + line_gen + line_cvae
        labels = [l.get_label() for l in lines]
        handles = list(lines)
        
        if has_fill:
            labels.append("Неопределенность CVAE (±1 std)")
            handles.append(plt.Rectangle((0, 0), 1, 1, color="crimson", alpha=0.18))
            
        ax1.legend(handles, labels, loc="upper left", fontsize=10)
        
        plt.title(f"Визуализация траекторий маржи | Датасет: {args.dataset} | Индекс: {idx}", fontsize=13, pad=15)
        fig.tight_layout()
        
        # Сохранение результатов
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f"trajectory_{args.dataset}_idx_{idx}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info("График успешно сохранен в: %s", plot_path)

    logging.info("🎉 Все запланированные графики сгенерированы в директории './plots'!")


if __name__ == "__main__":
    main()