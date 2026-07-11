import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Автоматический запуск пайплайна (Stage 1 -> Stage 2 -> Eval) с инкрементом сида."
    )
    parser.add_argument(
        "--start-seed", type=int, default=42, help="Начальный random seed"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Количество последовательных запусков",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10n", "cifar100n"],
        default="cifar100n",
        help="На каком датасете запускать",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./experiments",
        help="Корневая папка для сохранения всех результатов",
    )

    # Параметры для Stage 2
    parser.add_argument(
        "--lr2",
        type=float,
        default=0.001,
        help="Learning rate для генератора траекторий (Stage 2)",
    )
    parser.add_argument(
        "--epochs2",
        type=int,
        default=80,
        help="Количество эпох для генератора траекторий (Stage 2)",
    )

    args = parser.parse_args()

    # В новой версии проекта скрипты одни и те же для всех датасетов
    stage1_script = "train_stage1.py"
    stage2_script = "train_stage2.py"
    eval_script = "evaluate_ood.py"

    output_root = Path(args.output_root)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Чтобы логи выводились сразу, без задержек

    print(f"=== Запуск серии экспериментов для {args.dataset} ===")
    print(
        f"Начальный сид: {args.start_seed} | Количество запусков: {args.num_runs}\n"
    )

    for run_idx in range(args.num_runs):
        current_seed = args.start_seed + run_idx
        print(
            f"--- [ЗАПУСК {run_idx + 1}/{args.num_runs}] Используем SEED = {current_seed} ---"
        )

        # Создаем изолированные папки под текущий сид
        run_dir = output_root / f"seed_{current_seed}"
        artifacts_dir = run_dir / "artifacts"
        reports_dir = run_dir / "reports"

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # ==========================================
        # СТЕЙДЖ 1
        # ==========================================
        print(f"     Запуск Stage 1 ({stage1_script})...")
        cmd_stage1 = [
            sys.executable,
            stage1_script,
            "--dataset",
            args.dataset,
            "--seed",
            str(current_seed),
            "--output-dir",
            str(artifacts_dir),
        ]

        res1 = subprocess.run(cmd_stage1, env=env)
        if res1.returncode != 0:
            print(f"❌ Ошибка на Stage 1 с сидом {current_seed}. Прерываем серию.")
            sys.exit(res1.returncode)

        # ==========================================
        # СТЕЙДЖ 2
        # ==========================================
        print(f"     Запуск Stage 2 ({stage2_script})...")
        cmd_stage2 = [
            sys.executable,
            stage2_script,
            "--dataset",
            args.dataset,
            "--seed",
            str(current_seed),
            "--artifacts-dir",
            str(artifacts_dir),
            "--lr",
            str(args.lr2),
            "--epochs",
            str(args.epochs2),
        ]

        res2 = subprocess.run(cmd_stage2, env=env)
        if res2.returncode != 0:
            print(f"❌ Ошибка на Stage 2 с сидом {current_seed}. Прерываем серию.")
            sys.exit(res2.returncode)

        # ==========================================
        # ОЦЕНКА
        # ==========================================
        print(f"     Запуск Evaluation ({eval_script})...")
        cmd_eval = [
            sys.executable,
            eval_script,
            "--dataset",
            args.dataset,
            "--artifacts-dir",
            str(artifacts_dir),
            "--reports-dir",
            str(reports_dir),
        ]

        res3 = subprocess.run(cmd_eval, env=env)
        if res3.returncode != 0:
            print(f"❌ Ошибка при оценке с сидом {current_seed}. Прерываем серию.")
            sys.exit(res3.returncode)

        print(
            f"✅ Запуск {run_idx + 1} успешно завершен! Отчет сохранен в: {reports_dir}\n"
        )

    print("🎉 Все запланированные запуски успешно выполнены!")


if __name__ == "__main__":
    main()