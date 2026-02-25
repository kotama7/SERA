"""Internationalization support for the Setup Wizard.

Contains the MESSAGES dictionary for ja/en localization of all wizard prompts.
"""

from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    "ja": {
        "welcome": "SERA セットアップウィザードへようこそ！",
        "step_header": "Step {step}/{total}",
        "data_desc": "データの説明を入力してください（例: 画像分類用のCIFAR-10データセット）",
        "data_loc": "データの場所を入力してください（パス/URI/リポジトリURL）",
        "data_format": "データ形式を選択してください",
        "data_size": "データサイズの目安を選択してください",
        "domain_field": "研究分野を入力してください（例: HPC, NLP, CV, materials）",
        "domain_subfield": "より具体的な分野を入力してください（空可）",
        "task_brief": "研究タスクを1〜3文で説明してください",
        "task_type": "タスクの種類を選択してください",
        "goal_objective": "研究の目標を記述してください（例: 実行時間を最小化）",
        "goal_direction": "目標の方向を選択してください",
        "goal_metric": "評価指標の名前を入力してください（例: accuracy, runtime_sec）",
        "goal_baseline": "既知のベースライン値があれば入力してください（空可）",
        "add_constraint": "制約条件を追加しますか？",
        "constraint_name": "制約条件の名前",
        "constraint_type": "制約の種類を選択してください",
        "constraint_threshold": "閾値を入力してください",
        "notes": "その他の備考があれば入力してください（空可）",
        "preview_confirm": "この内容でよろしいですか？",
        "phase0_params": "Phase 0 パラメータを確認してください",
        "phase0_running": "Phase 0 実行中...",
        "phase0_done": "Phase 0 完了: {n_papers} 論文を収集しました",
        "review_papers": "収集された論文をレビューしてください",
        "spec_review": "生成されたSpecをレビューしてください",
        "freeze_confirm": "この設定でSpecをフリーズしますか？",
        "setup_done": "セットアップ完了！ sera research で探索を開始できます",
        "direction_estimate": '目標「{obj}」から direction = "{dir}" と推定しました。正しいですか？',
        "back_help": "(back: 戻る / help: ヘルプ / quit: 中断保存)",
        "state_saved": "状態を保存しました。--resume で再開できます",
        "resuming": "前回の状態から再開します（Step {step}）",
        "env_detect_gpu": "GPU検出: {info}",
        "env_detect_slurm": "SLURM検出: {info}",
    },
    "en": {
        "welcome": "Welcome to SERA Setup Wizard!",
        "step_header": "Step {step}/{total}",
        "data_desc": "Describe your data (e.g., 'CIFAR-10 image classification dataset')",
        "data_loc": "Enter data location (path/URI/repository URL)",
        "data_format": "Select data format",
        "data_size": "Select approximate data size",
        "domain_field": "Enter research field (e.g., HPC, NLP, CV, materials)",
        "domain_subfield": "Enter subfield (optional, press Enter to skip)",
        "task_brief": "Describe the research task in 1-3 sentences",
        "task_type": "Select task type",
        "goal_objective": "Describe the research goal (e.g., 'Minimize runtime')",
        "goal_direction": "Select optimization direction",
        "goal_metric": "Enter metric name (e.g., accuracy, runtime_sec)",
        "goal_baseline": "Enter baseline value if known (optional)",
        "add_constraint": "Add a constraint?",
        "constraint_name": "Constraint name",
        "constraint_type": "Select constraint type",
        "constraint_threshold": "Enter threshold value",
        "notes": "Any additional notes (optional)",
        "preview_confirm": "Does this look correct?",
        "phase0_params": "Review Phase 0 parameters",
        "phase0_running": "Running Phase 0...",
        "phase0_done": "Phase 0 complete: collected {n_papers} papers",
        "review_papers": "Review collected papers",
        "spec_review": "Review generated specs",
        "freeze_confirm": "Freeze specs with these settings?",
        "setup_done": "Setup complete! Run sera research to start exploration",
        "direction_estimate": 'Estimated direction = "{dir}" from goal "{obj}". Correct?',
        "back_help": "(back: go back / help: show help / quit: save & exit)",
        "state_saved": "State saved. Use --resume to continue later",
        "resuming": "Resuming from previous state (Step {step})",
        "env_detect_gpu": "GPU detected: {info}",
        "env_detect_slurm": "SLURM detected: {info}",
    },
}

TOTAL_STEPS = 11


def get_message(msg_key: str, lang: str, **kwargs: str | int) -> str:
    """Get a localized message string, formatted with the given kwargs.

    Args:
        msg_key: Key into the MESSAGES dictionary.
        lang: Language code ('ja' or 'en').
        **kwargs: Format parameters for the message string.

    Returns:
        Formatted message string.
    """
    return MESSAGES[lang][msg_key].format(**kwargs)
