def latex_table_from_dataframe(df, columns, float_cols=None):
    float_cols = set(float_cols or [])
    rows = []
    rows.append(" & ".join(latex_escape(col) for col in columns) + r" \\")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if col in float_cols and pd.notna(value):
                vals.append(f"{float(value):.3f}")
            else:
                vals.append(latex_escape(value))
        rows.append(" & ".join(vals) + r" \\")
    return "\n".join(rows)


def best_model_by_combined(df):
    best = df.sort_values("val50_combined_accuracy", ascending=False).iloc[0]
    return best["model"], float(best["val50_combined_accuracy"])


def get_summary_value(df, model_name, col):
    row = df[df["model"] == model_name]
    if row.empty:
        return np.nan
    return float(row.iloc[0][col])


def prompt_row(df, model_name, prompt_style):
    row = df[(df["model"] == model_name) & (df["prompt_style"] == prompt_style)]
    if row.empty:
        return None
    return row.iloc[0]

base_val50 = get_summary_value(final_experiment_comparison_df, "Base LLaVA", "val50_combined_accuracy")
orig_val50 = get_summary_value(final_experiment_comparison_df, "Original adapter q/v 1 epoch", "val50_combined_accuracy")
qv2_val50 = get_summary_value(final_experiment_comparison_df, "New adapter q/v 2 epochs", "val50_combined_accuracy")
qkvo_val50 = get_summary_value(final_experiment_comparison_df, "New adapter q/k/v/o 1 epoch", "val50_combined_accuracy")

best_model_name, best_model_acc = best_model_by_combined(final_experiment_comparison_df)

prompt_table_rows = []
for model_name in [
    "Base LLaVA",
    "Original adapter q/v 1 epoch",
    "New adapter q/v 2 epochs",
    "New adapter q/k/v/o 1 epoch",
]:
    for prompt_style in ["normal", "short"]:
        row = prompt_row(prompt_comparison_summary_df, model_name, prompt_style)
        if row is None:
            continue
        prompt_table_rows.append({
            "Model": model_name,
            "Prompt": prompt_style,
            "Combined": float(row["combined_accuracy"]),
            "Exact": float(row["exact_match"]),
            "Verbose": float(row["verbose_rate"]),
            "AvgLen": float(row["average_answer_length_words"]),
        })
prompt_table_df = pd.DataFrame(prompt_table_rows)

final_table_df = final_experiment_comparison_df.copy()
final_table_df = final_table_df.rename(columns={
    "target_modules": "Targets",
    "final_train_loss": "Loss",
    "val50_exact_match": "Exact",
    "val50_normalized_match": "Norm",
    "val50_numeric_tolerance": "Numeric",
    "val50_combined_accuracy": "Combined",
})
final_table_df["Model"] = final_table_df["model"]
final_table_df["Epochs"] = final_table_df["epochs"]
final_table_df = final_table_df[["Model", "Epochs", "Targets", "Loss", "Exact", "Norm", "Numeric", "Combined"]]

error_pivot = error_type_analysis_df.pivot(index="question_type", columns="model", values="combined_accuracy").reset_index()
for col in ["Base LLaVA", "Original adapter q/v 1 epoch", "New adapter q/v 2 epochs", "New adapter q/k/v/o 1 epoch"]:
    if col not in error_pivot.columns:
        error_pivot[col] = np.nan
error_counts = error_type_analysis_df.groupby("question_type", as_index=False)["n"].sum().rename(columns={"n": "Count"})
error_type_summary_small = error_counts.merge(error_pivot, on="question_type", how="left")

case_sections = []
for _, row in final_df.sort_values("index").iterrows():
    idx = int(row["index"])
    image_path = f"outputs/eval_images/eval_{idx}.png"
    case_sections.append(f'''
\\subsection*{{Case {idx}}}
\\begin{{center}}
\\includegraphics[width=0.78\\linewidth]{{{image_path}}}
\\end{{center}}
\\noindent\\textbf{{Question:}} {latex_escape(row["question"])}\\par
\\noindent\\textbf{{Ground Truth:}} {latex_escape(row["ground_truth"])}\\par
\\noindent\\textbf{{Base Model Answer:}} {latex_escape(row["base_answer"])}\\par
\\noindent\\textbf{{Fine-tuned Answer:}} {latex_escape(row["finetuned_answer"])}\\par
\\noindent\\textbf{{Analysis:}} {latex_escape(row["analysis"])}\\par
''')

train_summary = next((x for x in reversed(trainer_log_history) if "train_loss" in x), {})
train_loss = train_summary.get("train_loss", "N/A")
train_runtime = train_summary.get("train_runtime", "N/A")

integrated_tex = f'''
\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=0.8in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{fontspec}}
\\usepackage{{xeCJK}}
\\usepackage{{pdflscape}}
\\setmainfont{{Times New Roman}}
\\setCJKmainfont{{Microsoft JhengHei}}
\\hypersetup{{colorlinks=true, urlcolor=blue}}
\\title{{HW4: Multimodal AI - Visual Instruction Tuning (VQA)}}
\\author{{{STUDENT_ID} {STUDENT_NAME}}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\section*{{1. Title Page}}
\\textbf{{Title:}} HW4: Multimodal AI - Visual Instruction Tuning (VQA)\\\\
\\textbf{{Student:}} {STUDENT_ID} {STUDENT_NAME}\\\\
\\textbf{{GitHub Link:}} \\url{{{GITHUB_LINK}}}

\\section*{{2. Overview}}
This homework fine-tunes a vision-language model for domain-specific VQA. The dataset is ChartQA, the base model is LLaVA-v1.5-7B, and the method uses 4-bit quantization + QLoRA + PEFT LoRA adapters. Evaluation includes the original five case studies and an extended val[:50] quantitative analysis.

\\section*{{3. Dataset Choice}}
ChartQA is suitable because chart VQA requires numeric reading, color/label recognition, trend extraction, and precise visual reasoning. ChartQA labels are stored as lists, so the implementation uses \\texttt{{label[0]}} through \\texttt{{get\\_answer(sample)}} to avoid format errors.

\\section*{{4. Model and Method}}
Base model: \\texttt{{llava-hf/llava-1.5-7b-hf}}. Quantization: bitsandbytes 4-bit NF4 with double quantization. LoRA configuration uses \\texttt{{r=16}}, \\texttt{{alpha=32}}, and \\texttt{{dropout=0.05}}. Original adapter target modules are \\texttt{{["q\\_proj", "v\\_proj"]}}. The vision tower is frozen. Sequence truncation is avoided for VLM training because truncation may remove image tokens.

\\section*{{5. Training Setup}}
\\begin{{itemize}}
\\item Train split: \\texttt{{train[:1000]}}
\\item Validation split loaded: \\texttt{{val[:100]}}
\\item Original run: q/v, 1 epoch
\\item Additional run 1: q/v, 2 epochs
\\item Additional run 2: q/k/v/o, 1 epoch
\\item Learning rate: \\texttt{{2e-4}}
\\item Batch size: 1
\\item Gradient accumulation: 16
\\item Original run runtime: {latex_escape(train_runtime)} seconds
\\item Original run train loss: {latex_escape(train_loss)}
\\end{{itemize}}

\\section*{{6. Original 5-Case Evaluation}}
{''.join(case_sections)}

\\section*{{7. Original Loss Curve}}
\\begin{{center}}
\\includegraphics[width=0.85\\linewidth]{{outputs/loss_curve.png}}
\\end{{center}}

\\section*{{8. Additional Experiments Overview}}
Additional experiments were added because five examples are useful for qualitative analysis but too small for reliable quantitative conclusions. ChartQA val[:50] is used for a more stable evaluation. Prompt style, training epochs, and LoRA target modules are compared.

\\section*{{9. Larger Evaluation on ChartQA val[:50]}}
\\begin{{center}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
{latex_table_from_dataframe(larger_eval_summary_df, ["model", "exact_match", "normalized_match", "numeric_tolerance_match", "combined_accuracy"], ["exact_match", "normalized_match", "numeric_tolerance_match", "combined_accuracy"])}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
Base LLaVA combined accuracy is 0.240, while original adapter q/v 1 epoch combined accuracy is 0.180. The original adapter does not outperform the base model on val[:50].

\\section*{{10. Prompt Comparison}}
\\begin{{center}}
\\small
\\begin{{tabular}}{{llrrrr}}
\\toprule
{latex_table_from_dataframe(prompt_table_df, ["Model", "Prompt", "Combined", "Exact", "Verbose", "AvgLen"], ["Combined", "Exact", "Verbose", "AvgLen"])}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
Short-answer prompting reduces verbose outputs. Accuracy can still change because prompt formulation alters generation behavior.

\\section*{{11. Training Ablations}}
\\begin{{center}}
\\includegraphics[width=0.78\\linewidth]{{outputs/additional_experiments/loss_curve_qv_2epoch.png}}
\\end{{center}}
\\begin{{center}}
\\includegraphics[width=0.78\\linewidth]{{outputs/additional_experiments/loss_curve_qkvo_1epoch.png}}
\\end{{center}}
Comparison on val[:50] combined accuracy: Base LLaVA 0.240, original q/v 1 epoch 0.180, new q/v 2 epochs 0.220, new q/k/v/o 1 epoch 0.140. q/v 2 epochs improves over q/v 1 epoch but still does not exceed base. q/k/v/o 1 epoch is worse. Training loss decreases, but lower loss does not guarantee better validation accuracy.

\\section*{{12. Question Type Error Analysis}}
\\begin{{center}}
\\includegraphics[width=0.9\\linewidth]{{outputs/additional_experiments/error_type_accuracy.png}}
\\end{{center}}
\\begin{{center}}
\\small
\\begin{{tabular}}{{lrrrrr}}
\\toprule
{latex_table_from_dataframe(error_type_summary_small, ["question_type", "Count", "Base LLaVA", "Original adapter q/v 1 epoch", "New adapter q/v 2 epochs", "New adapter q/k/v/o 1 epoch"], ["Base LLaVA", "Original adapter q/v 1 epoch", "New adapter q/v 2 epochs", "New adapter q/k/v/o 1 epoch"])}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
Some question types have very small sample sizes, for example \\texttt{{comparison}} or \\texttt{{yes\\_no}}, so those bars should not be over-interpreted. Numeric questions are more representative because they make up most of val[:50]. Numeric and legend/label questions remain difficult, and exact chart-value reading is the main failure mode.

\\section*{{13. Final Comparison Table}}
\\begin{{center}}
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcrrrrrr}}
\\toprule
{latex_table_from_dataframe(final_table_df, ["Model", "Epochs", "Targets", "Loss", "Exact", "Norm", "Numeric", "Combined"], ["Loss", "Exact", "Norm", "Numeric", "Combined"])}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{center}}

\\section*{{14. Final Discussion}}
Best combined accuracy on val[:50] is Base LLaVA at 0.240. Among fine-tuned adapters, q/v 2 epochs is best at 0.220. More epochs helped compared with the original adapter, while more LoRA target modules did not help in this run. ChartQA remains difficult because many questions require OCR-like numeric extraction and exact label grounding.

\\section*{{15. Conclusion}}
The QLoRA pipeline works, the adapters train successfully, and training loss decreases. However, the fine-tuned adapters did not surpass the base model on val[:50]. The strongest adapter is q/v 2 epochs. Future work includes more data, more epochs, better assistant-only loss masking, stronger answer normalization, and larger validation evaluation.

\\end{{document}}
'''

Path('report.tex').write_text(integrated_tex, encoding='utf-8')
for _ in range(2):
    subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'report.tex'], check=True)

final_pdf = Path(OUTPUT_DIR) / f"HW4_{STUDENT_ID}_{STUDENT_NAME}.pdf"
shutil.copyfile('report.pdf', final_pdf)

for legacy in [
    Path(OUTPUT_DIR) / f"HW4_{STUDENT_ID}_{STUDENT_NAME}_extended.pdf",
    Path('extended_report.pdf'),
    Path('extended_report.tex'),
]:
    if legacy.exists():
        legacy.unlink()

print('Saved integrated final report to', final_pdf)
