import os
import pandas as pd

INPUT_PATH = "inputs/results.csv"
OUTPUT_PATH = "outputs/validity_report.md"


def main():
    print("🔍 Starting metabolomics auditor...")

    print(f"Checking for input file at: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ results.csv not found")
        return
    else:
        print("✅ results.csv found")

    print("Reading CSV...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded data with shape: {df.shape}")

    n_rows, n_cols = df.shape

    def find_col(keywords):
        for k in keywords:
            for c in df.columns:
                if k in c.lower():
                    return c
        return None

    p_col = find_col(["p", "pvalue"])
    fdr_col = find_col(["fdr", "q", "adj"])
    fc_col = find_col(["log2fc", "logfc", "fold"])

    print("Detected columns:")
    print("p-value:", p_col)
    print("FDR:", fdr_col)
    print("Fold-change:", fc_col)

    # Scientific interpretation
    confidence = 100
    interpretations = []
    recommendations = []

    if fc_col and not p_col:
        confidence -= 40
        interpretations.append(
            "Fold-change values are present without corresponding p-values. "
            "This indicates exploratory analysis only; statistical significance cannot be assessed."
        )
        recommendations.append(
            "Run a statistical test (e.g., T-test or ANOVA) to obtain p-values."
        )

    if p_col and not fdr_col:
        confidence -= 20
        interpretations.append(
            "P-values are present without multiple-testing correction (FDR/q-values). "
            "This increases the risk of false positives in high-dimensional metabolomics data."
        )
        recommendations.append(
            "Apply false discovery rate (FDR) correction to control for multiple comparisons."
        )

    if p_col and fdr_col:
        interpretations.append(
            "Both p-values and FDR-adjusted q-values are present, indicating statistically interpretable results."
        )

    if confidence < 0:
        confidence = 0

    print("Creating outputs folder (if needed)...")
    os.makedirs("outputs", exist_ok=True)

    print("Writing validity report...")
    with open(OUTPUT_PATH, "w") as f:
        f.write("# Metabolomics Validity Report\n\n")

        f.write("## Dataset Overview\n")
        f.write(f"- Number of features: {n_rows}\n")
        f.write(f"- Number of columns: {n_cols}\n\n")

        f.write("## Detected Statistical Columns\n")
        f.write(f"- Fold change: {fc_col}\n")
        f.write(f"- p-value: {p_col}\n")
        f.write(f"- FDR / q-value: {fdr_col}\n\n")

        f.write("## Scientific Interpretation\n")
        if interpretations:
            for i in interpretations:
                f.write(f"- {i}\n")
        else:
            f.write("- No major statistical issues detected.\n")

        f.write("\n## Recommendations\n")
        if recommendations:
            for r in recommendations:
                f.write(f"- {r}\n")
        else:
            f.write("- No immediate corrective actions required.\n")

        f.write("\n## Overall Confidence Score\n")
        f.write(f"**{confidence} / 100**\n")

    print("✅ Validity report written to outputs/validity_report.md")



def run_audit(csv_path="inputs/results.csv", report_path="outputs/validity_report.md", json_path=None):
    """
    Streamlit/Cloud-friendly wrapper.
    Reads csv_path and writes report_path. Returns report markdown text.
    """
    import os
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    n_rows, n_cols = df.shape

    def find_col(keywords):
        for k in keywords:
            for c in df.columns:
                if k in c.lower():
                    return c
        return None

    p_col = find_col(["p", "pvalue"])
    fdr_col = find_col(["fdr", "q", "adj"])
    fc_col = find_col(["log2fc", "logfc", "fold"])

    confidence = 100
    interpretations = []
    recommendations = []

    if fc_col and not p_col:
        confidence -= 40
        interpretations.append(
            "Fold-change values are present without corresponding p-values. Statistical significance cannot be assessed."
        )
        recommendations.append("Run a statistical test (e.g., t-test or ANOVA) to obtain p-values.")

    if p_col and not fdr_col:
        confidence -= 20
        interpretations.append(
            "P-values are present without multiple-testing correction (FDR/q-values). This increases false positive risk."
        )
        recommendations.append("Apply false discovery rate (FDR) correction to control multiple comparisons.")

    if p_col and fdr_col:
        interpretations.append(
            "Both p-values and FDR-adjusted q-values are present, indicating statistically interpretable results."
        )

    confidence = max(confidence, 0)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    md = []
    md.append("# Metabolomics Validity Report\n")
    md.append("## Dataset Overview")
    md.append(f"- Number of features (rows): {n_rows}")
    md.append(f"- Number of columns: {n_cols}\n")

    md.append("## Detected Statistical Columns")
    md.append(f"- Fold change: {fc_col}")
    md.append(f"- p-value: {p_col}")
    md.append(f"- FDR / q-value: {fdr_col}\n")

    md.append("## Scientific Interpretation")
    if interpretations:
        for i in interpretations:
            md.append(f"- {i}")
    else:
        md.append("- No major statistical issues detected.\n")

    md.append("\n## Recommendations")
    if recommendations:
        for r in recommendations:
            md.append(f"- {r}")
    else:
        md.append("- No immediate corrective actions required.\n")

    md.append("\n## Overall Confidence Score")
    md.append(f"**{confidence} / 100**\n")

    report_text = "\n".join(md)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
def main():
    run_audit()

if __name__ == "__main__":
    main()
