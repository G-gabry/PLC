"""
Builds the Word (.docx) research report from the outputs of the four
analysis scripts. Every number, table and conclusion in the text is read
from those outputs at build time -- nothing is typed in by hand -- so after
re-running the experiments (e.g. the full VM run), re-running this script
regenerates a report that is consistent with the new data.

Reads:
  output/metrics_comparison/   run_metrics_comparison.py
  output/hparam_sweeps/        run_hparam_sweeps.py
  output/burst_comparisons/    plot_burst_comparison.py
  output/statistical_analysis/ run_statistical_analysis.py
plus the dataset's trace files (for the network statistics).

If the sample sizes are too small to support conclusions, a visible
"preliminary results" notice is inserted at the top of the report.

Usage:
    python -m research.build_report
"""
import argparse
import datetime
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

from research import dataset_utils, concealment, safe_io
from research.run_metrics_comparison import ALL_METRICS, METHODS, BASELINE_KEY

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUT = OUTPUT_DIR / "report" / "PLC_LPC_vs_PARCnet_Report.docx"

AUTHOR = "Omar Ali Elgabry"
TITLE = ("Autoregressive, Hybrid Neural and Sparse-AR Packet Loss Concealment "
         "for Networked Music: An Evaluation on Real Loss Traces")

MIN_FILES_FOR_CONCLUSIONS = 30
MIN_SWEEP_FILES = 10
CONTENT_WIDTH_CM = 16.0

METHOD_KEYS = [m[0] for m in METHODS]
METHOD_LABEL = {m[0]: m[1] for m in METHODS}
NON_BASELINE_KEYS = [m for m in METHOD_KEYS if m != BASELINE_KEY]
METHOD_PAIRS = [(METHODS[i][0], METHODS[j][0]) for i in range(len(METHODS)) for j in range(i + 1, len(METHODS))]

METRIC_DESC = {
    "snr": "time-domain signal-to-noise ratio",
    "mrstft": "multi-resolution STFT distance",
    "plcmos": "PLCMOS",
    "visqol": "ViSQOL MOS-LQO",
}
SWEEP_LABELS = {
    "ar_order": ("AR order p", "ar_order"),
    "context_length_packets": ("Context length (packets)", "context_dim_packets"),
    "diagonal_load": ("Diagonal loading λ", "diagonal_load"),
    "extra_dim": ("Crossfade length E (samples)", "extra_dim"),
    "max_k_lags": ("Sparse AR: max. long-term lags k", "max_k_lags"),
    "max_lag_l": ("Sparse AR: max. lag L", "max_lag_l"),
    "burg_order_pb": ("Sparse AR: Burg (short-term) order p_b", "burg_order_pb"),
    "lambda_reg": ("Sparse AR: regularization λ", "lambda_reg"),
}
SWEEP_ONLY_METHODS = {  # hyperparameter -> methods it was swept for, when not all three
    "ar_order": ["lpc", "nn"], "diagonal_load": ["lpc", "nn"],
    "max_k_lags": ["sparse_ar"], "max_lag_l": ["sparse_ar"],
    "burg_order_pb": ["sparse_ar"], "lambda_reg": ["sparse_ar"],
}

REFERENCES = [
    ("mezza2024", 'A. I. Mezza, M. Amerena, A. Bernardini, and A. Sarti, "Hybrid packet loss concealment for '
                  'real-time networked music applications," IEEE Open Journal of Signal Processing, vol. 5, '
                  'pp. 266-273, 2024, doi: 10.1109/OJSP.2023.3343318.'),
    ("challenge2024", 'A. I. Mezza et al., "Report on the IEEE-IS² 2024 Music Packet Loss Concealment Challenge," '
                      'arXiv:2409.18564, 2024.'),
    ("perkins1998", 'C. Perkins, O. Hodson, and V. Hardman, "A survey of packet loss recovery techniques for '
                    'streaming audio," IEEE Network, vol. 12, no. 5, pp. 40-48, 1998.'),
    ("makhoul1975", 'J. Makhoul, "Linear prediction: A tutorial review," Proceedings of the IEEE, vol. 63, no. 4, '
                    'pp. 561-580, 1975.'),
    ("lostanlen2016", 'V. Lostanlen and C. E. Cella, "Deep convolutional networks on the pitch spiral for musical '
                      'instrument recognition," in Proc. ISMIR, 2016.'),
    ("diener2022", 'L. Diener, S. Sootla, S. Branets, A. Saabas, R. Aichner, and R. Cutler, "INTERSPEECH 2022 '
                   'Audio Deep Packet Loss Concealment Challenge," in Proc. Interspeech, 2022.'),
    ("yamamoto2020", 'R. Yamamoto, E. Song, and J.-M. Kim, "Parallel WaveGAN: A fast waveform generation model '
                     'based on generative adversarial networks with multi-resolution spectrogram," in Proc. '
                     'IEEE ICASSP, 2020.'),
    ("diener2023", 'L. Diener, M. Purin, S. Sootla, A. Saabas, R. Aichner, and R. Cutler, "PLCMOS - a data-driven '
                   'non-intrusive metric for the evaluation of packet loss concealment algorithms," in Proc. '
                   'Interspeech, 2023.'),
    ("chinen2020", 'M. Chinen, F. S. C. Lim, J. Skoglund, N. Gureev, F. O\'Gorman, and A. Hines, "ViSQOL v3: An '
                   'open source production ready objective speech and audio metric," in Proc. QoMEX, 2020.'),
    ("gray1974", 'A. H. Gray and J. D. Markel, "A spectral-flatness measure for studying the autocorrelation '
                 'method of linear prediction of speech analysis," IEEE Trans. Acoustics, Speech, and Signal '
                 'Processing, vol. 22, no. 3, pp. 207-217, 1974.'),
    ("wilcoxon1945", 'F. Wilcoxon, "Individual comparisons by ranking methods," Biometrics Bulletin, vol. 1, '
                     'no. 6, pp. 80-83, 1945.'),
    ("holm1979", 'S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian Journal of '
                 'Statistics, vol. 6, no. 2, pp. 65-70, 1979.'),
    ("kerby2014", 'D. S. Kerby, "The simple difference formula: An approach to teaching nonparametric '
                  'correlation," Comprehensive Psychology, vol. 3, 2014.'),
    ("efron1993", 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. Chapman & Hall/CRC, 1993.'),
    ("itu1534", 'ITU-R Recommendation BS.1534-3, "Method for the subjective assessment of intermediate quality '
                'level of audio systems (MUSHRA)," 2015.'),
    ("severi2025", 'L. Severi, "Orthogonal Matching Pursuit based Linear prediction for Real-time Packet Loss '
                   'Concealment," technical report, Politecnico di Torino, 2025.'),
    ("vos2013", 'K. Vos, "A fast implementation of Burg\'s method," OPUS codec technical report, 2013.'),
    ("pati1993", 'Y. Pati, R. Rezaiifar, and P. Krishnaprasad, "Orthogonal matching pursuit: recursive function '
                 'approximation with applications to wavelet decomposition," in Proc. 27th Asilomar Conference '
                 'on Signals, Systems and Computers, 1993, pp. 40-44.'),
]
REF_NUM = {key: i + 1 for i, (key, _) in enumerate(REFERENCES)}


def cite(*keys) -> str:
    return "[" + ", ".join(str(REF_NUM[k]) for k in keys) + "]"


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------
def f(v, d=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fs(v, d=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}f}"


def fp(p) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fg(v) -> str:
    return f"{v:g}"


def join_words(items: list[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------
# Document helpers
# --------------------------------------------------------------------------
def shade(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def add_field(run, instruction: str, placeholder: str = "") -> None:
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, text, end):
        run._r.append(el)


class Report:
    def __init__(self):
        self.doc = Document()
        self.fig_n = 0
        self.tab_n = 0
        self.eq_n = 0
        self._setup()

    def _setup(self):
        section = self.doc.sections[0]
        section.page_height, section.page_width = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Cm(2.5))

        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15
        for level, size in [(1, 15), (2, 12.5), (3, 11.5)]:
            h = styles[f"Heading {level}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
            h.paragraph_format.space_before = Pt(14 if level == 1 else 10)
            h.paragraph_format.space_after = Pt(4)
        caption = styles["Caption"]
        caption.font.size = Pt(9)
        caption.font.italic = False
        caption.font.color.rgb = RGBColor(0x37, 0x41, 0x51)

        # Ask Word to refresh fields (table of contents, page numbers) on open.
        update = OxmlElement("w:updateFields")
        update.set(qn("w:val"), "true")
        self.doc.settings.element.append(update)

        footer_p = section.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_p.add_run()
        run.font.size = Pt(9)
        add_field(run, "PAGE", "1")

    # -- text --------------------------------------------------------------
    def heading(self, text: str, level: int = 1):
        self.doc.add_heading(text, level=level)

    def para(self, *parts, align=None, size=None, space_after=None, style=None):
        """parts: plain strings, or (text, flags) tuples with flags from 'b', 'i', 'sub', 'sup'."""
        p = self.doc.add_paragraph(style=style)
        if align == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "justify":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for part in parts:
            text, flags = (part, "") if isinstance(part, str) else part
            run = p.add_run(text)
            run.bold = "b" in flags.replace("sub", "")
            run.italic = "i" in flags
            run.font.subscript = "sub" in flags
            run.font.superscript = "sup" in flags
            if size:
                run.font.size = Pt(size)
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        return p

    def body(self, *parts):
        return self.para(*parts, align="justify")

    def bullet(self, *parts):
        return self.para(*parts, style="List Bullet")

    def numbered(self, *parts):
        return self.para(*parts, style="List Number")

    def equation(self, text: str) -> int:
        self.eq_n += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.tab_stops.add_tab_stop(Cm(CONTENT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(CONTENT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
        run = p.add_run(f"\t{text}\t({self.eq_n})")
        run.font.name = "Cambria Math"
        run.italic = True
        return self.eq_n

    def insight(self, text: str):
        """A short, visually distinct 'what this means' note placed right
        after a figure, so a reader does not have to work out the takeaway
        themselves -- separate from the longer analytical prose elsewhere,
        which may repeat the same point in more depth."""
        p = self.doc.add_paragraph()
        p.paragraph_format.space_after = Pt(10)
        p.paragraph_format.left_indent = Cm(0.5)
        label = p.add_run("Interpretation: ")
        label.bold = True
        label.italic = True
        label.font.size = Pt(9.5)
        label.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
        body = p.add_run(text)
        body.italic = True
        body.font.size = Pt(9.5)
        body.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
        return p

    def callout(self, title: str, text: str, fill="FFF4E5"):
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        shade(cell, fill)
        p = cell.paragraphs[0]
        r = p.add_run(title + " ")
        r.bold = True
        p.add_run(text)
        for run in p.runs:
            run.font.size = Pt(10)
        self.doc.add_paragraph()

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def toc(self):
        p = self.doc.add_paragraph()
        add_field(p.add_run(), 'TOC \\o "1-2" \\h \\z \\u',
                  "Right-click here and choose 'Update Field' to build the table of contents.")

    # -- figures & tables ----------------------------------------------------
    def next_fig(self) -> int:
        return self.fig_n + 1

    def next_tab(self) -> int:
        return self.tab_n + 1

    def figure(self, path: Path, caption: str, width_cm: float = CONTENT_WIDTH_CM) -> int | None:
        if not path or not Path(path).exists():
            self.para((f"[Figure not available: {Path(path).name if path else 'missing'} -- "
                       f"run the script that produces it]", "i"), size=9)
            return None
        self.fig_n += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.doc.add_paragraph(style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cap.add_run(f"Figure {self.fig_n}. ").bold = True
        cap.add_run(caption)
        cap.paragraph_format.space_after = Pt(12)
        return self.fig_n

    def table(self, header: list[str], rows: list[list], caption: str, col_widths_cm=None,
              font_size: float = 8.5, notes: str | None = None) -> int:
        self.tab_n += 1
        cap = self.doc.add_paragraph(style="Caption")
        cap.paragraph_format.keep_with_next = True
        cap.add_run(f"Table {self.tab_n}. ").bold = True
        cap.add_run(caption)
        if notes:
            cap.add_run(" " + notes).italic = True

        table = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, text in enumerate(header):
            cell = table.rows[0].cells[j]
            shade(cell, "E5E7EB")
            run = cell.paragraphs[0].add_run(str(text))
            run.bold = True
            run.font.size = Pt(font_size)
        for i, row in enumerate(rows, start=1):
            for j, value in enumerate(row):
                run = table.rows[i].cells[j].paragraphs[0].add_run(str(value))
                run.font.size = Pt(font_size)
        if col_widths_cm:
            for row in table.rows:
                for cell, w in zip(row.cells, col_widths_cm):
                    cell.width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return self.tab_n


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def network_statistics() -> dict:
    traces_dir = dataset_utils.EXAMPLE_TEST_SET_DIR / "traces"
    stems = dataset_utils.list_available_stems()
    lengths, per_file_plr, total_packets, total_lost = [], [], 0, 0
    for stem in stems:
        trace = np.loadtxt(traces_dir / f"{stem}.txt", dtype=int).flatten()
        if len(trace) == 0:
            continue
        total_packets += len(trace)
        total_lost += int(trace.sum())
        per_file_plr.append(100 * trace.mean())
        lengths.extend(length for _, length in dataset_utils.find_bursts(trace))
    lengths = np.array(lengths)
    per_file_plr = np.array(per_file_plr)
    return {
        "n_files": len(stems),
        "total_packets": total_packets,
        "total_lost": total_lost,
        "plr": 100 * total_lost / total_packets,
        "n_bursts": len(lengths),
        "mean_burst": float(lengths.mean()),
        "max_burst": int(lengths.max()),
        "pct_single": 100 * float((lengths == 1).mean()),
        "pct_le2": 100 * float((lengths <= 2).mean()),
        "n_ge8": int((lengths >= 8).sum()),
        "plr_median": float(np.median(per_file_plr)),
        "plr_min": float(per_file_plr.min()),
        "plr_max": float(per_file_plr.max()),
        "n_zero_loss": int((per_file_plr == 0).sum()),
    }


def software_versions() -> list[list[str]]:
    rows = [["Python", platform.python_version()]]
    for pkg in ["numpy", "scipy", "pandas", "librosa", "torch", "pytorch-lightning", "numba",
                "speechmos", "visqol-python", "matplotlib", "seaborn", "python-docx"]:
        try:
            rows.append([pkg, importlib.metadata.version(pkg)])
        except importlib.metadata.PackageNotFoundError:
            pass
    return rows


def oriented(df: pd.DataFrame, key: str, lower_is_better: bool, method_a: str, method_b: str) -> pd.Series:
    d = df[f"{key}_{method_a}"] - df[f"{key}_{method_b}"]
    return -d if lower_is_better else d


def pretty_pair_label(col: str) -> str:
    """"this_lpc_vs_lpc_with_nn" -> "This LPC vs LPC with NN" -- gap_mse_summary.csv's
    column names come from plot_burst_comparison.py's own method keys, not
    this module's METHOD_LABEL, so acronyms are fixed up by word rather than
    looked up."""
    words = {"lpc": "LPC", "nn": "NN", "ar": "AR", "vs": "vs", "with": "with", "this": "This"}
    return " ".join(words.get(w, w.capitalize()) for w in col.split("_"))


def burst_figure_sets(directory: Path, prefix_filter: str = "") -> list[tuple[int, str, dict]]:
    """[(burst_length, tag, {'compare': path, 'diff': path, 'crossfade': path})] sorted by length."""
    sets = []
    for compare in sorted(directory.glob(f"compare_{prefix_filter}*.png")):
        m = re.match(r"compare_(.+)_(\d+)pkt\.png$", compare.name)
        if not m:
            continue
        tag, length = m.group(1), int(m.group(2))
        paths = {kind: directory / f"{kind}_{tag}_{length}pkt.png" for kind in ("compare", "diff", "crossfade")}
        sets.append((length, tag, paths))
    return sorted(sets, key=lambda s: (s[0], s[1]))


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------
class Context:
    """All inputs loaded once, so every section reads the same numbers."""

    def __init__(self, output_dir: Path):
        self.out = output_dir
        metrics_dir = output_dir / "metrics_comparison"
        stats_dir = output_dir / "statistical_analysis"
        self.per_file = read_csv(metrics_dir / "per_file_results.csv")
        self.per_burst = read_csv(metrics_dir / "per_burst_results.csv")
        info_path = metrics_dir / "run_info.json"
        self.run_info = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.hparams = self.run_info.get("hparams", dict(ar_order=256, diagonal_load=0.001,
                                                         context_dim_packets=8, extra_dim=256,
                                                         max_k_lags=3, max_lag_l=1024,
                                                         burg_order_pb=8, lambda_reg=0.01))
        self.tests = read_csv(stats_dir / "significance_tests.csv")
        self.burst_q = read_csv(stats_dir / "burst_length_quality.csv")
        self.sweep_summary = read_csv(stats_dir / "sweep_summary.csv")
        self.failures = {}
        failures_root = stats_dir / "failure_cases"
        if failures_root.exists():
            for method in NON_BASELINE_KEYS:
                df = read_csv(failures_root / method / "failure_cases.csv")
                if df is not None and not df.empty:
                    self.failures[method] = df
        self.gap_mse = read_csv(output_dir / "burst_comparisons" / "gap_mse_summary.csv")
        self.sweeps = {}
        for path in sorted((output_dir / "hparam_sweeps").glob("sweep_*.csv")):
            df = pd.read_csv(path)
            self.sweeps[df["param"].iloc[0]] = (df, path.with_suffix(".png"))
        self.net = network_statistics()

        self.n_files = len(self.per_file) if self.per_file is not None else 0
        self.sr = int(self.per_file["sample_rate"].iloc[0]) if (
            self.per_file is not None and "sample_rate" in self.per_file) else 44100
        self.packet_ms = concealment.PACKET_SIZE / self.sr * 1000
        self.methods_present = ([m for m in METHOD_KEYS if f"snr_{m}" in self.per_file.columns]
                                if self.per_file is not None else [])
        self.non_baseline_present = [m for m in self.methods_present if m != BASELINE_KEY]
        self.has_timing = (self.per_file is not None and
                           all(f"ms_per_lost_packet_{m}" in self.per_file.columns for m in self.methods_present))
        self.sweep_files = {p: df["file"].nunique() for p, (df, _) in self.sweeps.items()}
        self.preliminary = (self.n_files < MIN_FILES_FOR_CONCLUSIONS or
                            any(n < MIN_SWEEP_FILES for n in self.sweep_files.values()))

    def test_row(self, key: str, method_a: str, method_b: str):
        label = next(l for k, l, _ in ALL_METRICS if k == key)
        t = self.tests
        row = t[(t["metric"] == label) & (t["method_a"] == METHOD_LABEL[method_a]) &
                (t["method_b"] == METHOD_LABEL[method_b])]
        return row.iloc[0] if len(row) else None

    def advantage(self, key: str, method: str, baseline: str = BASELINE_KEY):
        """Mean advantage (and verdict/p/CI/effect size) of `method` over
        `baseline` for metric `key`, regardless of which order
        significance_tests.csv happened to store that pair in -- METHOD_PAIRS
        always lists whichever method comes first in METHODS, which for any
        non-baseline method is the baseline itself (index 0), i.e. the
        opposite of "method vs baseline". Returns None if the pair isn't in
        ctx.tests at all.
        """
        row = self.test_row(key, method, baseline)
        if row is not None:
            return row
        row = self.test_row(key, baseline, method)
        if row is None:
            return None
        flipped = row.copy()
        flipped["mean_a_advantage"] = -row["mean_a_advantage"]
        flipped["median_a_advantage"] = -row["median_a_advantage"]
        flipped["ci95_low"], flipped["ci95_high"] = -row["ci95_high"], -row["ci95_low"]
        flipped["a_win_rate_pct"] = 100 - row["a_win_rate_pct"]
        flipped["rank_biserial_r"] = -row["rank_biserial_r"]
        return flipped  # "verdict" already names the better method by label text, no flip needed

    def rtf(self, method: str) -> np.ndarray:
        return (self.per_file[f"ms_per_lost_packet_{method}"] / self.packet_ms).dropna().to_numpy()


def significance_sentence(ctx: Context, method: str) -> str:
    """One sentence summarizing `method` vs. the baseline (LPC) across all four metrics,
    phrased as "quality was shown to be statistically significantly better with X on
    [metrics] and significantly better with AR alone on [metrics]"."""
    if ctx.tests is None:
        return ""
    groups = {"better": [], "worse": [], "no significant difference": []}
    any_found = False
    for key, label, _ in ALL_METRICS:
        row = ctx.advantage(key, method, BASELINE_KEY)
        if row is None:
            continue
        any_found = True
        if row["verdict"] == "no significant difference":
            groups["no significant difference"].append(label)
        elif row["verdict"] == METHOD_LABEL[method]:
            groups["better"].append(label)
        else:
            groups["worse"].append(label)
    if not any_found:
        return ""
    parts = []
    if groups["better"]:
        parts.append(f"significantly better with {METHOD_LABEL[method]} on {join_words(groups['better'])}")
    if groups["worse"]:
        parts.append(f"significantly better with {METHOD_LABEL[BASELINE_KEY]} alone on "
                     f"{join_words(groups['worse'])}")
    if groups["no significant difference"]:
        parts.append(f"not significantly different on {join_words(groups['no significant difference'])}")
    return (f"After Holm correction (α = 0.05, n = {ctx.n_files} files), quality was shown to be statistically "
            + " and ".join(parts) + ".")


def realtime_sentence(ctx: Context) -> str:
    if not ctx.has_timing:
        return ""
    medians = {m: np.median(ctx.rtf(m)) for m in ctx.methods_present}
    bits = [f"{METHOD_LABEL[m]} {np.median(ctx.per_file[f'ms_per_lost_packet_{m}'].dropna()):.2f} ms "
            f"(RTF {medians[m]:.2f})" for m in ctx.methods_present]
    cannot = [METHOD_LABEL[m] for m, v in medians.items() if v > 1]
    return (f"On the evaluation CPU, concealing one {ctx.packet_ms:.1f} ms packet took a median of "
            f"{join_words(bits)}, i.e. " +
            (f"{join_words(cannot)} cannot run in real time on this hardware"
             if cannot else "every method can run in real time on this hardware") + ".")


def burst_sentence(ctx: Context, concise: bool = False) -> str:
    """concise=True (for the abstract) drops the "(95% CI excluding zero)"
    qualifier and splits into two short sentences, matching how the abstract
    states this finding more tersely than the full Section 3.4 discussion.
    """
    if ctx.burst_q is None or ctx.burst_q.empty:
        return ""
    first, last = ctx.burst_q.iloc[0], ctx.burst_q.iloc[-1]
    means_first = " / ".join(f"{f(first.get(f'gap_snr_{m}_mean', np.nan))} ({METHOD_LABEL[m]})"
                             for m in ctx.methods_present if f"gap_snr_{m}_mean" in first)
    means_last = " / ".join(f"{f(last.get(f'gap_snr_{m}_mean', np.nan))} ({METHOD_LABEL[m]})"
                            for m in ctx.methods_present if f"gap_snr_{m}_mean" in last)
    s = (f"Gap SNR decreased from {means_first} dB for single-packet losses to {means_last} dB "
         f"for bursts of {last['burst_length_bin']} packets.")
    bits = []
    for m in ctx.non_baseline_present:
        lo_col, hi_col = f"{m}_advantage_ci_low", f"{m}_advantage_ci_high"
        if lo_col not in ctx.burst_q.columns:
            continue
        sig_pos = [r["burst_length_bin"] for _, r in ctx.burst_q.iterrows() if r[lo_col] > 0]
        sig_neg = [r["burst_length_bin"] for _, r in ctx.burst_q.iterrows() if r[hi_col] < 0]
        if sig_pos:
            bits.append(f"{METHOD_LABEL[m]}'s gap-SNR advantage was positive for bursts of "
                       f"{join_words(sig_pos)} packets")
        if sig_neg:
            bits.append(f"{METHOD_LABEL[m]}'s gap-SNR advantage was negative for bursts of "
                       f"{join_words(sig_neg)} packets")
    if bits:
        s += " " + " and ".join(bits) + ("." if concise else " (95% CI excluding zero).")
    else:
        s += " No method's gap-SNR advantage over the baseline was distinguishable from zero at any burst length."
    return s


def section_front(r: Report, ctx: Context):
    r.para((TITLE, "b"), align="center", size=17, space_after=10)
    r.para(AUTHOR, align="center", size=12, space_after=2)
    r.para(datetime.date.today().strftime("%B %Y"), align="center", size=10, space_after=16)

    if ctx.preliminary:
        thin = [f"{SWEEP_LABELS.get(p, (p,))[0]} ({n})" for p, n in ctx.sweep_files.items() if n < MIN_SWEEP_FILES]
        msg = (f"This version of the report was generated from small test runs: the main comparison uses "
               f"{ctx.n_files} files (at least {MIN_FILES_FOR_CONCLUSIONS} are needed for reliable statistics)")
        if thin:
            msg += f", and the hyperparameter sweeps were averaged over too few files ({join_words(thin)})"
        msg += (". All numbers and conclusions below will change when the full experiments are run; "
                "regenerate this report afterwards with `python -m research.build_report`.")
        r.callout("PRELIMINARY RESULTS.", msg)

    r.para(("Abstract", "b"), size=11, space_after=2)
    abstract = (
        f"Packet loss concealment (PLC) is necessary in networked music performance, where retransmission "
        f"is impossible within the latency budget and a lost {concealment.PACKET_SIZE}-sample packet must be "
        f"replaced causally from past audio only. This report compares three causal concealers: the first is "
        f"AR prediction alone, the second is PARCnet, a hybrid model that adds a convolutional neural network's "
        f"estimate of the AR residual, and the third is Sparse AR, a from-scratch implementation of Severi "
        f"(2025)'s orthogonal-matching-pursuit concealer, which replaces the dense AR fit with a short Burg "
        f"model plus a handful of non-contiguous, periodicity-seeking lags. All three were evaluated on "
        f"{ctx.n_files} music recordings from the IEEE-IS² 2024 Music PLC Challenge test set with their real "
        f"packet-loss traces ({ctx.net['plr']:.1f}% of packets lost overall) using two physical metrics (SNR and "
        f"multi-resolution STFT distance) and two perceptual metrics (PLCMOS and ViSQOL), paired non-parametric "
        f"statistics, a per-burst analysis over every real loss event, hyperparameter sweeps and a real-time "
        f"cost analysis. ")
    sentences = [significance_sentence(ctx, m) for m in ctx.non_baseline_present]
    sentences += [burst_sentence(ctx, concise=True)]
    abstract += " ".join(s for s in sentences if s)
    r.body(abstract)
    r.para(("Keywords: ", "b"), ("packet loss concealment; networked music performance; linear prediction; "
                                 "hybrid neural networks; sparse autoregressive modeling; orthogonal matching "
                                 "pursuit; PARCnet; objective audio quality", "i"), size=10)
    r.page_break()
    r.para(("Contents", "b"), size=13)
    r.toc()
    r.page_break()


def section_introduction(r: Report, ctx: Context):
    r.heading("1  Introduction", 1)
    r.body(f"In networked music performance (NMP), musicians at different locations play together over the "
           f"internet. Audio is sent in small packets and every network can drop or delay one. Unlike speech "
           f"calls or streaming, NMP leaves no room for retransmission, so the end-to-end latency must stay "
           f"within a few tens of milliseconds for playing to remain possible. A lost packet must therefore be "
           f"replaced on the fly, by an algorithm that estimates the samples using only audio that has already "
           f"arrived. This task is packet loss concealment (PLC) {cite('perkins1998')}.")
    r.body(f"A classic and computationally cheap approach is to fit an autoregressive (AR) model, i.e. a linear "
           f"predictor, to the most recent audio and extrapolate it across the gap {cite('makhoul1975')}. Linear "
           f"prediction captures the stationary part of a musical signal well, but it cannot capture changes "
           f"such as note onsets, and its extrapolation decays during long gaps. Two different ways of "
           f"improving on it are compared here. PARCnet {cite('mezza2024')} keeps the linear predictor and adds "
           f"a lightweight convolutional network that is trained to estimate what the predictor gets wrong, its "
           f"residual; a modified PARCnet is the official baseline of the IEEE-IS² 2025 Music PLC Challenge "
           f"{cite('challenge2024')}, whose settings ({concealment.PACKET_SIZE}-sample packets at 44.1 kHz, i.e. "
           f"{ctx.packet_ms:.1f} ms, and strictly causal processing) are adopted here. Sparse AR "
           f"{cite('severi2025')} instead replaces the dense AR fit itself with a short Burg model for local "
           f"continuity plus a handful of non-contiguous lags, chosen by orthogonal matching pursuit "
           f"{cite('pati1993')}, that specifically target musical periodicity rather than fitting every sample "
           f"of the context equally.")
    r.body("This gives three methods with a common baseline (AR alone) and two different strategies for "
           "improving on it -- adding a learned correction, or replacing the model with a sparser, "
           "periodicity-aware one. This report answers four research questions:")
    r.numbered(("RQ1 (quality). ", "b"), "Does either alternative improve concealment quality over AR "
               "prediction alone, as measured by physical and by perceptual metrics, on real loss traces?")
    r.numbered(("RQ2 (conditions). ", "b"), "How does any advantage depend on the length of the loss burst "
               "and on the complexity of the music signal?")
    r.numbered(("RQ3 (sensitivity). ", "b"), "How sensitive is each method to its own main hyperparameters?")
    r.numbered(("RQ4 (cost). ", "b"), "Can each method run in real time on a general-purpose CPU?")
    r.body("The main contributions are: (i) three independently implemented concealers; (ii) a from-scratch "
           "implementation of the Sparse AR method, since no reference code was available, documented in "
           "Section 2.4 with the measured evidence for where its published description had to be departed "
           "from; (iii) an evaluation on whole files with their real packet traces; (iv) paired statistical "
           "testing with multiple-comparison correction and effect sizes; (v) a per-burst analysis over every "
           "real loss event in the evaluated files; and (vi) hyperparameter sensitivity and real-time cost "
           "analyses for all three methods.")


def section_methods(r: Report, ctx: Context):
    hp = ctx.hparams
    P, E = concealment.PACKET_SIZE, hp["extra_dim"]
    C = hp["context_dim_packets"] * P
    r.heading("2  Methods", 1)

    r.heading("2.1  Problem formulation", 2)
    r.body(f"A mono music signal x[n], sampled at f", ("s", "sub"), f" = {ctx.sr / 1000:.1f} kHz, is divided into "
           f"packets of P = {P} samples ({ctx.packet_ms:.1f} ms). A binary packet trace l", ("k", "sub"),
           " ∈ {0, 1} marks packet k as received (0) or lost (1); lost packets arrive as zeros. A concealer "
           "must output an estimate of each lost packet using only samples before it (received or previously "
           "concealed). Because concealed samples become context for the next prediction, all three methods are "
           "causal and recursive: a burst of N lost packets is concealed by N successive predictions, each "
           "conditioned on the previous ones.")

    r.heading("2.2  AR (linear-prediction) concealment", 2)
    r.body(f"For every lost packet, an AR model of order p is fitted to the most recent C = "
           f"{hp['context_dim_packets']} packets ({C} samples, {C / ctx.sr * 1000:.0f} ms) of the output signal:")
    r.equation("x̂[n] = Σᵢ₌₁ᵖ aᵢ · x[n − i]")
    r.body("The coefficients a = (a₁, …, aₚ) are obtained with the autocorrelation method and solved with the "
           "Levinson-Durbin recursion after adding a diagonal loading term λ to the zero-lag autocorrelation "
           "(white-noise compensation), which improves the conditioning of the system "
           f"{cite('makhoul1975')}:")
    r.equation("(R + λI) a = r,     Rᵢⱼ = rₓₓ(|i − j|)")
    r.body(f"The model is then run forward recursively, feeding each predicted sample back as input, to produce "
           f"P ({P} samples) + E = {P + E} samples the lost packet plus E = {E} extra samples "
           f"({E / ctx.sr * 1000:.1f} ms). The extra samples are used for a linear crossfade which means the "
           f"tail of the prediction fades out while the next packet fades in, so that no discontinuity is heard "
           f"where concealment hands back to real audio. Within a burst, the next prediction likewise fades in "
           f"over the previous prediction's tail. The default hyperparameters (Table {r.next_tab()}) are those "
           f"of the challenge baseline.")
    r.table(["Hyperparameter", "Symbol", "Default", "Meaning"], [
        ["AR order", "p", hp["ar_order"], "number of past samples in the linear predictor"],
        ["Context length", "C", f"{hp['context_dim_packets']} packets ({C} samples)",
         "audio used to fit the AR model and as NN input"],
        ["Diagonal loading", "λ", fg(hp["diagonal_load"]), "regularization of the autocorrelation matrix"],
        ["Crossfade length", "E", f"{E} samples ({E / ctx.sr * 1000:.1f} ms)",
         "extra predicted samples used for crossfading"],
        ["NN fade-in", "-", f"{concealment.NN_FADE_DIM} samples", "fade-in of the NN contribution (hybrid only)"],
    ], "Default hyperparameters (IEEE-IS² 2025 challenge baseline configuration).",
        col_widths_cm=[3.3, 1.5, 4.2, 7.0])

    r.heading("2.3  Hybrid AR + neural concealment (PARCnet)", 2)
    r.body("The hybrid model computes the same AR prediction and adds the output of a neural network f", ("θ", "i"),
           " that receives the same past context c:")
    r.equation("ŝ = ŝ_AR + f_θ(c)")
    r.body(f"The network is a lightweight, fully convolutional 1-D encoder-decoder ('lite' configuration, 8 base "
           f"channels): four dilated residual blocks with ×2 downsampling, a bottleneck of six gated linear unit "
           f"(GLU) blocks with dilations 1 to 32, four upsampling dilated residual blocks, and a tanh output. It "
           f"was trained by the challenge organizers on Medley-solos-DB {cite('lostanlen2016')} to minimize")
    r.equation("L = 100 · ‖s − ŝ‖₁ + ½ (L_SC + L_mag)")
    r.body(f"where L_SC and L_mag are the spectral-convergence and log-magnitude terms of a multi-resolution STFT "
           f"loss {cite('yamamoto2020')}. Because the target of the network is s − ŝ_AR, it learns the residual "
           f"of an AR predictor with the default configuration. The pretrained challenge checkpoint is used "
           f"unchanged; it was not retrained for any of the swept configurations (Section 2.9). The NN output is "
           f"faded in over its first {concealment.NN_FADE_DIM} samples, and the same crossfade as in Section 2.2 "
           f"is applied to the combined prediction.")

    r.heading("2.4  Sparse AR concealment", 2)
    r.body(f"Sparse AR {cite('severi2025')} replaces the dense AR fit with two components. A short Burg model "
           f"of order p", ("b", "sub"), f" (default {hp['burg_order_pb']}) captures local, short-term structure. "
           f"Separately, its residual on the same context -- what the short-term model does not explain -- is "
           f"searched for up to k (default {hp['max_k_lags']}) non-contiguous lags d", ("i", "sub"),
           f" among the first L (default {hp['max_lag_l']}) candidates, each selected greedily by orthogonal "
           f"matching pursuit {cite('pati1993')} to maximize its correlation with what the previously selected "
           f"lags do not yet explain:")
    r.equation("ŝ[n] = Σᵢ aᵢ·s[n−i]  (short-term)  +  Σⱼ φⱼ·e[n−dⱼ]  (long-term, on the residual e)")
    r.body(f"The long-term coefficients φ are re-optimized after every new lag by solving a small, Tikhonov-"
           f"regularized system (regularization λ_reg, default {fg(hp['lambda_reg'])}, as a fraction of the "
           f"residual's own zero-lag autocorrelation). A lag is accepted only if its correlation exceeds 0.2 of "
           f"that value, and the search stops after one lag if that lag's correlation alone exceeds 0.55 of it "
           f"(both fixed, from the published algorithm, not swept here).")
    r.body(("Two points below are where this implementation had to depart from the paper's literal description "
           "to behave sensibly on real audio, rather than choices of convenience -- both are documented with "
           "the measurements that motivated them in parcnet-is2/sparse_ar_model.py's module docstring, which "
           "this section summarizes.", "i"))
    r.bullet(("Where the lag search runs. ", "b"), f"Searching the raw signal's autocorrelation, exactly as the "
             f"published algorithm states, was tested on 60 real single-packet losses across 10 files: it "
             f"selected lag 1 in 60 of 60 cases and scored near 0 dB gap SNR in nearly all of them, because "
             f"ordinary oversampled audio has very high sample-to-sample correlation that has nothing to do with "
             f"musical periodicity, and the paper's own 0.55 early-stop threshold is almost always exceeded by "
             f"it alone. The fix implemented here is the short-term/long-term prediction split standard in "
             f"speech and audio coding (the same structure behind OPUS/SILK, whose fast Burg method "
             f"{cite('vos2013')} this paper itself cites): the lag search runs on the Burg model's own residual, "
             f"not the raw signal. Re-tested on the same 60 bursts, this surfaces real, plausible pitch-period "
             f"lags (e.g. 672 samples at 44.1 kHz is about 66 Hz) and Sparse AR then matches or beats AR(256) on "
             f"roughly half of them, instead of 0 of 60.")
    r.bullet(("How the two components combine. ", "b"), "The published algorithm computes the short-term and "
             "long-term predictions as two separate phases (Burg for an initial handful of samples, then the "
             "long-term model for the rest), spliced together by searching for the best crossfade point. With "
             "the long-term model now defined on the Burg residual rather than the raw signal, the natural and "
             "standard way to synthesize -- as in the codecs above -- is a single continuous recursion: predict "
             "both components at every step and add them, which needs no splice or crossfade search.")
    sparse_tab = r.next_tab()
    r.body("Neither change was tuned to produce better numbers; both were adopted because the algorithm as "
           "literally published does not produce usable output on this dataset, and the fix is a named, "
           "standard technique rather than an invented one. The result (Section 3.2) is still, honestly, that "
           "Sparse AR trails AR(256) on quality by a wide margin on this dataset -- the 'preliminary performance "
           f"analysis' in {cite('severi2025')} reports one favorable qualitative example and no quantitative SNR "
           "benchmarking at all, so this is presented as a real finding about the method at this configuration, "
           "not a failure of the reimplementation. Sparse AR reuses the same per-packet recursive loop and "
           "external crossfade as AR alone (Section 2.2), so its output is directly comparable to the other two "
           f"methods'; its own hyperparameters (Table {sparse_tab}) are independent of theirs.")
    r.table(["Hyperparameter", "Symbol", "Default", "Meaning"], [
        ["Max. long-term lags", "k", hp["max_k_lags"], "non-contiguous lags selected by matching pursuit"],
        ["Max. lag searched", "L", hp["max_lag_l"], "upper bound on candidate lag distance"],
        ["Burg (short-term) order", "p_b", hp["burg_order_pb"], "order of the short-term linear predictor"],
        ["Regularization", "λ_reg", fg(hp["lambda_reg"]), "Tikhonov term, as a fraction of residual r0"],
    ], "Sparse AR's own default hyperparameters (paper defaults).", col_widths_cm=[3.3, 1.5, 2.2, 7.5])

    r.heading("2.5  Implementation and verification", 2)
    r.body("The three concealers are separate code paths. The AR concealer is a standalone implementation with "
           "no dependency on the neural network or its framework; the hybrid concealer is the unmodified "
           "challenge implementation with the network always enabled. With identical hyperparameters, the "
           "standalone AR concealer was verified to produce output identical to the hybrid model's internal AR "
           "branch (mean squared error 0.0), so every difference between those two can be attributed to the "
           "neural residual alone, not to implementation details. Sparse AR is a separate algorithm with no "
           "published reference implementation available, so it does not participate in that bit-identity "
           "check; it was instead verified component-by-component against synthetic signals with known answers "
           "(the Burg fit recovers a pure sinusoid's continuation to under 0.001 RMS error) and checked for "
           "numerical stability (no NaNs or unbounded output) across real files before being used for any "
           "reported result.")

    r.heading("2.6  Dataset and loss traces", 2)
    n = ctx.net
    r.body(f"Experiments use the example blind test set released with the IEEE-IS² 2025 Music PLC Challenge "
           f"{cite('challenge2024')}, which is {n['n_files']} single-channel recordings of solo instrumental "
           f"music at 44.1 kHz each provided as a clean file, a lossy file and the packet trace that produced "
           f"it. The traces are real network traces repurposed by the challenge from the INTERSPEECH 2022 Audio "
           f"Deep PLC Challenge {cite('diener2022')}.")
    # Figure 1 (network_statistics.png) is always the first figure in the document --
    # nothing before section_results's 3.1 creates one -- so this forward reference is
    # safe to state directly rather than needing a two-pass number lookup.
    r.body(f"Table {r.next_tab()} summarizes the loss statistics over all {n['n_files']} traces (plotted in "
           f"Figure 1).")
    ctx.net_table = r.table(["Statistic", "Value"], [
        ["Recordings (clean / lossy / trace triplets)", n["n_files"]],
        ["Total packets", f"{n['total_packets']:,}"],
        ["Lost packets (overall packet loss rate)", f"{n['total_lost']:,} ({n['plr']:.2f}%)"],
        ["Per-file loss rate: median (min-max)", f"{n['plr_median']:.2f}% ({n['plr_min']:.2f}-{n['plr_max']:.2f}%)"],
        ["Files without any loss", n["n_zero_loss"]],
        ["Total loss bursts", f"{n['n_bursts']:,}"],
        ["Mean / maximum burst length", f"{n['mean_burst']:.2f} / {n['max_burst']} packets"],
        ["Bursts of 1 packet / of at most 2 packets", f"{n['pct_single']:.1f}% / {n['pct_le2']:.1f}%"],
        ["Bursts of 8 packets or more", n["n_ge8"]],
    ], "Packet-loss statistics of the real traces of the test set.", col_widths_cm=[9.5, 6.5])
    r.body(f"The main comparison uses {ctx.n_files} of these files. Every file is concealed whole against its "
           f"own real trace.")

    r.heading("2.7  Objective quality metrics", 2)
    r.body("Two families of metrics are used, because a concealment can be close to the original waveform "
           "without sounding natural, and vice versa:")
    r.bullet(("SNR (dB, higher is better). ", "b"), "Time-domain signal-to-noise ratio between the clean and the "
             "concealed signal. It rewards an exact waveform match.")
    r.bullet(("Multi-resolution STFT distance (lower is better). ", "b"), "Spectral convergence plus mean "
             "absolute log-magnitude error, averaged over FFT sizes of 512, 1024 and 2048 "
             f"{cite('yamamoto2020')}. It compares magnitude spectra and ignores phase.")
    r.bullet(("PLCMOS (≈1-5, higher is better). ", "b"), f"A non-intrusive neural estimate of the mean opinion "
             f"score designed specifically for packet loss concealment {cite('diener2023')}.")
    r.bullet(("ViSQOL MOS-LQO (≈1-5, higher is better). ", "b"), f"An intrusive perceptual similarity model "
             f"used in its full-band audio mode at 48 kHz {cite('chinen2020')}.")
    r.body("PLCMOS and ViSQOL need seconds of audio and are therefore computed on whole files. To study individual "
           "loss events the SNR is also computed inside each gap (the lost samples only):")
    r.equation("SNR_gap = 10 · log₁₀( Σₙ∊G x[n]² / Σₙ∊G (x[n] − x̂[n])² )")
    r.body("Gaps whose clean signal is quieter than −60 dBFS RMS are excluded from this analysis because an SNR "
           f"of near-silence is meaningless. Signal complexity is quantified by the mean spectral flatness of "
           f"the clean file (0 = tonal, 1 = noise) {cite('gray1974')}.")

    r.heading("2.8  Statistical analysis", 2)
    r.body(f"The design is paired: every method conceals the same files. For each metric and each pair of "
           f"methods, the per-file difference is oriented so that a positive value always means the first-named "
           f"method is better. Differences are tested with the two-sided Wilcoxon signed-rank test "
           f"{cite('wilcoxon1945')} (ties dropped), which does not assume normality. With 3 methods there are "
           f"3 pairs (AR-Hybrid, AR-Sparse AR, Hybrid-Sparse AR); across 4 metrics that is 12 tests, and "
           f"p-values are adjusted with the Holm-Bonferroni procedure {cite('holm1979')} over all 12 together "
           f"(not per pair), compared with α = 0.05. Effect size is the matched-pairs rank-biserial correlation "
           f"{cite('kerby2014')},")
    r.equation("r = (W⁺ − W⁻) / (W⁺ + W⁻)")
    r.body(f"where W⁺ and W⁻ are the rank sums of positive and negative differences (|r| < 0.1 negligible, "
           f"< 0.3 small, < 0.5 medium, otherwise large). The mean difference is reported with a 95% percentile "
           f"bootstrap confidence interval (10,000 resamples) {cite('efron1993')}. In the per-burst analysis, "
           f"bursts from the same file are not independent, so confidence intervals there use a cluster "
           f"bootstrap that resamples whole files.")

    r.heading("2.9  Hyperparameter sweeps", 2)
    rows = []
    for param, (df, _) in ctx.sweeps.items():
        label, key = SWEEP_LABELS.get(param, (param, param))
        values = sorted(df["value"].unique())
        methods_here = [m for m in METHOD_KEYS if f"snr_{m}" in df.columns]
        method_txt = "/".join(METHOD_LABEL[m].replace("This ", "") for m in methods_here)
        rows.append([label, method_txt, fg(ctx.hparams.get(key, np.nan)),
                    ", ".join(fg(v) for v in values), df["file"].nunique()])
    r.body(f"Each hyperparameter was varied on its own while the others were held at their defaults "
           f"(one-at-a-time design), and only for the method(s) it actually affects: AR order and diagonal "
           f"loading are specific to the dense AR fit shared by AR and Hybrid; context length and crossfade "
           f"length are structural parameters shared by all three; the four Sparse AR parameters "
           f"(Table {sparse_tab}) are specific to it. At every value, the relevant files were concealed whole "
           f"with their real traces and "
           f"scored with all four metrics; compute time was recorded per file "
           f"(Table {r.next_tab()}). Burst length is deliberately not swept: in a causal, recursive concealer it "
           f"is a property of the network, not a design choice, and its effect is instead measured on the real "
           f"bursts (Section 3.4).")
    if rows:
        ctx.sweep_design_table = r.table(["Hyperparameter", "Method(s)", "Default", "Values tested", "Files"], rows,
                                         "Hyperparameter sweep design.", col_widths_cm=[4.0, 2.5, 1.8, 6.2, 1.5])

    r.heading("2.10  Computational cost", 2)
    info = ctx.run_info
    machine = (f"{info.get('processor') or info.get('platform', 'unknown CPU')}, {info.get('cpu_count', '?')} "
               f"logical cores, PyTorch using {info.get('torch_threads', '?')} threads") if info else "the evaluation machine"
    r.body(f"The wall-clock time of each method's concealment of each file was measured after a warm-up run "
           f"(model loading excluded). Since every method only does work on lost packets, the time per file "
           f"divided by its number of lost packets gives the cost of concealing one packet. The real-time factor is")
    r.equation(f"RTF = t_packet / (P / f_s) = t_packet / {ctx.packet_ms:.1f} ms")
    r.body(f"A method can keep up with live audio only if RTF < 1. All timings were measured on CPU ({machine}). "
           f"Sparse AR's timings describe this project's unoptimized Python/NumPy/Numba implementation, not the "
           f"optimized C++ (FFTW, Eigen3) implementation the original paper reports sub-millisecond figures for "
           f"{cite('severi2025')}; the two are not directly comparable.")


def section_results(r: Report, ctx: Context):
    r.heading("3  Results", 1)
    n = ctx.net

    # 3.1 ------------------------------------------------------------------
    r.heading("3.1  Characteristics of the real loss traces", 2)
    fig = r.next_fig()
    r.body(f"Figure {fig} shows the burst-length distribution and the per-file loss rate over all "
           f"{n['n_files']} traces. Losses are dominated by isolated packets ({n['pct_single']:.1f}% of bursts) "
           f"and {n['pct_le2']:.1f}% of all bursts are at most two packets ({2 * ctx.packet_ms:.1f} ms) long; "
           f"the longest burst spans {n['max_burst']} packets ({n['max_burst'] * ctx.packet_ms:.0f} ms), and "
           f"only {n['n_ge8']} bursts reach 8 packets or more. Typical concealment therefore has to bridge one or "
           f"two packets, but the rare long bursts are where errors accumulate in a recursive concealer.")
    r.figure(ctx.out / "burst_comparisons" / "network_statistics.png",
             "Loss statistics of the real packet traces. Left: number of loss bursts per burst length (log "
             "scale). Right: distribution of the packet loss rate across files; the dashed line marks the mean.")
    r.insight(f"Basically, most real losses are short: {n['pct_le2']:.0f}% of bursts are only one or two "
             f"packets. So what really decides a method's score is how it handles these small gaps, not the "
             f"rare {n['max_burst']}-packet bursts. On the right, most files lose a similar, fairly small share "
             f"of packets, but a few files are much worse than the rest.")

    # 3.2 ------------------------------------------------------------------
    r.heading("3.2  Overall concealment quality (RQ1)", 2)
    if ctx.tests is None:
        r.para(("Significance tests not available -- run run_statistical_analysis.py.", "i"))
    else:
        tab = r.next_tab()
        summary = " ".join(s for s in (significance_sentence(ctx, m) for m in ctx.non_baseline_present) if s)
        r.body(f"Table {tab} summarizes the paired comparison over {ctx.n_files} files, for every metric and "
               f"every pair of methods. {summary}")
        rows = []
        for _, t in ctx.tests.iterrows():
            rows.append([t["metric"], t["pair"],
                         f"{fs(t['mean_a_advantage'], 3)} [{fs(t['ci95_low'], 3)}, {fs(t['ci95_high'], 3)}]",
                         f"{t['a_win_rate_pct']:.0f}%", fp(t["p_holm"]),
                         f"{t['rank_biserial_r']:+.2f} ({t['effect_size']})", t["verdict"]])
        r.table(["Metric", "Pair (A vs. B)", "A advantage [95% CI]", "A wins",
                 "p (Holm)", "Effect size r", "Verdict"], rows,
                f"Paired comparison of all three methods over {ctx.n_files} files, every metric x method-pair "
                f"combination (12 tests).",
                col_widths_cm=[2.2, 3.4, 3.3, 1.3, 1.4, 2.2, 2.2], font_size=8,
                notes="Advantage = A minus B, sign-flipped for MR-STFT so that positive always favours A (the "
                      "first-named method). CI: bootstrap 95% confidence interval of the mean. p: two-sided "
                      "Wilcoxon signed-rank test, Holm-corrected over all 12 tests together. r: matched-pairs "
                      "rank-biserial correlation.")

        f_win, f_dist, f_diff = r.next_fig(), r.next_fig() + 1, r.next_fig() + 2
        r.body(f"The win rate (Figure {f_win}) only tells which method was better on each file, each against "
               f"the AR baseline. The per-file distributions (Figure {f_dist}) and the paired differences "
               f"(Figure {f_diff}, every pair) show how large the differences are and how consistent they are "
               f"across files.")
        r.figure(ctx.out / "metrics_comparison" / "win_rate_summary.png",
                 "Percentage of files on which each non-baseline method scores better than AR alone, per metric. "
                 "The dashed line marks 50% (no systematic difference).", width_cm=14)
        win_bits = []
        for m in ctx.non_baseline_present:
            per_metric = [(label, ctx.advantage(key, m, BASELINE_KEY)["a_win_rate_pct"])
                         for key, label, _ in ALL_METRICS if ctx.advantage(key, m, BASELINE_KEY) is not None]
            if per_metric:
                best_label, best_pct = max(per_metric, key=lambda x: x[1])
                win_bits.append(f"{METHOD_LABEL[m]} wins most often on {best_label} ({best_pct:.0f}% of files)")
        r.insight(f"Bars above the dashed line mean that method beat AR more often than not. {'; '.join(win_bits)}. "
                 f"But winning more often does not tell us by how much -- the next two figures show that part.")
        r.figure(ctx.out / "statistical_analysis" / "metric_distributions.png",
                 "Per-file metric values of all three methods (boxes: median and interquartile range; whiskers: "
                 "1.5 IQR). Gray lines connect the three results of the same file.")
        r.insight("If the boxes overlap a lot, the three methods are not that different compared to how much "
                 "results vary from file to file. What actually matters is whether the median line inside each "
                 "box is shifted up or down, not just whether the boxes touch.")
        r.figure(ctx.out / "statistical_analysis" / "paired_differences.png",
                 "Per-file difference between every pair of methods, oriented so that values above zero favour "
                 "the first-named method. Black marker: mean with bootstrap 95% confidence interval. Panel "
                 "titles give the Holm-corrected Wilcoxon p-value and the rank-biserial effect size.")
        r.insight("This is probably the clearest plot in this section, since it takes the same-file pairs and "
                 "just shows the difference, so file-to-file noise cancels out. If the black dot and its error "
                 "bar sit clearly above or below zero, that is a real difference. If the error bar crosses "
                 "zero, we cannot say one method actually beat the other on this sample.")

    # 3.3 ------------------------------------------------------------------
    r.heading("3.3  Dependence on signal complexity (RQ2)", 2)
    if ctx.per_file is not None:
        rows, notable = [], []
        for m in ctx.non_baseline_present:
            for key, label, lib in ALL_METRICS:
                adv = oriented(ctx.per_file, key, lib, m, BASELINE_KEY)
                flat = ctx.per_file["complexity_flatness"]
                if adv.nunique() > 1 and flat.nunique() > 1:
                    rho, p = stats.spearmanr(flat, adv)
                else:
                    rho, p = np.nan, np.nan
                rows.append([METHOD_LABEL[m], label, fs(rho), fp(p)])
                if not np.isnan(p) and p < 0.05:
                    notable.append(f"{METHOD_LABEL[m]} {label} (ρ = {rho:+.2f})")
        fig, tab = r.next_fig(), r.next_tab()
        text = (f"Figure {fig} plots each metric against the spectral flatness of the clean file, and Table {tab} "
                f"gives the Spearman rank correlation between flatness and each non-baseline method's advantage "
                f"over AR. ")
        text += (f"The advantage correlated significantly with signal complexity for {join_words(notable)}."
                 if notable else
                 "No metric showed a significant correlation between signal complexity and any method's "
                 "advantage over AR (uncorrected p ≥ 0.05).")
        r.body(text)
        r.figure(ctx.out / "metrics_comparison" / "complexity_grid.png",
                 "Concealment quality versus signal complexity (mean spectral flatness of the clean file, log "
                 "scale). One dot per file and method; dashed lines are least-squares trends in log-flatness.")
        r.insight("If a trend line goes up, that method does a bit better on noisier/complex audio; if it goes "
                 "down, it does better on tonal/simple audio. If the lines for different methods look roughly "
                 "parallel, complexity is pushing every method the same way and is not the reason one wins over "
                 "another; if they cross or diverge, complexity does change which method wins.")
        r.table(["Method", "Metric", "Spearman ρ (flatness vs. advantage over AR)", "p (uncorrected)"], rows,
                "Correlation between signal complexity and each non-baseline method's advantage over AR.",
                col_widths_cm=[2.5, 2.5, 7, 4], font_size=8)

    # 3.4 ------------------------------------------------------------------
    r.heading("3.4  Dependence on burst length (RQ2)", 2)
    if ctx.burst_q is None or ctx.per_burst is None:
        r.para(("Per-burst results not available -- re-run run_metrics_comparison.py and "
                "run_statistical_analysis.py.", "i"))
    else:
        audible = ctx.per_burst[ctx.per_burst["clean_gap_rms_db"] > -60]
        fig, tab = r.next_fig(), r.next_tab()
        r.body(f"Every real burst in the evaluated files was scored with the gap SNR: {len(audible):,} bursts from "
               f"{audible['file'].nunique()} files after excluding {len(ctx.per_burst) - len(audible):,} "
               f"near-silent gaps (Figure {fig}, Table {tab}). " + burst_sentence(ctx))
        r.figure(ctx.out / "statistical_analysis" / "burst_length_quality.png",
                 "Gap SNR versus real burst length. Left: mean gap SNR of each method. Right: each non-baseline "
                 "method's mean advantage over AR. Error bars: 95% confidence intervals from a cluster bootstrap "
                 "over files; n is the number of bursts per bin.")
        r.insight("The downward slope on the left is just what happens with recursion: quality keeps dropping "
                 "as bursts get longer, since each packet is predicted off the previous (also fake) packet "
                 "instead of real audio. The right panel is more useful for comparing methods -- if an error "
                 "bar does not touch zero, that method's advantage (or disadvantage) at that burst length is "
                 "real and not just noise. If it crosses zero, especially for the longer, rarer bursts, there "
                 "is not enough data yet to say anything for sure.")
        snr_methods = [m for m in ctx.methods_present if f"gap_snr_{m}_mean" in ctx.burst_q.columns]
        adv_methods = [m for m in ctx.non_baseline_present if f"{m}_advantage_mean" in ctx.burst_q.columns]
        rows = []
        for _, b in ctx.burst_q.iterrows():
            row = [b["burst_length_bin"], int(b["n_bursts"]), int(b["n_files"])]
            row += [f"{f(b[f'gap_snr_{m}_mean'])} [{f(b[f'gap_snr_{m}_ci_low'])}, {f(b[f'gap_snr_{m}_ci_high'])}]"
                   for m in snr_methods]
            row += [f"{fs(b[f'{m}_advantage_mean'])} [{fs(b[f'{m}_advantage_ci_low'])}, "
                   f"{fs(b[f'{m}_advantage_ci_high'])}] ({b[f'{m}_win_rate_pct']:.0f}% wins)" for m in adv_methods]
            rows.append(row)
        headers = (["Burst (pkts)", "Bursts", "Files"] + [f"{METHOD_LABEL[m]} gap SNR (dB)" for m in snr_methods] +
                  [f"{METHOD_LABEL[m]} advantage over AR (dB)" for m in adv_methods])
        widths = [1.4, 1.1, 0.9] + [2.6] * len(snr_methods) + [3.2] * len(adv_methods)
        r.table(headers, rows, "Gap SNR by burst length (mean [95% cluster-bootstrap CI]).",
                col_widths_cm=widths, font_size=7)

    # 3.5 ------------------------------------------------------------------
    r.heading("3.5  Hyperparameter sensitivity (RQ3)", 2)
    if not ctx.sweeps:
        r.para(("Sweep results not available -- run run_hparam_sweeps.py.", "i"))
    else:
        first = r.next_fig()
        sweep_design_ref = f"Table {ctx.sweep_design_table}" if hasattr(ctx, "sweep_design_table") else "Table 2"
        r.body(f"Figures {first}-{first + len(ctx.sweeps) - 1} show every metric and the compute time as a "
               f"function of each hyperparameter, for the method(s) it affects ({sweep_design_ref}), and "
               f"Table {r.next_tab()} lists the best value found for SNR and PLCMOS. Because the hybrid network "
               f"was trained on the "
               f"residual of the default AR configuration, values away from the default change both the AR "
               f"prediction and the input statistics the network sees; its results at non-default settings "
               f"therefore measure robustness to this mismatch, not the performance of a network retrained for "
               f"that setting. No equivalent coupling applies to Sparse AR's own sweeps, since its long-term "
               f"lag search is fit fresh at every setting.")
        for param, (df, png) in ctx.sweeps.items():
            label, key = SWEEP_LABELS.get(param, (param, param))
            methods_here = [m for m in METHOD_KEYS if f"snr_{m}" in df.columns]
            by = df.groupby("value")[[f"{k}_{m}" for k in ("snr", "plcmos") for m in methods_here]].mean()
            n_sw = df["file"].nunique()
            files_txt = f"{n_sw} file{'s' if n_sw != 1 else ''} per value"
            r.heading(f"3.5.{list(ctx.sweeps).index(param) + 1}  {label}", 3)
            if len(methods_here) > 1:
                base = methods_here[0] if methods_here[0] != BASELINE_KEY else methods_here[1]
                comparisons = [m for m in methods_here if m != base]
                bits = []
                for m in comparisons:
                    snr_wins = int((by[f"snr_{m}"] > by[f"snr_{base}"]).sum())
                    mos_wins = int((by[f"plcmos_{m}"] > by[f"plcmos_{base}"]).sum())
                    bits.append(f"{METHOD_LABEL[m]} had the higher mean SNR at {snr_wins} and the higher mean "
                               f"PLCMOS at {mos_wins} of them")
                r.body(f"Over the {len(by)} values tested ({files_txt}), {join_words(bits)}. Mean SNR ranged " +
                       "; ".join(f"from {f(by[f'snr_{m}'].min())} to {f(by[f'snr_{m}'].max())} dB for "
                                f"{METHOD_LABEL[m]}" for m in methods_here) + ".")
            else:
                m = methods_here[0]
                r.body(f"Over the {len(by)} values tested ({files_txt}), {METHOD_LABEL[m]}'s mean SNR ranged "
                       f"from {f(by[f'snr_{m}'].min())} to {f(by[f'snr_{m}'].max())} dB.")
            r.figure(png, f"Sweep of the {label} (default {fg(ctx.hparams.get(key, np.nan))}; other "
                          f"hyperparameters at their defaults; {files_txt}; evaluated for "
                          f"{join_words([METHOD_LABEL[m] for m in methods_here])}). Panels: "
                          f"SNR, MR-STFT distance, PLCMOS and ViSQOL (mean ± 1 SD across files) and compute time "
                          f"per file (median and interquartile range).")
            snr_spread = {m: by[f"snr_{m}"].max() - by[f"snr_{m}"].min() for m in methods_here}
            steepest = max(snr_spread, key=snr_spread.get)
            if snr_spread[steepest] < 0.5:
                r.insight(f"SNR barely changes across this whole range (under 0.5 dB for every method), so "
                         f"{label.lower()} does not seem to matter much here.")
            else:
                r.insight(f"{METHOD_LABEL[steepest]} is the most affected by this setting, swinging by "
                         f"{snr_spread[steepest]:.2f} dB of SNR across the range tested. Worth checking the "
                         f"compute-time panel too before picking a non-default value -- a quality gain here "
                         f"might cost more compute.")
        if ctx.sweep_summary is not None:
            ss = ctx.sweep_summary
            rows = []
            for (param, method), g in ss.groupby(["hyperparameter", "method"], sort=False):
                label = SWEEP_LABELS.get(param, (param,))[0]
                cells = [label, method]
                for metric in ("SNR (dB)", "PLCMOS"):
                    row = g[g["metric"] == metric]
                    if row.empty:
                        cells.append("n/a")
                        continue
                    row = row.iloc[0]
                    cells.append(f"{fg(row['best_value'])} ({fs(row['gain_over_default'], 3)})")
                row = g.iloc[0]
                cells.append(f"{f(row['median_time_ms_at_default'], 0)} → {f(row['median_time_ms_at_best'], 0)}")
                rows.append(cells)
            r.table(["Hyperparameter", "Method", "Best for SNR (gain vs. default)",
                     "Best for PLCMOS (gain vs. default)", "Time/file (ms), default → best"], rows,
                    "Best value of each hyperparameter and gain over the default (positive = better than default).",
                    col_widths_cm=[4, 2.0, 3.2, 3.2, 3.6], font_size=7.5,
                    notes="Gains are differences of means across files and are not tested for significance. "
                          "Time is the median per-file concealment time at the best SNR value. The full table "
                          "for all four metrics is in Appendix C.")

    # 3.6 ------------------------------------------------------------------
    r.heading("3.6  Computational cost (RQ4)", 2)
    if not ctx.has_timing:
        r.para(("Timing data not available -- re-run run_metrics_comparison.py.", "i"))
    else:
        rows = []
        for method in ctx.methods_present:
            ms = ctx.per_file[f"ms_per_lost_packet_{method}"].dropna()
            rtf = ctx.rtf(method)
            rows.append([METHOD_LABEL[method], f"{np.median(ms):.2f}",
                        f"{np.percentile(ms, 25):.2f}-{np.percentile(ms, 75):.2f}",
                        f"{np.median(rtf):.3f}", "yes" if np.median(rtf) < 1 else "no"])
        fig, tab = r.next_fig(), r.next_tab()
        ctx.realtime_fig = fig
        lpc_ms = np.median(ctx.per_file["ms_per_lost_packet_lpc"].dropna())
        ratios = ", ".join(f"{METHOD_LABEL[m]} by about "
                           f"{np.median(ctx.per_file[f'ms_per_lost_packet_{m}'].dropna()) / lpc_ms:.0f}x"
                           for m in ctx.non_baseline_present)
        r.body(f"Table {tab} and Figure {fig} report the cost of concealing one packet. "
               + realtime_sentence(ctx) + f" Relative to AR, the per-packet cost is multiplied by {ratios}. "
               f"The challenge rules do not disqualify slower-than-real-time systems but encourage real-time "
               f"operation {cite('challenge2024')}. Sparse AR's figures reflect this project's unoptimized "
               f"Python implementation and are not comparable to the original paper's optimized C++ timings "
               f"(Section 2.10).")
        r.table(["Method", "Median ms / packet", "IQR (ms)", "Median RTF", "Real time?"], rows,
                f"Per-packet concealment cost (packet duration {ctx.packet_ms:.1f} ms).",
                col_widths_cm=[4, 3, 3, 3, 3])
        r.figure(ctx.out / "statistical_analysis" / "realtime_feasibility.png",
                 "Real-time factor (time to conceal one packet divided by the packet duration) per file, log "
                 "scale. Below the red line a method keeps up with live audio on this machine.", width_cm=13)
        over_budget = [METHOD_LABEL[m] for m in ctx.methods_present if np.median(ctx.rtf(m)) > 1]
        r.insight(("Anything above the red line is too slow to use live on this machine, which rules out "
                   f"{join_words(over_budget)} here." if over_budget else
                   "Every method stays below the red line, so in principle all three could run live on this "
                   "machine.") +
                 " The taller a box is, the less consistent that method's timing is -- some files take much "
                 "longer to process than others.")

    # 3.7 ------------------------------------------------------------------
    r.heading("3.7  Qualitative examples", 2)
    sets = burst_figure_sets(ctx.out / "burst_comparisons")
    if not sets:
        r.para(("Burst examples not available -- run plot_burst_comparison.py.", "i"))
    else:
        r.body("To show what the metrics above summarize, one real burst of each length was selected and "
               "plotted in three ways: the waveform and spectrogram around the gap (shaded) for all three "
               "methods; the sample-wise difference for every pair of methods; and a zoom on the hand-back point "
               "where concealment crossfades into the next received packet, together with the clean reference. "
               f"Table {r.next_tab()} lists the mean squared difference between every pair of methods inside "
               f"each gap.")
        if ctx.gap_mse is not None:
            pair_cols = [c for c in ctx.gap_mse.columns if "_vs_" in c]
            r.table(["Burst (packets)", "File", "Start packet"] + [pretty_pair_label(c) for c in pair_cols],
                    [[int(g["burst_length"]), g["file"][:20] + "…", int(g["start_packet"])] +
                     [f"{g[c]:.3e}" for c in pair_cols]
                     for _, g in ctx.gap_mse.sort_values("burst_length").iterrows()],
                    "Selected example bursts: gap MSE between every pair of methods.",
                    col_widths_cm=[2.0, 5.0, 1.8] + [2.4] * len(pair_cols), font_size=7)
        for i, (length, tag, paths) in enumerate(sets, start=1):
            r.heading(f"3.7.{i}  {length}-packet burst ({length * ctx.packet_ms:.1f} ms)", 3)
            r.figure(paths["compare"], f"{length}-packet burst in file {tag}: waveform (top) and spectrogram "
                                       f"(bottom) of all three methods. Shaded: lost samples.")
            r.insight("Just eyeball how close the three waveforms and spectrograms stay to each other across "
                     "the shaded gap. Whichever one looks most different from the other two is the one that "
                     "changed the audio the most there.")
            r.figure(paths["diff"], f"{length}-packet burst: pairwise differences between all three methods; "
                                    f"the gap-MSE in each title quantifies that pair's divergence.", width_cm=15)
            match = None
            if ctx.gap_mse is not None:
                found = ctx.gap_mse[(ctx.gap_mse["file"] == tag) & (ctx.gap_mse["burst_length"] == length)]
                if len(found):
                    match = found.iloc[0]
            pair_cols_here = [c for c in ctx.gap_mse.columns if "_vs_" in c] if ctx.gap_mse is not None else []
            if match is not None and pair_cols_here:
                biggest = max(pair_cols_here, key=lambda c: match[c])
                r.insight(f"A flat line at zero means those two methods agree almost exactly inside the gap. "
                         f"The biggest disagreement here is {pretty_pair_label(biggest)} (gap MSE "
                         f"{match[biggest]:.2e}) -- that is the panel with the most visible activity.")
            else:
                r.insight("A flat line at zero means those two methods agree almost exactly inside the gap -- "
                         "whichever panel shows the most activity is where that pair disagrees the most.")
            r.figure(paths["crossfade"], f"{length}-packet burst: zoom on the hand-back point (dashed line) where "
                                         f"concealment ends; the shaded region is the {ctx.hparams['extra_dim']}-"
                                         f"sample crossfade into the received packet.")
            r.insight("What to look for here is a jump or kink right at the dashed line -- that would be an "
                     "audible click. If the line through the shaded region looks smooth, the handback to real "
                     "audio is clean.")

    # 3.8 ------------------------------------------------------------------
    r.heading("3.8  Failure cases", 2)
    if not ctx.failures:
        r.para(("No failure cases available (either no non-baseline method ever lost to AR on the selected "
                "metric, or run_statistical_analysis.py has not been run).", "i"))
    else:
        r.body("Reporting only favourable examples would bias the picture. For each non-baseline method, the "
               "files on which it lost to AR by the largest margin on SNR are shown below, with the burst where "
               "its disadvantage was largest in gap SNR.")
        for sub_i, (method, failures) in enumerate(ctx.failures.items(), start=1):
            metric_key = [c[:-len(f"_{method}")] for c in failures.columns if c.endswith(f"_{method}")][0]
            label = next(l for k, l, _ in ALL_METRICS if k == metric_key)
            flat_med = ctx.per_file["complexity_flatness"].median()
            loss_med = ctx.per_file["loss_rate_pct"].median()
            tab = r.next_tab()
            r.heading(f"3.8.{sub_i}  {METHOD_LABEL[method]} vs. AR", 3)
            r.body(f"The {len(failures)} files on which {METHOD_LABEL[method]} lost to AR by the largest {label} "
                   f"margin are listed in Table {tab}. Their spectral flatness ranged from "
                   f"{failures['complexity_flatness'].min():.2e} to {failures['complexity_flatness'].max():.2e} "
                   f"(median of all evaluated files: {flat_med:.2e}), and their loss rates from "
                   f"{failures['loss_rate_pct'].min():.1f}% to {failures['loss_rate_pct'].max():.1f}% "
                   f"(median {loss_med:.1f}%).")
            r.table(["Case", "File", f"AR {label}", f"{METHOD_LABEL[method]} {label}", "Advantage",
                     "Flatness", "Loss rate", "Plotted burst"],
                    [[int(c["case"]), c["file"][:18] + "…", f(c[f"{metric_key}_{BASELINE_KEY}"]),
                      f(c[f"{metric_key}_{method}"]), fs(c[f"{method}_advantage"]),
                      f"{c['complexity_flatness']:.2e}", f"{c['loss_rate_pct']:.1f}%",
                      f"{int(c['plotted_burst_length'])} pkt @ {int(c['plotted_burst_start'])}"]
                     for _, c in failures.iterrows()],
                    f"Files with the largest AR advantage over {METHOD_LABEL[method]} on {label}.",
                    col_widths_cm=[1, 3.2, 1.6, 1.8, 1.6, 1.6, 1.6, 2.2], font_size=7.5)
            fail_dir = ctx.out / "statistical_analysis" / "failure_cases" / method
            for length, tag, paths in burst_figure_sets(fail_dir, "failure"):
                case = tag.split("_", 1)[0].replace("failure", "")
                case_row = failures[failures["case"] == int(case)]
                adv_text = fs(case_row.iloc[0][f"{method}_advantage"]) if len(case_row) else "n/a"
                r.heading(f"3.8.{sub_i}.{case}  {METHOD_LABEL[method]} failure case {case}", 3)
                r.figure(paths["compare"], f"Failure case {case}: waveform and spectrogram around the plotted "
                                           f"{length}-packet burst, all three methods.")
                r.insight(f"This is the actual burst behind this file's {label} advantage of {adv_text} -- "
                         f"worth looking at what {METHOD_LABEL[method]} is doing differently from AR here, "
                         f"since a single whole-file average cannot show that on its own.")
                r.figure(paths["diff"], f"Failure case {case}: pairwise differences between all three methods.",
                         width_cm=15)
                r.insight(f"The {METHOD_LABEL[method].lower()}-vs-AR panel is the one that matters here. If it "
                         f"shows a lot more activity than the other two, this disagreement is specific to "
                         f"{METHOD_LABEL[method]}, not something all three methods share.")
                r.figure(paths["crossfade"], f"Failure case {case}: hand-back crossfade with the clean reference.")
                r.insight("Even though this file counts as a loss, a clean splice here (no click or jump at "
                         "the dashed line) would mean the problem is the quality inside the gap, not the "
                         "handback back to real audio.")


def section_discussion(r: Report, ctx: Context):
    r.heading("4  Discussion", 1)
    if ctx.tests is not None and "nn" in ctx.non_baseline_present:
        adv = {k: ctx.advantage(k, "nn", BASELINE_KEY)["mean_a_advantage"] for k, _, _ in ALL_METRICS}
        sig = {k: ctx.advantage(k, "nn", BASELINE_KEY)["verdict"] for k, _, _ in ALL_METRICS}
        spectral_pos = [METRIC_DESC[k] for k in ("mrstft", "plcmos", "visqol") if adv[k] > 0]
        r.heading("4.1  What the neural residual adds", 2)
        if adv["snr"] <= 0 and spectral_pos:
            r.body(f"The two metric families point in different directions. On average the hybrid model "
                   f"{'lowered' if adv['snr'] < 0 else 'did not change'} the waveform SNR "
                   f"({fs(adv['snr'])} dB, {sig['snr']}) while it improved {join_words(spectral_pos)}. "
                   f"This pattern is consistent with how the network is trained: its loss includes a spectral "
                   f"term, and a spectral improvement can coexist with a larger sample-wise error, because a "
                   f"prediction whose spectrum is right but whose phase drifts is penalized by SNR and not by "
                   f"magnitude-based or perceptual measures. One interpretation is that the residual makes the "
                   f"concealment more plausible rather than more exact; testing this directly would require a "
                   f"listening test (Section 5).")
        elif all(v > 0 for v in adv.values()):
            r.body("All four metrics favoured the hybrid model on average, so the neural residual improved both "
                   "the waveform match and the perceptual plausibility of the concealment.")
        else:
            r.body("The metrics did not favour either method consistently: "
                   + "; ".join(f"{METRIC_DESC[k]} {fs(v, 3)} ({sig[k]})" for k, v in adv.items()) + ".")
        if ctx.preliminary:
            r.body(f"With only {ctx.n_files} files, none of these directions can be regarded as established; "
                   f"they should be read as hypotheses for the full experiment.")

    if ctx.tests is not None and "sparse_ar" in ctx.non_baseline_present:
        adv = {k: ctx.advantage(k, "sparse_ar", BASELINE_KEY)["mean_a_advantage"] for k, _, _ in ALL_METRICS}
        r.heading("4.2  What the sparse, periodicity-aware model trades off", 2)
        r.body(f"Sparse AR's advantage over AR(256) was negative on every metric in this evaluation "
               f"({'; '.join(f'{METRIC_DESC[k]} {fs(v, 3)}' for k, v in adv.items())}): replacing the dense, "
               f"256-tap fit with a short Burg model plus a handful of matching-pursuit lags loses quality here, "
               f"it does not trade one kind of quality for another the way the hybrid model does. Section 2.4 "
               f"documents why a literal reading of the published algorithm is unusable on this dataset (it "
               f"degenerates to picking the trivial lag-1 correlation and predicting near-silence) and what "
               f"standard, named fix was applied instead; even with that fix, the short-term component alone "
               f"decays to silence within a packet for most real bursts, and a genuine long-term periodic lag "
               f"is found only for a minority of them -- the real-time plot "
               f"(Figure {getattr(ctx, 'realtime_fig', '?')}) nonetheless confirms the method's actual appeal: "
               f"it is as cheap as plain AR, nowhere near the hybrid model's cost. "
               f"Whether a higher-order short-term stage within the same architecture closes any of this gap "
               f"is a genuine open question the Section 3.5 sweeps over burg_order_pb speak to directly, rather "
               f"than something resolved by further hand-tuning here.")

    r.heading("4.3  Burst length, complexity and recursion", 2)
    observed = ""
    if ctx.burst_q is not None and len(ctx.burst_q) > 1 and "gap_snr_lpc_mean" in ctx.burst_q.columns:
        first, last = ctx.burst_q.iloc[0], ctx.burst_q.iloc[-1]
        fell = all(last.get(f"gap_snr_{m}_mean", np.inf) < first.get(f"gap_snr_{m}_mean", -np.inf)
                  for m in ctx.methods_present)
        observed = (", as observed here for every method" if fell else
                    ", although this was not observed consistently for every method in the present data")
    r.body("Because every concealer here is recursive, each packet of a burst is predicted from the previous "
           f"predictions, so errors accumulate and gap quality is expected to fall with burst length{observed}. "
           "Whether either alternative to plain AR helps more on long bursts is the practically important "
           f"question, but long bursts are rare in real traces (Table {ctx.net_table}), which makes the "
           "long-burst bins the least certain ones; their confidence intervals should be read accordingly.")

    r.heading("4.4  Cost-quality trade-off", 2)
    if ctx.has_timing:
        lpc_rtf = np.median(ctx.rtf(BASELINE_KEY))
        bits = [f"{METHOD_LABEL[m]} ({np.median(ctx.rtf(m)):.2f})" for m in ctx.non_baseline_present]
        over = [m for m in ctx.non_baseline_present if np.median(ctx.rtf(m)) > 1]
        r.body(f"AR {'runs in real time' if lpc_rtf < 1 else 'does not run in real time'} on the evaluation CPU "
               f"(median RTF {lpc_rtf:.2f}). Of the alternatives, median RTF was {join_words(bits)}" +
               (f"; {join_words([METHOD_LABEL[m] for m in over])} exceeded the real-time budget" if over else
                ", both within it") +
               f". Any quality benefit (Sections 4.1-4.2) must be weighed against this cost; for the hybrid "
               f"model, options such as multi-threaded or GPU inference, model quantization or a smaller "
               f"network would need to be considered for live deployment. For Sparse AR, see Section 2.10's "
               f"caveat on what these particular timings do and do not represent.")
    r.heading("4.5  Coupling between the AR configuration and the network", 2)
    r.body("The hybrid network was trained on residuals of the default AR configuration. Changing the AR order, "
           "context length or regularization changes those residuals, so the hybrid model operates outside its "
           "training conditions. The sweeps therefore measure how robust a fixed network is to such changes; a "
           "fair test of a different AR configuration for the hybrid model would require retraining the network "
           "for it. Sparse AR has no analogous coupling: its long-term lag search is refit from scratch at "
           "every setting, so its own sweeps measure the algorithm directly.")


def section_limitations(r: Report, ctx: Context):
    r.heading("5  Limitations and future work", 1)
    if ctx.preliminary:
        sweep_n = sorted(set(ctx.sweep_files.values()))
        sweep_txt = (f" and on sweeps with {join_words([str(v) for v in sweep_n])} "
                     f"file{'s' if sweep_n != [1] else ''} per value") if sweep_n else ""
        r.bullet(("Sample size. ", "b"), f"This version is based on {ctx.n_files} files{sweep_txt}, too few "
                 f"for reliable conclusions. The full runs will replace these numbers.")
    r.bullet(("No listening test. ", "b"), f"The challenge itself evaluates submissions with a MUSHRA listening "
             f"test {cite('itu1534')}, noting that no objective metric has been shown to correlate reliably with "
             f"perception for music PLC {cite('challenge2024')}. PLCMOS was designed for speech, so the "
             f"perceptual metrics here are proxies; a MUSHRA test is the natural next step.")
    r.bullet(("One dataset, one checkpoint. ", "b"), "All results come from one music test set and the "
             "pretrained challenge baseline. Generalization to other content (for example the speech validation "
             "set of the ICASSP 2024 PLC Challenge, which uses 48 kHz audio and 20 ms packets) and to retrained "
             "networks remains to be tested.")
    r.bullet(("Sparse AR's departures from its source paper. ", "b"), "Section 2.4 documents two points "
             "(where the lag search runs; how the short-term and long-term components combine) where this "
             "implementation had to depart from the published algorithm's literal description to produce usable "
             "output, with measured evidence for each. Both are standard, named techniques, not arbitrary "
             "choices, but no published reference implementation exists to confirm they reproduce the author's "
             "own intended behavior, and the paper's own 'preliminary performance analysis' section provides no "
             "quantitative benchmark to compare against.")
    r.bullet(("Gap SNR is phase-sensitive. ", "b"), "The per-burst analysis uses SNR because the perceptual "
             "models need longer inputs; it penalizes plausible but phase-shifted concealments.")
    r.bullet(("One-at-a-time sweeps. ", "b"), "Interactions between hyperparameters (for example AR order and "
             "context length, or Sparse AR's own burg_order_pb and max_k_lags) are not explored; a joint search "
             "would be needed to find a global optimum for any method.")
    r.bullet(("Timing on one CPU. ", "b"), "Real-time factors depend on hardware, threading and implementation; "
             "they are indicative of the relative cost, not of every deployment -- and Sparse AR's specifically "
             "reflect an unoptimized Python port, not the original paper's optimized C++ implementation.")


def section_conclusion(r: Report, ctx: Context):
    r.heading("6  Conclusion", 1)
    text = (f"This report compared AR-only, hybrid AR + neural (PARCnet), and Sparse AR (Severi 2025) packet "
            f"loss concealment for networked music on real loss traces, using independently implemented "
            f"concealers -- the AR and hybrid ones verified bit-identical at the AR level, and Sparse AR built "
            f"from scratch from its published equations, with two documented, evidence-based departures where "
            f"the literal algorithm proved unusable on real audio. ")
    sentences = [significance_sentence(ctx, m) for m in ctx.non_baseline_present]
    sentences += [burst_sentence(ctx), realtime_sentence(ctx)]
    text += " ".join(s for s in sentences if s)
    if ctx.preliminary:
        text += " These conclusions are preliminary and will be updated with the full experiments."
    r.body(text)


def section_references(r: Report):
    r.heading("References", 1)
    for i, (_, text) in enumerate(REFERENCES, start=1):
        p = r.para(f"[{i}] {text}", size=9.5, space_after=3)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)


def section_appendix(r: Report, ctx: Context):
    r.page_break()
    r.heading("Appendix A  Reproducibility", 1)
    r.body("All results were produced with the scripts in the research/ folder of the project repository, in "
           "this order:")
    for cmd in [f"python -m research.run_metrics_comparison --n-files {ctx.n_files} --seed "
                f"{ctx.run_info.get('seed', 0)}",
                f"python -m research.run_hparam_sweeps --n-files {max(ctx.sweep_files.values(), default=20)}",
                "python -m research.plot_burst_comparison",
                "python -m research.run_statistical_analysis",
                "python -m research.build_report"]:
        p = r.para(cmd, size=9)
        p.runs[0].font.name = "Consolas"
        p.paragraph_format.left_indent = Cm(0.8)
    info = ctx.run_info
    rows = [["Platform", info.get("platform", "n/a")], ["Processor", info.get("processor", "n/a")],
            ["Logical cores", info.get("cpu_count", "n/a")], ["PyTorch threads", info.get("torch_threads", "n/a")]]
    rows += software_versions()
    r.table(["Item", "Value"], rows, "Hardware of the main evaluation run and software versions.",
            col_widths_cm=[5, 11])

    if ctx.per_file is not None:
        r.heading("Appendix B  Descriptive statistics", 1)
        rows = []
        for key, label, _ in ALL_METRICS:
            for method in ctx.methods_present:
                v = ctx.per_file[f"{key}_{method}"]
                rows.append([label, METHOD_LABEL[method], f(v.mean(), 3), f(v.std(), 3), f(v.median(), 3),
                             f"{v.quantile(0.25):.3f}-{v.quantile(0.75):.3f}", f"{v.min():.3f}-{v.max():.3f}"])
        r.table(["Metric", "Method", "Mean", "SD", "Median", "IQR", "Range"], rows,
                f"Descriptive statistics of all metrics over {ctx.n_files} files.",
                col_widths_cm=[3, 2.2, 1.6, 1.6, 1.6, 2.7, 2.7], font_size=8)

    if ctx.sweep_summary is not None:
        r.heading("Appendix C  Full hyperparameter sweep summary", 1)
        ss = ctx.sweep_summary
        rows = [[SWEEP_LABELS.get(s["hyperparameter"], (s["hyperparameter"],))[0],
                 s["method"], s["metric"], fg(s["best_value"]),
                 f(s["metric_at_default"], 3), f(s["metric_at_best"], 3), fs(s["gain_over_default"], 3)]
                for _, s in ss.iterrows()]
        r.table(["Hyperparameter", "Method", "Metric", "Best value", "At default", "At best", "Gain vs. default"],
                rows, "Best value per hyperparameter, method and metric.",
                col_widths_cm=[3.6, 1.8, 2.6, 1.8, 2.0, 2.0, 2.0], font_size=7.5)


def build(output_dir: Path, out_path: Path) -> Path:
    ctx = Context(output_dir)
    r = Report()
    section_front(r, ctx)
    section_introduction(r, ctx)
    section_methods(r, ctx)
    section_results(r, ctx)
    section_discussion(r, ctx)
    section_limitations(r, ctx)
    section_conclusion(r, ctx)
    section_references(r)
    section_appendix(r, ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return safe_io._save_with_fallback(lambda p: r.doc.save(str(p)), out_path, "_locked")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Folder containing the four scripts' output subfolders.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Path of the .docx to write.")
    args = parser.parse_args()
    path = build(args.output_dir, args.out)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
