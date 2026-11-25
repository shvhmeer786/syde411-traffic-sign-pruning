"""Compose the team-review report from measured results and the proposal's references."""

import csv
from pathlib import Path
from statistics import mean, stdev

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEST = ROOT / "deliverables"
DEST.mkdir(exist_ok=True)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


rows = read(RESULTS / "test_results.csv")
paired = read(RESULTS / "paired_differences.csv")
sweep = read(RESULTS / "validation_sensitivity.csv")
layers = read(RESULTS / "layer_retention.csv")
audit = read(RESULTS / "training_objective_audit.csv")
sparse_audit = [r for r in audit if r["method"] != "dense"]


def subset(method, s):
    return [r for r in rows if r["method"] == method and int(r["sparsity"]) == s]


def avg(rs, key, scale=1):
    return mean(float(r[key]) * scale for r in rs)


def stats(rs, key, scale=1, digits=2):
    v = [float(r[key]) * scale for r in rs]
    return f"{mean(v):.{digits}f} ± {stdev(v):.{digits}f}"


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.72)
sec.bottom_margin = Inches(0.68)
sec.left_margin = Inches(0.85)
sec.right_margin = Inches(0.85)
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.07
for style, size in (("Title", 23), ("Heading 1", 17), ("Heading 2", 13)):
    st = doc.styles[style]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor.from_string(
        "422333" if style == "Heading 1" else "282725"
    )
    st.paragraph_format.space_before = Pt(5)
    st.paragraph_format.space_after = Pt(5)
    borders = st.element.xpath(".//w:pBdr")
    for b in borders:
        b.getparent().remove(b)
footer = sec.footer.paragraphs[0]
footer.alignment = 2
footer.add_run("SYDE 411   |   Group 28   |   ").font.size = Pt(9)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def h(s):
    doc.add_heading(s, level=1)


def sub(s):
    doc.add_heading(s, level=2)


def p(s):
    return doc.add_paragraph(s)


def eq(s):
    q = p(s)
    q.alignment = 1
    for r in q.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)


def page(s):
    doc.add_page_break()
    h(s)


def table(headers, values):
    t = doc.add_table(rows=0, cols=len(headers))
    t.style = "Normal Table"
    for idx, row in enumerate([headers] + values):
        cells = t.add_row().cells
        for col, (c, value) in enumerate(zip(cells, row)):
            props = c._tc.get_or_add_tcPr()
            if idx == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "F0EDE8")
                props.append(shade)
            edges = OxmlElement("w:tcBorders")
            for edge in ("top", "bottom", "left", "right"):
                rule = OxmlElement("w:" + edge)
                visible = (idx == 0 and edge in ("top", "bottom")) or (
                    idx == len(values) and edge == "bottom"
                )
                rule.set(qn("w:val"), "single" if visible else "nil")
                rule.set(qn("w:sz"), "5")
                rule.set(qn("w:color"), "9F9790")
                edges.append(rule)
            props.append(edges)
            q = c.paragraphs[0]
            q.paragraph_format.space_after = Pt(2)
            q.paragraph_format.space_before = Pt(2)
            r = q.add_run(str(value))
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            r.bold = idx == 0
            q.alignment = 0 if col == 0 or (headers[0] == "Pruned" and col == 1) else 2


def fig(name, width, cap):
    doc.add_picture(str(RESULTS / name), width=Inches(width))
    doc.paragraphs[-1].alignment = 1
    q = p(cap)
    q.runs[0].italic = True
    q.runs[0].font.size = Pt(9)


def cite(text, url):
    q = p("")
    hlink = OxmlElement("w:hyperlink")
    hlink.set(qn("r:id"), doc.part.relate_to(url, RT.HYPERLINK, is_external=True))
    rr = OxmlElement("w:r")
    f = OxmlElement("w:rPr")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    f.append(u)
    rr.append(f)
    t = OxmlElement("w:t")
    t.text = text
    rr.append(t)
    hlink.append(rr)
    q._p.append(hlink)


doc.add_paragraph("Pruning neural networks for traffic sign recognition", style="Title")
p("SYDE 411 Optimization and Numerical Methods  |  Technical report draft  |  Group 28")
p("Shahmeer Ali, Amin Ambike and Brayden Petersen")
h("Research question")
p(
    "A vehicle-mounted camera may need to distinguish visually similar signs with limited onboard storage. GTSRB supplies 43 sign classes [4], and compact networks have been studied for this application [7]. For a fixed classifier, we ask which connections to retain. Magnitude pruning ranks weights once [1]. The alternating direction method of multipliers (ADMM) can revise that selection during learning [2,3]. Does this flexibility reduce classification loss or preserve accuracy when both methods have the same weight and training budgets?"
)
d80 = avg(subset("admm", 80), "accuracy", 100) - avg(
    subset("magnitude", 80), "accuracy", 100
)
dc80 = avg(subset("admm", 80), "ce") - avg(subset("magnitude", 80), "ce")
p(
    "The methods were within 0.05 percentage points of each other in mean test accuracy at 50% sparsity. At 80%, ADMM achieved lower test cross-entropy, while magnitude pruning retained a small accuracy advantage. At 90%, magnitude pruning led in each paired run and averaged 67.37% accuracy versus 56.57% for ADMM. All 18 sparse models satisfied their caps after reloading."
)
sub("Optimization problem")
p(
    "Let w contain the P = 98,528 convolutional and output weights and b the q = 267 biases. For a labeled example (xᵢ, yᵢ), let ℓᵢ denote the negative log probability assigned to its true class. Training uses N = 31,379 images. We optimize 98,795 real variables with one cardinality constraint."
)
from lxml import etree

for formula in etree.parse(str(ROOT / "authoring/formulation.xml")).getroot():
    q = doc.add_paragraph()
    q.alignment = 1
    q._p.append(formula)
p(
    "The network f returns 43 logits. For each image, ℓ is the negative log probability of its correct label. F gives each training image equal weight. The fixed coefficient λ = 0.00001 penalizes weight size. The count ‖w‖₀ excludes biases. We hold the architecture fixed and give each retained weight equal cost. For pruning proportion s, K = floor((1 − s)P) gives 49,264, 19,705 and 9,852 retained weights at 50%, 80% and 90% sparsity. K measures parameter count. Storage and inference time also depend on encoding and hardware."
)
sub("Geometry and possible solutions")
p(
    "With two weights and K = 1, (1,0) and (0,1) are feasible, but their midpoint is infeasible. The full feasible set is a closed, nonconvex union of coordinate subspaces with free biases. The count changes discontinuously when a zero weight becomes nonzero. Even at the tightest cap, selecting exactly 9,852 weights from 98,528 gives over 10¹³⁹⁰⁷ different supports (the binomial coefficient for choosing K of P). The CNN objective is continuous and piecewise differentiable, with kinks at ReLU boundaries. Even on one support the layers interact nonlinearly. Channel permutations can give equal objective values, and different supports can lead to different solutions. The formulation gives no basis for assuming a single basin of attraction."
)

page("Methods and experimental controls")
sub("Two ways to satisfy the same constraint")
p(
    "Magnitude pruning follows the prune-and-retrain baseline [1]. We select the K largest absolute weights in the dense model and keep that support fixed for four epochs. The optimizer updates the retained weights and all biases. Applying the mask after every update keeps all other weights at zero."
)
p(
    "Zhang et al. assign each layer a pruning budget [3]. We use one global budget so the solver can allocate capacity across layers. ADMM introduces a copy z subject to ‖z‖₀ ≤ K and enforces w = z. The learning step can use backpropagation through the network, while the sparse subproblem has an exact Euclidean projection [2,3]. This avoids evaluating the loss for each possible support. The scaled multiplier u carries the mismatch between the two weight vectors into the next iteration."
)
eq("w,b step: minimize approximately   F(w,b) + (ρ/2) ‖w − z + u‖₂²")
eq("z step: z ← Hₖ(w + u)                 multiplier step: u ← u + w − z")
p(
    "Start from a dense checkpoint with z = w and u = 0. In each outer iteration, take 246 minibatch Adam steps on the penalized objective, keep the K largest magnitudes of w + u, then update u with ρ fixed at 0.001. Retaining the largest entries minimizes the sum of squared discarded entries. A stable sort breaks ties by flat index. Stop after three outer iterations to respect the training budget. Apply the last support to the learned w, retain its weights on that support, and retrain for the remaining 246 steps. This yields a feasible output even when w and z disagree at termination."
)
sub("Data and controls")
p(
    "We used the authors’ final 39,209-image training distribution. A track contains frames of the same physical sign. Within each class we assigned complete tracks to training or validation using seed 411. The result was 31,379 training images from 1,046 tracks and 7,830 validation images from 261 tracks, with all 43 classes in both sets and zero shared tracks. We cropped to the annotated sign region, resized to 32 × 32 RGB and scaled pixels to [−1,1]. The official 12,630 test images were reserved for evaluation after settings were frozen [4]."
)
p(
    "Three 3 × 3 convolutions have 32, 64 and 128 channels, with ReLU after each and 2 × 2 average pooling after the first two. Global average pooling feeds a 43-class linear head. We trained dense checkpoints for 12 epochs with batch size 128 and learning rate 0.003. Initialization seeds were 411, 412 and 413. The data split, sample order and optimizer settings stayed fixed. Each pair started from byte-identical weights and received 984 gradient updates, including all retraining, at learning rate 0.001. Matching retraining effort follows the comparisons in [5,6]."
)
p(
    "Accuracy is the fraction of correct predictions. Macro recall averages the 43 class recalls equally. Cross-entropy measures the probability assigned to the true class. Timings use the same Apple M1 Pro GPU via PyTorch MPS, synchronized at epoch boundaries. Pruning times include initialization of the pruning state, projections and training, and exclude validation and file I/O."
)

page("Classification on the official test set")
vals = []
for s in (0, 50, 80, 90):
    for method in ["dense"] if s == 0 else ["magnitude", "admm"]:
        rs = subset(method, s)
        vals.append(
            [
                f"{s}%",
                method.upper() if method == "admm" else method.title(),
                rs[0]["nonzeros"],
                stats(rs, "accuracy", 100),
                stats(rs, "macro_recall", 100),
                stats(rs, "ce", 1, 3),
            ]
        )
table(
    [
        "Pruned",
        "Method",
        "Nonzeros",
        "Accuracy (%)",
        "Macro recall (%)",
        "Cross-entropy",
    ],
    vals,
)
p(
    "Table 1. Test metrics and final weight counts. Values after ± are sample standard deviations across three initializations. The cap was checked on each reloaded classifier, including after the final masked update. Biases are excluded from all counts."
)
fig(
    "test_comparison.png",
    6.65,
    "Figure 1. The points show mean performance on the official test set. Error bars show ±1 sample standard deviation across seeds. The dotted line gives the dense reference mean.",
)
for s in (50, 80, 90):
    ps = [r for r in paired if int(r["sparsity"]) == s]
    p(
        f"At {s}% sparsity, the paired accuracy differences were "
        + ", ".join(f"{float(r['admm_minus_magnitude_pp']):+.2f}" for r in ps)
        + f" percentage points for seeds 411, 412 and 413. The mean ADMM-minus-magnitude loss difference was {avg(ps, 'admm_minus_magnitude_ce'):+.3f}."
    )
p(
    "At 80% sparsity, ADMM reduced mean test cross-entropy even as its mean accuracy fell. At 90%, magnitude pruning won on both measures in the three-seed average. The 50% models had four additional training epochs beyond the dense reference, so their gains cannot be attributed to pruning alone."
)
p(
    "The starting checkpoint mattered most at 90% sparsity. With the split and minibatch order fixed, ADMM test accuracy ranged from 51.96% to 62.70% across starts. Magnitude pruning ranged from 65.27% to 68.69%. Macro recall also fell to 40.54% for ADMM versus 56.23% for magnitude pruning. Since macro recall weights each class equally, the accuracy gap alone understates the difference in average per-class recall. Three starts on one split limit how widely these results generalize."
)

page("Convergence and computational effort")
fig(
    "solver_behavior.png",
    6.45,
    "Figure 2. Seed 411 at 80% sparsity. Training objective is averaged along each epoch’s updating weights. Validation loss is measured at the epoch boundary. ADMM fixes the support after update 738. At update zero, magnitude pruning already has a sparse mask, whereas ADMM still has the dense starting weights. The validation curves become directly comparable as feasible models at update 984.",
)
hist = read(ROOT / "runs/comparison/seed411/admm80.csv")
rs = [r for r in hist if r["primal"]]
p(
    f"The primal residual ‖w − z‖₂ went from {float(rs[0]['primal']):.2f} to {float(rs[-1]['primal']):.2f} across the three outer steps. The dual residual ρ‖zᵗ − zᵗ⁻¹‖₂ went from {float(rs[0]['dual']):.4f} to {float(rs[-1]['dual']):.4f}. These trends provide no evidence of convergence to consensus under the short budget. Final projection enforces feasibility even when the learning and sparse iterates still disagree."
)
h90 = read(ROOT / "runs/comparison/seed411/admm90.csv")
p(
    f"At 90% for the same seed, validation accuracy was {100 * float(h90[-2]['accuracy']):.2f}% on the weight iterate immediately before final masking and {100 * float(h90[-1]['accuracy']):.2f}% after masked retraining. Intermediate accuracy overstated what the final sparse classifier delivered."
)
table(
    ["Weights pruned", "Magnitude time (s)", "ADMM time (s)", "Updates per method"],
    [
        [
            f"{s}%",
            stats(subset("magnitude", s), "training_seconds", digits=1),
            stats(subset("admm", s), "training_seconds", digits=1),
            "984",
        ]
        for s in (50, 80, 90)
    ],
)
p(
    "Table 2. Training seconds (mean ± sample SD) for equal update budgets. At 80%, the means were 12.0 s for magnitude pruning and 12.1 s for ADMM. Both methods use dense training kernels. Magnitude pruning sorts P weights once. ADMM repeats the global sort three times, with O(P log P) work per sort and O(P) storage for its auxiliary vectors. The timing spread is comparable to several mean differences. These runs support similar training cost on this GPU. Deployment latency would require a separate inference benchmark."
)

page("Sensitivity and conclusions")
table(
    [
        "Setting",
        "ρ",
        "Learning rate",
        "Valid. accuracy (%)",
        "Valid. loss",
        "Primal residual",
    ],
    [
        [
            r["setting"].replace("comparison", "nominal").replace("_", " "),
            r["rho"],
            r["learning_rate"],
            f"{100 * float(r['validation_accuracy']):.2f}",
            f"{float(r['validation_ce']):.3f}",
            f"{float(r['final_split_primal']):.2f}",
        ]
        for r in sweep
    ],
)
p(
    "Table 3. ADMM at 80% sparsity, seed 411. Residuals precede final masking. All five retrained models met the 19,705-weight cap."
)
best = min(sweep, key=lambda r: float(r["validation_ce"]))
p(
    "Doubling ρ reduced the primal residual from 26.58 to 20.13, yet validation loss rose from 0.787 to 0.800. Doubling the learning rate improved accuracy by 1.09 points while raising loss to 0.808. The nominal settings were fixed before testing."
)
sub("How the global budget was distributed")
table(
    ["Method", "Conv 1", "Conv 2", "Conv 3", "Linear"],
    [
        [m.title() if m == "magnitude" else "ADMM"]
        + [
            f"{avg([r for r in layers if r['method'] == m and r['sparsity'] == '80' and r['layer'] == l], 'pct'):.1f}%"
            for l in ("Conv 1", "Conv 2", "Conv 3", "Linear")
        ]
        for m in ("magnitude", "admm")
    ],
)
p(
    "Table 4. Mean layer retention at 80% global sparsity. ADMM kept 42.1% of first-layer weights versus 37.9% for magnitude pruning. Both kept about 16% of third-layer weights. One global budget allowed this uneven allocation."
)
sub("Quality of the feasible solutions")
means = {
    (m, s): avg(
        [r for r in sparse_audit if r["method"] == m and int(r["sparsity"]) == s],
        "training_F",
    )
    for m in ("magnitude", "admm")
    for s in (50, 80, 90)
}
p(
    "We evaluated F, including its weight penalty, on all 31,379 training images. The three-seed means (magnitude, ADMM) were "
    + ", ".join(
        f"{means['magnitude', s]:.3f} and {means['admm', s]:.3f} at {s}%"
        for s in (50, 80, 90)
    )
    + ". At 80%, magnitude pruning fit the training objective better, while ADMM achieved lower test cross-entropy."
)
p(
    "Holding all weights fixed leaves the 43 output biases free. Their gradient g is the mean predicted class-probability vector minus the observed class frequencies. F is smooth in these biases. The feasible direction −g has directional derivative −‖g‖₂²."
)
decreases = [float(r["feasible_decrease"]) for r in sparse_audit]
norms = [float(r["output_bias_gradient_norm"]) for r in sparse_audit]
p(
    f"All 18 sparse checkpoints had nonzero gradient norms ({min(norms):.4f}–{max(norms):.4f}). Subtracting g from the biases reduced F by {min(decreases):.5f}–{max(decreases):.5f}, preserving every support. A feasible descent direction rules out local and global minimality at each saved point. The original checkpoints supplied all test metrics."
)
sub("Conclusion and next experiment")
p(
    "At 90% sparsity, magnitude pruning led by 10.80 test-accuracy points and lowered mean test loss by 0.428. It had four epochs to adapt to its final support. ADMM had one. A two-epoch search followed by two masked epochs would test that explanation at the same 984-update budget. Repeating the comparison on other track splits would test robustness."
)

page("References and reproducibility")
for text, url in [
    (
        "[1] S. Han, J. Pool, J. Tran and W. J. Dally. Learning both Weights and Connections for Efficient Neural Network. NeurIPS, 2015.",
        "https://proceedings.neurips.cc/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html",
    ),
    (
        "[2] M. Á. Carreira-Perpiñán and Y. Idelbayev. Learning-Compression Algorithms for Neural Net Pruning. CVPR, 2018.",
        "https://openaccess.thecvf.com/content_cvpr_2018/html/Carreira-Perpinan_Learning-Compression_Algorithms_for_CVPR_2018_paper.html",
    ),
    (
        "[3] T. Zhang et al. A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers. ECCV, 2018.",
        "https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.php",
    ),
    (
        "[4] J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. IJCNN, 2011.",
        "https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf",
    ),
    (
        "[5] A. Renda, J. Frankle and M. Carbin. Comparing Rewinding and Fine-tuning in Neural Network Pruning. ICLR, 2020.",
        "https://arxiv.org/abs/2003.02389",
    ),
    (
        "[6] D. Blalock, J. J. Gonzalez Ortiz, J. Frankle and J. Guttag. What is the State of Neural Network Pruning? MLSys, 2020.",
        "https://proceedings.mlsys.org/paper_files/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html",
    ),
    (
        "[7] X. Luo, J. Zhu and Q. Yu. Efficient convNets for fast traffic sign recognition. IET Intelligent Transport Systems, 2019.",
        "https://doi.org/10.1049/iet-its.2018.5489",
    ),
]:
    cite(text, url)
sub("Reproducing the analysis")
p(
    "The Python package includes exact settings and checkpoint hashes. To reproduce Table 1, run evaluate.py on the supplied checkpoints and group results/test_results.csv by method and sparsity. figures.py reconstructs both figures. To reproduce the optimality check, run prepare.py and audit_objective.py. The resulting training_objective_audit.csv records each checkpoint’s training loss, penalty, bias-gradient norm and feasible objective decrease."
)
# Use the same typeface in the footer, captions, hyperlinks and underlying style defaults.
for font in doc.styles.element.xpath(".//w:rFonts"):
    for field in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme", "csTheme"):
        font.attrib.pop(qn("w:" + field), None)
    for field in ("ascii", "hAnsi", "eastAsia", "cs"):
        font.set(qn("w:" + field), "Times New Roman")
for r in footer.runs:
    r.font.name = "Times New Roman"
doc.save(DEST / "Group28_Technical_Report_Draft.docx")
