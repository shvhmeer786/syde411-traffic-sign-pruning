# Pruning neural networks for traffic sign recognition

SYDE 411 · Group 28 · Shahmeer Ali, Amin Ambike and Brayden Petersen

Which weights should a traffic-sign classifier keep? We compared global magnitude pruning and ADMM on a 43-class CNN. Each pair started from the same dense checkpoint, received 984 gradient updates and had the same final weight budget.

| Weights pruned | Weight cap | Magnitude accuracy | ADMM accuracy |
| ---: | ---: | ---: | ---: |
| 50% | 49,264 | 82.19% | 82.24% |
| 80% | 19,705 | 80.46% | 79.79% |
| 90% | 9,852 | 67.37% | 56.57% |

Mean accuracy on the 12,630 official GTSRB test images across three initializations. At 90% sparsity, magnitude pruning won in every pair. The comparison applies to the four-epoch pruning budget.

[Training and pruning notebook](notebooks/train_and_prune.ipynb) · [Original results notebook](notebooks/results.ipynb) · [Technical report (PDF)](deliverables/Group28_Technical_Report_Draft.pdf) · [AI acknowledgement](deliverables/Group28_AI_Acknowledgement.pdf)

## Run the project in Jupyter

Clone or download this repository. From its root, create a Python 3.11 environment and start JupyterLab:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-notebook.txt
python -m jupyterlab notebooks/train_and_prune.ipynb
```

On Windows PowerShell, use `py -3.11 -m venv .venv` and `.\.venv\Scripts\Activate.ps1` for the first two commands. The first data download requires internet access and `curl`. Training uses Apple MPS when available and CPU elsewhere.

- **For one experiment**, run steps 1–5. Train a dense CNN, prune half its weights with each method, and compare validation accuracy and per-layer retention.
- **For the whole project**, choose **Run → Run All Cells**. Steps 6–9 finish the three-seed grid, run the ADMM sensitivity study, evaluate the official test set, audit the training objective, and generate both report figures. The test table displays your measurements alongside the original report.
- **To resume**, keep the same run name and rerun the cells. Completed models are reused. To retrain from scratch, set a new name in step 1. Checkpoints, CSVs, and figures go in `notebook-runs/<run_name>/`.

## Explore the original results

The original experiments ran as Python scripts on an Apple M1 Pro. The [results notebook](notebooks/results.ipynb) reconstructs the report table from those saved runs, verifies the weight caps and update counts, and displays the figures. GitHub shows its saved outputs. For analysis alone, install `requirements-notebook.txt` and open `notebooks/results.ipynb` in JupyterLab.

## Reproduce the experiments

Install the training dependencies in the same environment with `python -m pip install -r requirements.txt`. The scripts use MPS on supported Macs and otherwise use CPU. Numerical results and timing can vary with hardware and library versions.

| Step | Command | Output |
| --- | --- | --- |
| Prepare data | `python prepare.py` | Verified official training archive, RGB crops and class-stratified sign-track split |
| Train and compare | `python grid.py` | Three dense checkpoints and nine paired comparisons at 50%, 80% and 90% sparsity |
| Vary hyperparameters | `python sensitivity.py` | Validation runs varying either the learning rate or ADMM penalty |
| Evaluate | `python evaluate.py` | Official test metrics, paired differences, layer counts and sensitivity table |
| Audit optimality | `python audit_objective.py` | Full training objective and feasible bias-step check for saved checkpoints |
| Redraw figures | `python figures.py` | Both report figures |

**Evaluate existing checkpoints.** The repository includes the completed runs. Start with `evaluate.py` to regenerate test tables from them. It downloads the official test archives as needed. Run `prepare.py` before the training-objective audit.

**Train from scratch.** To run the six commands above from a checkout with a fresh output directory, set `export SYDE411_WORKSPACE="$PWD/notebook-runs/cli-repeat"` before running the first command. The grid and sensitivity scripts reuse completed checkpoints in the selected workspace. With the default workspace (the repository root), you can also prepare the data and run a single fresh pair under an unused tag:

```sh
python experiment.py dense --epochs 12 --lr .003 --tag repeat
python experiment.py magnitude --epochs 4 --lr .001 --tag repeat
python experiment.py admm --epochs 4 --lr .001 --tag repeat
python summarize.py runs/repeat/seed411
```

## Data and protocol

Use the [dataset authors’ official training archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). The data builder verifies the archive checksum, crops the annotated sign regions and resizes them to 32 × 32 RGB. Seed 411 assigns whole tracks within each class to training (31,379 images) or validation (7,830). No physical sign track crosses that split.

The CNN has 98,528 prunable weights and 267 free biases. Dense training uses 12 epochs, Adam, batch size 128 and learning rate 0.003. Both pruning methods then get four epochs at learning rate 0.001, with the same batch order. Magnitude pruning fixes its global mask immediately. ADMM uses three outer iterations with ρ = 0.001 and one final fixed-mask epoch. The regularization coefficient is 0.00001. The main comparison uses initialization seeds 411, 412 and 413. Full configurations are saved alongside the checkpoints.

## Repository layout

- `data/manifest.csv` records the partition, class and track of each training-archive image. The image archives and prepared arrays are downloaded or regenerated locally.
- `runs/` contains the validation pilot, main grid and sensitivity runs, including checkpoints, configurations and logs.
- `results/` contains the seed-level CSVs and publication figures.
- `notebooks/` contains the end-to-end training walkthrough and the executed analysis of the original runs.
- `deliverables/` contains the editable report, separate acknowledgement and their PDFs.
- `authoring/` contains the report and acknowledgement builders (require `python-docx` and `lxml`) and `package.py` to build the course code ZIP.

## References

- Han et al. [1] motivate pruning by magnitude and retraining the surviving connections. [NeurIPS paper](https://proceedings.neurips.cc/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html)
- Carreira-Perpiñán and Idelbayev [2] justify the Euclidean cardinality projection. Zhang et al. [3] give the ADMM split and masked retraining. Our projection uses one budget across all layers, as specified in the proposal. [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Carreira-Perpinan_Learning-Compression_Algorithms_for_CVPR_2018_paper.html) · [ECCV paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.php)
- Stallkamp et al. [4] supply the benchmark and describe sequences of images of the same sign. [Dataset authors](https://github.com/houbensebastian/GermanTrafficSignBenchmarks)
- Renda et al. [5] and Blalock et al. [6] motivate attention to retraining and fair comparisons. Luo et al. [7] provide application context for compact traffic-sign classifiers. [Renda](https://arxiv.org/abs/2003.02389) · [Blalock](https://proceedings.mlsys.org/paper_files/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html) · [Luo](https://doi.org/10.1049/iet-its.2018.5489)
