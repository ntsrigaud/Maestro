# Work Log

## Log Information

- **Date:** 2025-12-04
- **Developer:** Neil Taison Rigaud
- **Branch:** finetune-vit-hagrid
- **Related Issue/Feature:** Jupyter Notebook for Finetuning ViT-B16 on the HaGrid Dataset
- **Module(s):** recognition-engine

---

## Objective

In this work session, the objective is to learn how to finetune a base Vision Transformer model to recognize simple gestures. We will use the HaGrid hand gesture dataset to train the model to do this task.

---

## Tasks Checklist

- [x] Create a notebook with a conda setup and training cells for the training.
- [ ] Configure or download the dataset to start training the model
- [ ] Train the model using the GPU ressources provided by the lab using SSH connection

---

## Work Log

### [17:40] - Session Start

**What I'm working on:**

- **Current state of the code:**

  - [x] Create the Jupyter notebook
  - [x] Set up the conda environment
  - [x] Research on selecting customized gesture samples from specific category from the dataset under 1-3 GB
  - [x] Create a download helper script for specific gestures with explanatory documentation
  - [ ] Provide different download options depending on the size using Kaggle downloader and HuggingFace as alternatives
  - [ ] Use Kaggle option for local training
  - [ ] Download and verify data
  - [x] Update notebook to use subset dataset structure and add instructions for running the download script
  - [ ] Add cell to visualize sample images and update the model
  - [ ] Add hyperparameter configuration
  - [ ] Update training loop cells and add comprehensive training orchestration cell
  - [ ] Add cell for plotting training history and evaluating test set
  - [ ] Include next steps for model export and integration
  - [ ] Summary document for training setup
  - [x] Create environment YAML file for conda to make setup easier

- **Changes made:**

- Files modified:
  - None
- Files created:
  - notebooks/finetune_vit_hagrid.ipynb
  - environment.yml
  - scripts/download_hagrid_kaggle.py
  - scripts/download_hagrid_subset.py
  - scripts/download_hagrid_huggingface.py
  - docs/HAGRID_DOWNLOAD_GUIDE.md
  - docs/HAGRID_DOWNLOAD_SOLUTIONS.md
  - docs/TRAINING_SETUP_SUMMARY.md
  - docs/HaGRID_DATASET_ASSESSMENT.md
  - QUICK_REFERENCE.md
  - TRAINING_README.md
- Files deleted:
  - None

**Observations/Discoveries:**

- **Issues encountered:**

  - Downloading the full HaGRID dataset is impractical due to size (280+ GB). Need to implement a more efficient method.
  - Need to provide multiple download options for users with different bandwidth and storage constraints.

- **Solutions applied:**
  - Created a Kaggle-based downloader script to fetch a pre-made subset of the dataset (~1.5 GB).
  - Developed a custom downloader script to allow users to select specific gesture classes and sample sizes.
  - Documented the download process and troubleshooting steps in `HAGRID_DOWNLOAD_GUIDE.md` and `HAGRID_DOWNLOAD_SOLUTIONS.md`.

### [20:00] - Session End

**Summary of completed work:**

- **Current status:**

- [x] Created Jupyter notebook for finetuning ViT on HaGRID dataset
- [x] Configured multiple dataset download options
- [ ] Downloaded and verified dataset using Kaggle method
- [ ] Updated notebook for subset dataset structure

**Next steps:**

- **Open questions/concerns:**
  - How to best visualize training progress and results in the notebook?
  - What hyperparameter settings yield optimal performance for this task?

## Technical Notes

### Dependencies

- Python 3.8+
- PyTorch
- timm
- transformers
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- kaggle (for Kaggle dataset download)

### Dataset Details

- HaGRID Dataset: Hand gesture recognition dataset with 280+ GB of images and annotations.
- Subset options: Pre-made Kaggle subset (~1.5 GB) or custom subsets using specific gestures and sample sizes.
- Download scripts located in `scripts/` directory.
- Dataset structure: `data/hagrid_subset/` with `annotations.csv` for labels.
- Training/validation/test splits maintained as per original dataset ratios.

### Configuration Changes

- Added `environment.yml` for easy conda environment setup.
- Updated notebook to accept different dataset paths based on download method.

### Performance Considerations

- Training time expected to be ~30-60 minutes on GPU for 8,000 images.
- Model size after training: ~330 MB for ViT-B16 finetuned on HaGRID.
- Expected accuracy: 85-90% on test set after 10 epochs.
- Training can be adjusted based on available hardware (CPU vs GPU).

---

## References

- Documentation:
  - [HaGRID Dataset Assessment](docs/HaGRID_DATASET_ASSESSMENT.md)
  - [Download Guide](docs/HAGRID_DOWNLOAD_GUIDE.md)
  - [Training Setup Summary](docs/TRAINING_SETUP_SUMMARY.md)
- Related PRs: [link]
- External resources:
  - [HaGRID Dataset on Kaggle](https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p)
  - [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)

---

## Review Notes

**Self-review checklist:**

- [ ] Code follows project style guidelines
- [x] Comments added where necessary
- [x] No debugging code left in
- [ ] Tests pass
- [x] Documentation updated

**Notes for reviewer:**

- **Areas to focus on:**
  - Clarity and completeness of the download scripts and documentation.
  - Correctness of the notebook cells for training and evaluation.
  - Appropriateness of the environment setup for reproducibility.

## Lessons Learned

- Efficient dataset handling is crucial for large datasets; downloading only necessary subsets can save significant time and storage.
- Providing multiple options for dataset access can accommodate users with varying resources.
- Clear documentation is essential for guiding users through complex setup processes.
- Automating environment setup with conda YAML files enhances reproducibility and ease of use.
- Planning for different hardware capabilities (CPU vs GPU) is important for accessibility.
- Visualizing training progress effectively can aid in understanding model performance and diagnosing issues.
- Hyperparameter tuning is a key aspect of achieving optimal model performance and should be considered early in the training process.
- Keeping track of work through detailed logs helps in reflecting on progress and planning future tasks.
