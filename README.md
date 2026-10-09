# SpinLab

SpinLab predicts the fiber diameter of solution blow spun polymer fibers. It accompanies the article "Machine Learning and Deep Learning Prediction of Nanofiber Diameter in Solution Blow Spinning Using Physics-Derived Polymer Descriptors".

SpinLab uses the eight machine-learning models of the study (Extra Trees, Random Forest, XGBoost, LightGBM, CatBoost, SVR, ElasticNet, and KNN), trained on 344 literature data points from 47 papers.

- **Inputs:** polymer concentration, intrinsic viscosity (from Mark–Houwink constants), c/c*, solvent surface tension, solvent boiling point, air pressure, feed rate, and working distance.
- **Zero-shot mode:** predictions of the pre-trained models.
- **Calibration mode:** the frozen models are applied to your own measured conditions, and an affine correction of the predicted ln(diameter) is fitted to your measured diameters. The models are not retrained, and your data are not combined with the literature data.
- **Domain warning:** a warning is shown when inputs lie outside the domain of the training data.
- **Scope:** the models predict diameter for conditions that form fibers. They do not predict whether fibers will form.

## How to run

1. Download `final_models_ml.pkl` from the Zenodo record https://doi.org/10.5281/zenodo.23254490 and place it in this folder, next to `app.py`.
2. Install the packages (Python 3.10 or later):
   ```
   pip install -r requirements.txt
   ```
3. Start SpinLab:
   ```
   python app.py
   ```
4. Open http://localhost:7860 in a browser.

The dataset, the analysis pipeline, and all trained models are archived on Zenodo: https://doi.org/10.5281/zenodo.23254490

## License

MIT
