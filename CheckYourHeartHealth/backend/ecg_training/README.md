# ECG experiment

Data: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) by Wagner et al., DOI 10.13026/kfzx-aw45. Follow the dataset license and cite the authors. Data is not bundled here.

Run these scripts from this directory. Download `ptbxl_database.csv`, `scp_statements.csv` (save as `scp.csv`) and the `records100` waveforms from the dataset. Put each `.dat` in `dat/` with its original basename. The preparation script decodes the dataset's 16-bit interleaved, 1000 ADC-units/mV signals into mV. This is specific to PTB-XL, not a general WFDB reader.

```
python prep.py
python cnn.py 12 90
# Re-run cnn.py with the same epoch count to resume its checkpoint.
python ev.py
python lgbf.py
```

Dependencies are those in the app requirements. CPU threads: 2. Seed: 0. AdamW, 12 epochs, batch size 64, OneCycle learning rate, random time shifts. Architecture is in cnn.py. Normalization mean/std comes only from the first 3,000 training records and is saved in mu.npy. The app carries the same constants.

Five diagnostic superclasses, multi-label. Exclude records without a diagnostic superclass. Official patient-wise folds: 1-8 training (17,084), 9 validation (2,146), 10 held-out test (2,158). Patient-set intersections were checked and are zero. Best epoch is chosen on validation macro AUC, never test AUC. Model selection should not use test results for future tuning.

CNN test macro AUC 0.9303. Feature LightGBM macro AUC 0.8764 (fixed feature set and parameters, not extensively tuned). These are single-run measurements, without confidence intervals. The app stores the CNN weights at float16 precision, loads them into a float32 model, and uses independent sigmoid scores. This conversion leaves the reported four-decimal test AUC unchanged. AUC is a ranking metric, not accuracy or medical certainty. Scores have not been clinically calibrated.

`lgbf.py` describes the comparison features. `ev.py` evaluates the selected checkpoint. Keep test data untouched during changes, or use a new independent test set. To update app weights, export the selected CNN state as float16, torch.save to memory, zlib-compress, base64-encode, split into five chunks under src/ecg_weights_0.b64 ... src/ecg_weights_4.b64, and update normalization constants in src/ecg_classifier.py. These encoded files are model weights, not Python source.
