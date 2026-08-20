# HandXRay-Inpainting

app.py: Contains helper functions for running inference with the bone age estimation model.

bone_age_results_confidence_intervals.py: Computes MSE and RMSE against ground truth and calculates confidence intervals.

confusion_matrix_gender.py: Generates the confusion matrix for gender classification results.

debbug_notebook.ipynb: Analyzes statistics from both the original test set and the inpainted dataset.

display_samples.py: Plots side-by-side comparisons of original and inpainted images.

generate_mult.py: Uses the OpenAI API to generate synthetic images for the dataset.

mult_orignal_vs_calibrated.py: Performs regression analysis, applies curve calibration, and generates related plots.

pixel_analysis.py: Produces histograms of pixel intensity distributions.

train_gender.py: Trains the gender classification model.

validate_boneage.py: Runs inference using the bone age model.

validate_fender.py: Runs inference using the gender classifier.
