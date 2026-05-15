# DJGSC
Image Classification via Discriminative Multi-Task Group Sparse Learning

# DJGSC Classification Experiment on Extended Yale B for example

This project provides a MATLAB implementation for evaluating a sparse representation-based classification method on the Extended Yale B face dataset.

The code constructs training and testing samples, computes the representation coefficient matrix using `DJGSC`, and classifies each test image according to the minimum reconstruction residual.

---

## 1. Project Structure

```text
project/
│
├── demo.m                 # Main experiment script
├── yaleborigin.mat        # Extended Yale B dataset
├── DJGSC.m               # Proposed classification method
├── fista_lasso.m          # Optimization function
└── jgsc_yaleb.txt         # Output result file
```

---

## 2. Dataset

The experiment uses the Extended Yale B face dataset.

The dataset file should be named:

```text
yaleborigin.mat
```

The loaded `.mat` file should contain the variable:

```matlab
dataset
```

Each column of `dataset` represents one face image sample.

In the default setting:

```matlab
nClass = 38;
nImgPerClass = 64;
```

This means the experiment uses 38 subjects, and each subject contains 64 images.

---

## 3. Main Parameters

The main experimental parameters are defined as follows:

```matlab
nClass = 38;
nImgPerClass = 64;
nRepeat = 10;
nTrain = 32;
```

Parameter meanings:

| Parameter | Description |
|---|---|
| `nClass` | Number of subjects/classes |
| `nImgPerClass` | Number of images per subject |
| `nRepeat` | Number of repeated experiments |
| `nTrain` | Number of training images per class |

---

## 4. Training and Testing Split

In the default setting, the first 32 images of each subject are used for training:

```matlab
train_numbers = 1:32;
```

The remaining 32 images are used for testing:

```matlab
test_numbers = 33:64;
```

The training samples are stored in:

```matlab
S
```

The testing samples are stored in:

```matlab
T
```

The corresponding labels are stored in:

```matlab
S_label
T_label
```

---

## 5. Representation Coefficient Learning

The representation coefficient matrix `W` is obtained by calling:

```matlab
W = DJGSC(S, T, nTrain, 0.01, 10);
```

Here, `W` represents the contribution of each training sample when reconstructing each test sample.

The size of `W` is:

```text
number of training samples × number of testing samples
```

---

## 6. Classification Rule

For each test sample, the reconstruction residual is computed for every class.

The residual for class `c` is calculated as:

```matlab
R = T - S(:, idx) * W(idx, :);
residual(:, c) = sqrt(sum(R.^2, 1))';
```

where `idx` indicates the training samples belonging to class `c`.

The predicted label is assigned to the class with the minimum reconstruction residual:

```matlab
[~, pred_label] = min(residual, [], 2);
```

---

## 7. Accuracy Calculation

The classification accuracy is calculated by comparing the predicted labels with the ground-truth labels:

```matlab
acc = mean(pred_label' == T_label);
```

The accuracy of each run is printed in the MATLAB command window:

```text
Run 1/10, acc = 0.956414
Run 2/10, acc = 0.962171
Run 3/10, acc = 0.959704
...
```

---

## 8. Output Results

The final statistical results are saved to:

```text
jgsc_yaleb.txt
```

The output includes:

| Output | Description |
|---|---|
| `avgbcc` | Average classification accuracy |
| `minbcc` | Minimum classification accuracy |
| `maxbcc` | Maximum classification accuracy |
| `avgtim` | Average running time |
| `stdbcc` | Standard deviation of accuracy |
| `alpha` | Experimental parameter |

Example output:

```text
#n=10.0000, class:38.0, trainnumber:32.0, avgbcc:0.9605, minbcc:0.9523, maxbcc:0.9688, avgtim:1.234567, stdbcc:0.004321, alpha:0.100000
```

---

## 9. How to Run

Open MATLAB and set the current folder to the project directory.

Then run:

```matlab
main
```

Before running the code, make sure the required function path has been added:

```matlab
addpath('D:\my_SRC-main');
```

You may need to change this path according to your own file location.

---

## 10. Required Files

The following files are required:

```text
yaleborigin.mat
GLSRC2.m
fista_lasso.m
```

If any function is missing, MATLAB may report an error such as:

```text
Undefined function or variable 'GLSRC2'
```

In this case, check whether the corresponding `.m` file exists and whether its folder has been added to the MATLAB path.

---

## 11. Optional Random Split

The current code uses a fixed training/testing split:

```matlab
train_numbers = 1:32;
test_numbers = 33:64;
```

To use a random split, replace the above lines with:

```matlab
perm = randperm(nImgPerClass);
train_numbers = perm(1:nTrain);
test_numbers = perm(nTrain+1:end);
```

This will randomly select training and testing samples for each class.

---

## 12. Code Summary

The main workflow of the code is:

1. Load the Extended Yale B dataset.
2. Set experimental parameters.
3. Construct training and testing sets.
4. Compute representation coefficients using `GLSRC2`.
5. Classify test samples by minimum reconstruction residual.
6. Calculate classification accuracy.
7. Repeat the experiment several times.
8. Save the final statistical results.

---

## 13. Notes

- The dataset should be arranged class by class.
- Each subject should contain 64 images.
- The current experiment uses 32 images per subject for training and 32 images for testing.
- The result file `jgsc_yaleb.txt` will be created automatically if it does not already exist.
- If the file already exists, new results will be appended to the end of the file.assification method
