# DJGSC

Image Classification via Discriminative Multi-Task Group Sparse Learning

This repository provides a MATLAB implementation of the proposed **Discriminative Joint Group Sparse Classifier (DJGSC)** for image classification.

The demo script uses the **Extended Yale B** face dataset as an example. It constructs training and testing samples, learns the representation coefficient matrix using DJGSC, classifies each test image according to the minimum reconstruction residual, and records all experimental results in a TXT file.

---

## 1. Project Structure

```text
project/
│
├── demo.m                         # Main experiment script
├── yaleborigin.mat                # Extended Yale B dataset
├── DJGSC.m                        # Proposed DJGSC optimization function
├── jgsc_yaleb_param_search.txt    # Output result file, generated automatically
└── README.md                      # Usage instructions
```

---

## 2. Dataset

The experiment uses the **Extended Yale B** face dataset.

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
nSet = 38;
number = 64;
```

This means the experiment uses 38 subjects/classes, and each subject contains 64 images.

The dataset is assumed to be arranged class by class:

```text
Class 1: images 1–64
Class 2: images 65–128
...
Class 38: images 2369–2432
```

---

## 3. Main Experimental Settings

The main experimental settings in `demo.m` are:

```matlab
nSet = 38;
nmax = 1;
number = 64;
number_train = 32;
```

Parameter meanings:

| Parameter | Description |
|---|---|
| `nSet` | Number of classes |
| `nmax` | Number of repeated runs |
| `number` | Number of images per class |
| `number_train` | Number of training images per class |

---

## 4. Parameter Search

The demo performs a grid search over two regularization parameters of DJGSC:

```matlab
lambda_list = 10.^(-4:2);
gamma_list  = 10.^(-4:2);
```

That is:

```text
0.0001, 0.001, 0.01, 0.1, 1, 10, 100
```

The function call is:

```matlab
W = DJGSC(S, T, number_train(n1), lambda1, gamma1);
```

where:

| Parameter | Description |
|---|---|
| `lambda1` | Regularization parameter for the $\ell_{2,1,2}$ mixed-norm term |
| `gamma1` | Regularization parameter for the discriminative regularization term |

The script evaluates all combinations of `lambda1` and `gamma1`.

---

## 5. Training and Testing Split

In the default setting, the first 32 images of each subject are used for training:

```matlab
train_numbers = 1:number_train(n1);
```

Since `number_train = 32`, this means:

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

## 6. Representation Coefficient Learning

DJGSC learns a representation coefficient matrix:

```matlab
W
```

The size of `W` is:

```text
number of training samples × number of testing samples
```

For Extended Yale B under the default setting:

```text
number of training samples = 38 × 32 = 1216
number of testing samples  = 38 × 32 = 1216
```

Therefore:

```text
size(W) = 1216 × 1216
```

The optimization problem solved by DJGSC can be written as:

```text
min_X ||T - S X||_F^2
    + lambda ||X||_{2,1,2}^2
    + gamma sum_{i != j} ||X_i X_j^T||_F^2
```

where:

- `S` is the training dictionary;
- `T` is the testing sample matrix;
- `X` is the representation coefficient matrix;
- `X_i` denotes the coefficient block corresponding to class `i`;
- the $\ell_{2,1,2}$ mixed norm encourages joint group sparsity;
- the discriminative term encourages separation among class-specific coefficient blocks.

---

## 7. Classification Rule

After obtaining the representation coefficient matrix `W`, each test sample is classified according to the minimum reconstruction residual.

For each test sample `j` and each class `k`, the residual is computed as:

```matlab
fi(j,k) = norm(T(:,j) - S(:,S_label == k) * W(S_label == k,j));
```

The predicted label is assigned to the class with the minimum residual:

```matlab
[~, max_indices] = min(fi, [], 2);
```

---

## 8. Accuracy Calculation

The classification accuracy is calculated by comparing the predicted labels with the ground-truth labels:

```matlab
correct_num = 0;

for j = 1:numRows
    if T_label(1,j) == max_indices(j,1)
        correct_num = correct_num + 1;
    end
end

acc = correct_num / numRows;
```

For each pair of `lambda1` and `gamma1`, the accuracy and running time are printed in the MATLAB command window.

Example output:

```text
lambda=0.01, gamma=10, run=1, acc=0.9662, time=8.123456
```

---

## 9. Output Results

The experimental results are saved to:

```text
jgsc_yaleb_param_search.txt
```

For each parameter pair, the output includes:

| Output | Description |
|---|---|
| `lambda` | Regularization parameter for the $\ell_{2,1,2}$ mixed norm |
| `gamma` | Regularization parameter for the discriminative term |
| `run` | Current repeated experiment index |
| `class` | Number of classes |
| `trainnumber` | Number of training images per class |
| `acc` | Classification accuracy of the current run |
| `time` | Running time of the current run |
| `avgbcc` | Average classification accuracy |
| `minbcc` | Minimum classification accuracy |
| `maxbcc` | Maximum classification accuracy |
| `avgtim` | Average running time |
| `stdbcc` | Standard deviation of accuracy |

Example output:

```text
Current parameters: lambda=0.01, gamma=10

run=1, lambda=0.01, gamma=10, class=38, trainnumber=32, acc=0.9662, time=8.123456

SUMMARY: lambda=0.01, gamma=10, class=38, trainnumber=32, avgbcc=0.9662, minbcc=0.9662, maxbcc=0.9662, avgtim=8.123456, stdbcc=0.000000
```

At the end of the TXT file, the best parameter combination is recorded:

```text
BEST RESULT
best_lambda=0.01, best_gamma=10, best_acc=0.9662, best_time=8.123456
```

---

## 10. How to Run

Open MATLAB and set the current folder to the project directory.

Then run:

```matlab
demo
```

Before running the code, make sure the required function path has been added if necessary:

```matlab
addpath('D:\my_SRC-main\fista_lasso.m')
```

You may need to change this path according to your own file location.

---

## 11. Required Files

The following files are required:

```text
yaleborigin.mat
demo.m
DJGSC.m
```

If any function is missing, MATLAB may report an error such as:

```text
Undefined function or variable 'DJGSC'
```

In this case, check whether `DJGSC.m` exists and whether its folder has been added to the MATLAB path.

---

## 12. Optional Random Split

The current code uses a fixed training/testing split:

```matlab
train_numbers = 1:32;
test_numbers = 33:64;
```

To use a random split, replace the above lines with:

```matlab
perm = randperm(number);
train_numbers = perm(1:number_train(n1));
test_numbers = perm(number_train(n1)+1:end);
```

This will randomly select training and testing samples for each class.

---

## 13. Code Summary

The main workflow of the demo is:

1. Load the Extended Yale B dataset.
2. Set experimental parameters.
3. Define candidate values of `lambda1` and `gamma1`.
4. Construct training and testing sets.
5. Compute representation coefficients using DJGSC.
6. Classify test samples by minimum reconstruction residual.
7. Calculate classification accuracy.
8. Repeat the experiment for all parameter combinations.
9. Save all results to `jgsc_yaleb_param_search.txt`.
10. Record the best parameter combination at the end of the TXT file.

---

## 14. Notes

- The dataset should be arranged class by class.
- Each subject should contain 64 images.
- The current experiment uses 32 images per subject for training and 32 images for testing.
- The result file `jgsc_yaleb_param_search.txt` will be created automatically if it does not already exist.
- If the result file already exists, new results will be appended to the end of the file.
- For reproducibility, please report the selected `lambda` and `gamma` values when using this code.
- If datasets cannot be redistributed due to licensing restrictions, users should download them from the official dataset source and follow the same preprocessing protocol.

---

## 15. Citation

If you use this code, please cite the following paper:

```text
Image Classification via Discriminative Multi-Task Group Sparse Learning
```
