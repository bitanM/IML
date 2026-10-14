---
layout: default
title: "Fisher's LDA Assignment"
nav_order: 1
parent: Fisher's LDA
has_toc: true
permalink: /fisher-lda/Assignment/
---

# Assignment 1: Fisher's Linear Discriminant Analysis

**Course:** Introduction to Machine Learning  
**Instructor:** Dr. Anirban Sen  
**Institution:** Ashoka University  
**Date:** September 2026  

**Total: 30 points** — Take-home submission + live presentation/viva  

---

## Core Task

Reconstruct Fisher's linear discriminant analysis on the classic Iris measurements using basic NumPy linear algebra, **without using a library implementation of LDA**. Compute the scatter matrices, solve the generalized eigenvalue problem, obtain the two discriminant directions, project the observations into two dimensions, and compare the results with Fisher (1936).

---

## Viva Eligibility

Only students who submit a valid assignment by the submission deadline are eligible for the live presentation/viva. **No submission, no viva.**

---

## 1. Background

R. A. Fisher's 1936 paper, *The Use of Multiple Measurements in Taxonomic Problems*, studies how several measurements can be combined into linear functions that discriminate between populations. The paper uses measurements of Iris flowers and develops discriminating linear functions from observed means and within-species sums of squares and products.

Fisher first works explicitly with two species, *Iris setosa* and *Iris versicolor*, and later considers the three-species case including *Iris virginica*. The original paper therefore provides a useful historical reference for both the two-class discriminant calculation and the multi-class extension.

Your task is to implement the central calculation yourself and then explain the linear algebra behind it.

---

## 2. Data

Use the classic Iris dataset with 150 observations, 50 from each of three species, and four flower measurements:

- (a) sepal length
- (b) sepal width
- (c) petal length
- (d) petal width

Fisher's original paper contains the underlying measurements in Table I. You may use a machine-readable copy of the standard Iris dataset for implementation, but verify the observations, class labels, and feature definitions against the original source.

---

## 3. Implementation Restriction

**Restriction for the core calculation:** Do not use a library LDA routine to obtain the discriminant directions.

In particular, do **not** use:
- scikit-learn's `LinearDiscriminantAnalysis`
- MATLAB's built-in discriminant-analysis functions
- R's `lda` routine
- Another ready-made LDA implementation

You may use NumPy for array operations and numerical linear algebra, including matrix multiplication, linear solves, eigendecomposition, and norms. You may use Pandas for data handling and Matplotlib (or an equivalent plotting package) for visualisation.

A library LDA implementation may be used only as a **post-hoc check** after your own implementation is complete. If used, clearly label it as verification and do not use its output as the basis of your solution.

---

## 4. Mathematical Setup

Let an observation be $\mathbf{x} \in \mathbb{R}^4$, and let there be $C = 3$ classes. Let $\boldsymbol{\mu}_k$ denote the mean vector of class $k$ and $\boldsymbol{\mu}$ the overall mean.

The **within-class scatter matrix** is:

$$\mathbf{S}_W = \sum_{k=1}^C \sum_{\mathbf{x}_i \in \text{class } k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

The **between-class scatter matrix** is:

$$\mathbf{S}_B = \sum_{k=1}^C n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

where $n_k$ is the number of observations in class $k$.

For a direction $\mathbf{w}$, Fisher's criterion is:

$$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

Maximising this criterion leads to the generalized eigenvalue problem:

$$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$

For $C$ classes, there are at most $C - 1$ useful discriminant directions. Therefore, for the three Iris species, at most two dimensions are required to represent the discriminant subspace. Because $\mathbf{S}_B$ has rank at most $C - 1$, at most two generalized eigenvalues can be nonzero for this three-class problem, up to floating-point and conditioning effects.

---

## 5. Required Tasks

### 5.1 Task 1 – Load, Inspect, and Document the Data

1. Load the Iris dataset.
2. Report its shape.
3. Identify the four measurements and three species.
4. Verify that each class contains 50 observations.
5. Compute the mean vector of each class.
6. Compute the overall mean vector.
7. Include a compact table of the three class means.

### 5.2 Task 2 – Reconstruct Fisher's Two-Species Calculation

Before moving to the full three-class formulation, reproduce the key idea in Fisher's original setosa-versicolor calculation.

For this task, use **only** the observations from *Iris setosa* and *Iris versicolor*. Let $\mathbf{d} = \boldsymbol{\mu}_2 - \boldsymbol{\mu}_1$.

Construct the within-class scatter matrix using these two classes and solve:

$$\mathbf{S}_W \mathbf{a} = \mathbf{d}$$

Then:
1. Report the resulting coefficient vector $\mathbf{a}$.
2. Rescale the vector so that the sepal-length coefficient is 1.
3. Compare the resulting coefficient ratios with Fisher's reported two-species linear compound.
4. Explain any numerical differences.

Fisher reports, after choosing the coefficient of sepal length to be unity, the approximate two-species linear function:

$$X = x_1 + 5.9037 x_2 - 7.1299 x_3 - 10.1036 x_4$$

**Do not simply copy these numbers.** Obtain your coefficients from your own calculation and then compare them with Fisher's result.

### 5.3 Task 3 – Compute the Within-Class Scatter Matrix

For all three species, implement:

$$\mathbf{S}_W = \sum_{k=1}^C \sum_{\mathbf{x}_i \in \text{class } k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

You must:
1. Show the mathematical expression implemented.
2. Report the resulting $4 \times 4$ matrix.
3. Verify numerically that $\mathbf{S}_W$ is symmetric up to numerical precision.

### 5.4 Task 4 – Compute the Between-Class Scatter Matrix

Implement:

$$\mathbf{S}_B = \sum_{k=1}^C n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

Report:
1. The resulting $4 \times 4$ matrix.
2. Its numerical rank.
3. An explanation of why its rank is at most $C - 1$.

### 5.5 Task 5 – Solve the Generalized Eigenvalue Problem

Solve:

$$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$

A direct NumPy implementation may use:

```python
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
```

but **prefer a linear solve** such as `np.linalg.solve(S_W, S_B)` **over explicitly computing the inverse**.

You must:
1. Compute all eigenvalues and eigenvectors.
2. Sort the eigenpairs by decreasing eigenvalue.
3. Retain the two eigenvectors corresponding to the two largest eigenvalues.
4. Report the eigenvalues and the two retained directions.
5. Compute and report the generalized-eigenvalue residual for each retained direction:
   $$\|\mathbf{S}_B \mathbf{w}_j - \lambda_j \mathbf{S}_W \mathbf{w}_j\|_2$$

For three classes, the remaining discriminant eigenvalues should be zero or numerically close to zero, up to floating-point and conditioning effects.

### 5.6 Task 6 – Project the Iris Observations into Two Dimensions

Form:

$$\mathbf{W} = [\mathbf{w}_1 \quad \mathbf{w}_2] \in \mathbb{R}^{4 \times 2}$$

and project each observation using:

$$\mathbf{z}_i = \mathbf{W}^T \mathbf{x}_i \in \mathbb{R}^2$$

This gives the mapping $\mathbb{R}^4 \to \mathbb{R}^2$.

Create a two-dimensional scatter plot of all 150 projected observations. Clearly identify the three Iris species and label both axes.

### 5.7 Task 7 – Compare Your Result with Fisher (1936)

Read the relevant numerical parts of Fisher's paper, including the discussion of the two-species discriminant and the later three-species construction.

Address the following:

1. How do your class means compare with Fisher's reported means?
2. How do your two-class coefficient ratios compare with Fisher's reported ratios?
3. Fisher's later three-species construction reports a linear compound with coefficients approximately $(-3.308998, -2.759132, 8.866048, 9.392551)$ in the order (sepal length, sepal width, petal length, petal width). How does this relate to your multi-class discriminant analysis?
4. Why need the numerical coefficients not match exactly even when the discriminant direction is effectively the same?
5. What do sign and scale ambiguities of eigenvectors mean in practice?
6. Does your two-dimensional discriminant projection provide qualitative separation consistent with Fisher's analysis?

The paper's reported coefficients are included only to make the comparison precise; students must still perform the calculation and explain the relationship.

### 5.8 Task 8 – Interpretation

Answer briefly and precisely:

1. Why does Fisher's criterion contain within-class scatter in the denominator?
2. Why are there at most two useful discriminant directions for three classes?
3. What does a large generalized eigenvalue indicate?
4. Which original measurements have the largest absolute coefficients in each retained direction? How should those coefficients be interpreted?
5. What information is discarded by reducing the observations from four dimensions to two?

---

## 6. Numerical Verification

Your submission must include at least the following checks:

1. Symmetry check for $\mathbf{S}_W$.
2. Symmetry check for $\mathbf{S}_B$.
3. Numerical rank of $\mathbf{S}_B$.
4. Generalized-eigenvalue residuals for the two retained directions.
5. A check of the dimensions of all matrices and vectors.
6. A consistency check showing that rescaling or sign-flipping an eigenvector does not change its discriminant axis.

For the symmetry checks, report numerical quantities such as $\|\mathbf{S}_W - \mathbf{S}_W^T\|_F$.

For the eigenvector sign/scale check, demonstrate numerically that replacing $\mathbf{w}$ by $c\mathbf{w}$ for a nonzero scalar $c$ produces the same one-dimensional subspace.

---

## 7. Optional Viva Extensions

The following are **not required additions to the written submission**, but you may be asked about them during the live viva:

1. What happens if $\mathbf{S}_W$ is singular or ill-conditioned?
2. Why might standardising the measurements change the discriminant directions?
3. How would unequal class sizes affect $\mathbf{S}_B$?
4. Why is the two-class discriminant based on $\mathbf{S}_W^{-1}\mathbf{d}$?
5. How would you verify your implementation without using sklearn's LDA implementation?
6. Why can two numerically different coefficient vectors represent the same discriminant axis?
7. What information is lost in the two-dimensional projection?
8. How would you classify a new observation after obtaining the discriminant subspace?
9. How would you test whether the observed separation is robust to resampling the dataset?
10. What numerical issues might arise from solving the generalized eigenvalue problem through $\mathbf{S}_W^{-1}\mathbf{S}_B$?

---

## 8. What You May and May Not Use

### Allowed:
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter notebooks
- Standard matrix operations and eigenvalue routines
- The standard Iris dataset
- The original Fisher (1936) paper and other cited references

### Not allowed for the core calculation:
- sklearn's LDA implementation
- MATLAB or R routines that directly return an LDA solution
- Copying the final discriminant coefficients from Fisher without performing the calculation
- A library-generated LDA plot presented as your own implementation

LLMs may be used as learning and coding aids. You remain responsible for understanding and verifying all mathematics, code, numerical results, and sources used in the submission.

---

## 9. Submission Requirements – 5 Points

Submit **one Jupyter notebook (.ipynb)** as the primary submission.

The notebook must:
- Run from top to bottom without manual intervention.
- Contain both code and concise written explanations.
- Display the required numerical outputs and figures.
- Use reproducible code and clearly identified data sources.

The notebook should contain the following sections:

1. Dataset and class means
2. Two-species Fisher reconstruction
3. Three-class within-class scatter
4. Three-class between-class scatter
5. Generalized eigenvalue solution
6. Two-dimensional LDA projection and figure
7. Numerical verification
8. Comparison with Fisher (1936)
9. Interpretation questions
10. References

### 5-Point Submission Rubric

| **Criterion** | **Points** |
|---|---|
| Complete required tasks and outputs | 2 |
| Correct NumPy implementation and numerical results | 2 |
| Clarity, organisation, reproducibility, and citations | 1 |
| **Total** | **5** |

A notebook that runs only partially, has missing outputs, or relies on hidden or manual calculations will not demonstrate the required implementation.

---

## 10. Live Presentation and Viva – 25 Points

The take-home submission will be evaluated through a live presentation and viva. The live component is intended to test whether you understand the mathematics, implementation, numerical results, and interpretation of the work you submitted.

### 10.1 Logistics

Each **student** will receive approximately 10 minutes for the live evaluation:

- Approximately 4 minutes to present the problem, method, key calculations, and main figure.
- Approximately 6 minutes for questions and viva discussion.

**Only students who have submitted the assignment are eligible for the viva. No submission, no viva.**

Be prepared to open your notebook and explain or rerun relevant sections of your implementation during the viva.

The presentation should focus on reasoning and results rather than reading code line by line.

### 10.2 Viva Rubric

| **Component** | **Points** |
|---|---|
| Mathematical derivation and conceptual understanding | 6 |
| NumPy implementation and numerical linear algebra | 5 |
| Scatter matrices, eigenvalues, eigenvectors, and verification | 4 |
| Projection, visualisation, and interpretation | 4 |
| Comparison with Fisher (1936) and interpretation of differences | 4 |
| Clarity, independence, and quality of responses | 2 |
| **Total** | **25** |

---

## 11. Final Checklist

Before submitting, verify:

- [ ] Iris data loaded and documented
- [ ] Three class means and overall mean computed
- [ ] Two-species Fisher calculation reproduced
- [ ] $\mathbf{S}_W$ computed from its definition
- [ ] Symmetry of $\mathbf{S}_W$ verified
- [ ] $\mathbf{S}_B$ computed from its definition
- [ ] Numerical rank of $\mathbf{S}_B$ reported
- [ ] Generalized eigenproblem solved using NumPy
- [ ] Two leading discriminant directions identified
- [ ] Eigenvalue residuals reported
- [ ] All 150 observations projected to 2D
- [ ] Three species shown in final 2D plot
- [ ] Eigenvector sign/scale ambiguity explained
- [ ] Interpretation questions answered
- [ ] Fisher (1936) comparison completed
- [ ] Notebook runs reproducibly
- [ ] All external sources cited

---

## 12. References

1. R. A. Fisher, "The Use of Multiple Measurements in Taxonomic Problems," *Annals of Eugenics*, 7(2), 179–188, 1936. doi:10.1111/j.1469-1809.1936.tb02137.x.

---

> **Understand the calculation, not only the final plot.**