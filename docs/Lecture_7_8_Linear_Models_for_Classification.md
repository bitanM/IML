# Lecture 7-8: Linear Models for Classification

**Instructor:** Dr. Anirban Sen  
**Date:** September 23, 2026  

---

## Table of Contents

1. [Introduction to Generative Classification](#1-introduction-to-generative-classification)
2. [Two-Class Case: From Bayes to Logistic Sigmoid](#2-two-class-case-from-bayes-to-logistic-sigmoid)
3. [Log-Odds Interpretation](#3-log-odds-interpretation)
4. [Decision Boundary in Two-Class Case](#4-decision-boundary-in-two-class-case)
5. [Extension to K Classes: Softmax Function](#5-extension-to-k-classes-softmax-function)
6. [Generative Models and Discriminant Functions](#6-generative-models-and-discriminant-functions)
7. [Continuous Inputs: Gaussian Class-Conditional Densities](#7-continuous-inputs-gaussian-class-conditional-densities)
8. [Two-Class Gaussian: Deriving the Linear Discriminant](#8-two-class-gaussian-deriving-the-linear-discriminant)
9. [General K-Class Gaussian Result](#9-general-k-class-gaussian-result)
10. [Quadratic Discriminant Analysis (Different Covariances)](#10-quadratic-discriminant-analysis-different-covariances)
11. [Maximum Likelihood Parameter Estimation](#11-maximum-likelihood-parameter-estimation)
12. [Singular Covariance Matrix: Issues and Solutions](#12-singular-covariance-matrix-issues-and-solutions)

---

## 1. Introduction to Generative Classification

### 1.1 Bayes' Theorem for Classification

For classes $C_1$ and $C_2$:

$$p(C_1 | \mathbf{x}) = \frac{p(\mathbf{x} | C_1)p(C_1)}{p(\mathbf{x} | C_1)p(C_1) + p(\mathbf{x} | C_2)p(C_2)}$$

### 1.2 Generative Approach

- **Model:** How data is generated ($p(\mathbf{x} | C_k)$ and $p(C_k)$)
- **Inference:** Apply Bayes' theorem to get posteriors $p(C_k | \mathbf{x})$
- **Advantage:** Can answer richer questions (likelihood, typicality, changing priors)

---

## 2. Two-Class Case: From Bayes to Logistic Sigmoid

### 2.1 Introducing the Odds

Divide numerator and denominator by $p(\mathbf{x} | C_1)p(C_1)$:

$$p(C_1 | \mathbf{x}) = \frac{1}{1 + \frac{p(\mathbf{x}|C_2)p(C_2)}{p(\mathbf{x}|C_1)p(C_1)}}$$

### 2.2 Define the Log-Odds

$$a = \ln \frac{p(\mathbf{x} | C_1)p(C_1)}{p(\mathbf{x} | C_2)p(C_2)}$$

Then:
$$\frac{p(\mathbf{x}|C_2)p(C_2)}{p(\mathbf{x}|C_1)p(C_1)} = e^{-a}$$

### 2.3 The Logistic Sigmoid Function

$$\sigma(a) = \frac{1}{1 + e^{-a}}$$

Therefore:
$$p(C_1 | \mathbf{x}) = \sigma(a)$$
where
$$a = \ln \frac{p(\mathbf{x} | C_1)p(C_1)}{p(\mathbf{x} | C_2)p(C_2)}$$

**Key Result:** The posterior probability in the two-class case is a logistic sigmoid of the log-odds.

### 2.4 Properties of the Sigmoid

- Maps $\mathbb{R} \to (0, 1)$
- $\sigma(0) = \frac{1}{2}$
- $a > 0 \implies p(C_1 | \mathbf{x}) > \frac{1}{2}$
- $a < 0 \implies p(C_1 | \mathbf{x}) < \frac{1}{2}$

---

## 3. Log-Odds Interpretation

### 3.1 Posterior Odds

For two classes:
$$p(C_2 | \mathbf{x}) = 1 - p(C_1 | \mathbf{x})$$

Posterior odds:
$$\frac{p(C_1 | \mathbf{x})}{p(C_2 | \mathbf{x})}$$

### 3.2 Log Posterior Odds

$$\ln \frac{p(C_1 | \mathbf{x})}{p(C_2 | \mathbf{x})} = a$$

**Interpretation:** $a$ is the **log posterior odds** — the logarithm of the ratio of posterior probabilities.

---

## 4. Decision Boundary in Two-Class Case

### 4.1 Classification Rule

Choose $C_1$ when $p(C_1 | \mathbf{x}) > p(C_2 | \mathbf{x})$, equivalent to $a > 0$.

### 4.2 Decision Boundary

Occurs at $a = 0$:
$$\ln \frac{p(\mathbf{x} | C_1)p(C_1)}{p(\mathbf{x} | C_2)p(C_2)} = 0$$

Equivalently:
$$p(\mathbf{x} | C_1)p(C_1) = p(\mathbf{x} | C_2)p(C_2)$$

### 4.3 Interpretation

At the boundary:
$$p(C_1 | \mathbf{x}) = p(C_2 | \mathbf{x}) = \frac{1}{2}$$

The classifier is **exactly indifferent** between the two classes.

---

## 5. Extension to K Classes: Softmax Function

### 5.1 Multiclass Bayes' Theorem

$$p(C_k | \mathbf{x}) = \frac{p(\mathbf{x} | C_k)p(C_k)}{\sum_{j=1}^K p(\mathbf{x} | C_j)p(C_j)}$$

### 5.2 Class Scores

Define:
$$a_k(\mathbf{x}) = \ln [p(\mathbf{x} | C_k)p(C_k)]$$

Then:
$$p(\mathbf{x} | C_k)p(C_k) = e^{a_k(\mathbf{x})}$$

### 5.3 Softmax Function

$$\text{softmax}(a_k) = \frac{e^{a_k}}{\sum_j e^{a_j}}$$

Therefore:
$$p(C_k | \mathbf{x}) = \text{softmax}_k(a_1(\mathbf{x}), \ldots, a_K(\mathbf{x}))$$

**Softmax is the multiclass analogue of the logistic sigmoid.**

### 5.4 Why "Softmax"?

Example: $a_1 = 2, a_2 = 1, a_3 = -1$

$$e^{a_1} \approx 7.39,\quad e^{a_2} \approx 2.72,\quad e^{a_3} \approx 0.37$$

$$p(C_1 | \mathbf{x}) \approx \frac{7.39}{7.39 + 2.72 + 0.37} \approx 0.704$$

The largest score receives the largest probability.

### 5.5 Softmax Properties

- $p(C_k | \mathbf{x}) > 0$ for all $k$
- $\sum_k p(C_k | \mathbf{x}) = 1$
- Forms a valid probability distribution

### 5.6 Classification Rule

$$\hat{C} = \arg\max_k p(C_k | \mathbf{x}) = \arg\max_k e^{a_k} = \arg\max_k a_k(\mathbf{x})$$

The denominator is common to all classes, and exponential is monotonic.

---

## 6. Generative Models and Discriminant Functions

### 6.1 Connection

$$a_k(\mathbf{x}) = \ln [p(\mathbf{x} | C_k)p(C_k)]$$

$$\hat{C} = \arg\max_k a_k(\mathbf{x})$$

A probabilistic generative model **automatically gives** a set of discriminant functions.

### 6.2 Conceptual Difference

| Approach | Description |
|----------|-------------|
| **Discriminant** | Model the decision function directly |
| **Generative** | Model how data arises, then derive decision function |

### 6.3 Training vs Prediction

**Training:** Training data $\to$ estimate parameters $\{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}, \pi_k\}$

**Prediction:** New $\mathbf{x} \to$ substitute $p(\mathbf{x} | C_k)$ and $p(C_k)$ into Bayes' theorem

**Pipeline:** Learn the distribution $\to$ infer the class

---

## 7. Continuous Inputs: Gaussian Class-Conditional Densities

### 7.1 Setup

For $\mathbf{x} \in \mathbb{R}^D$, assume:
$$p(\mathbf{x} | C_k) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma})$$

Explicitly:
$$p(\mathbf{x} | C_k) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left[-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)\right]$$

**Crucial Assumption:** All classes share the **same covariance matrix** $\boldsymbol{\Sigma}$.

---

## 8. Two-Class Gaussian: Deriving the Linear Discriminant

### 8.1 Starting Point

$$p(C_1 | \mathbf{x}) = \frac{p(\mathbf{x} | C_1)p(C_1)}{p(\mathbf{x} | C_1)p(C_1) + p(\mathbf{x} | C_2)p(C_2)} = \sigma(a(\mathbf{x}))$$

where
$$a(\mathbf{x}) = \ln \frac{p(\mathbf{x} | C_1)p(C_1)}{p(\mathbf{x} | C_2)p(C_2)}$$

### 8.2 Substitute Gaussian Densities

$$a(\mathbf{x}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_1) + \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_2) + \ln \frac{p(C_1)}{p(C_2)}$$

**Note:** Normalization constants cancel because $\boldsymbol{\Sigma}$ is shared.

### 8.3 Expand Quadratic Forms

$$(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - 2\boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} + \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$$

### 8.4 Quadratic Terms Cancel

$$\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - \frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = 0$$

**This cancellation happens because the covariance is shared.**

### 8.5 Collect Linear Terms

After cancellation:
$$a(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - \frac{1}{2}\boldsymbol{\mu}_1^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_2 + \ln \frac{p(C_1)}{p(C_2)}$$

Since $\boldsymbol{\Sigma}^{-1}$ is symmetric:
$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \mathbf{x}$$

### 8.6 Define Discriminant Parameters

$$\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$$

$$w_0 = -\frac{1}{2}\boldsymbol{\mu}_1^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_2 + \ln \frac{p(C_1)}{p(C_2)}$$

### 8.7 Final Result

$$p(C_1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + w_0)$$

**Decision boundary:** $\mathbf{w}^T \mathbf{x} + w_0 = 0$ (a hyperplane with normal $\mathbf{w}$)

**Key Result:** Shared-covariance Gaussian model $\implies$ Linear decision boundary.

---

## 9. General K-Class Gaussian Result

### 9.1 Expand Class Scores

$$a_k(\mathbf{x}) = \ln [p(\mathbf{x} | C_k)p(C_k)]$$

$$= -\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} + \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - \frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \ln p(C_k) + \text{constant}$$

### 9.2 Remove Common Term

The term $-\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x}$ is identical for all classes $\implies$ subtract it.

Define:
$$\mathbf{w}_k = \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k$$

$$w_{k0} = -\frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \ln p(C_k)$$

### 9.3 Linear Discriminant Functions

$$a_k'(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}$$

### 9.4 Softmax Posterior

$$p(C_k | \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + w_{k0})}{\sum_j \exp(\mathbf{w}_j^T \mathbf{x} + w_{j0})}$$

### 9.5 Pairwise Decision Boundaries

Boundary between $C_i$ and $C_j$:
$$p(C_i | \mathbf{x}) = p(C_j | \mathbf{x}) \implies \mathbf{w}_i^T \mathbf{x} + w_{i0} = \mathbf{w}_j^T \mathbf{x} + w_{j0}$$

$$(\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (w_{i0} - w_{j0}) = 0$$

**Every pairwise boundary is a hyperplane.**

---

## 10. Quadratic Discriminant Analysis (Different Covariances)

### 10.1 If $\boldsymbol{\Sigma}_1 \neq \boldsymbol{\Sigma}_2$

Quadratic terms:
$$-\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}_1^{-1} \mathbf{x} + \frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}_2^{-1} \mathbf{x} = \frac{1}{2}\mathbf{x}^T (\boldsymbol{\Sigma}_2^{-1} - \boldsymbol{\Sigma}_1^{-1}) \mathbf{x}$$

Generally $\neq 0$.

### 10.2 Result

| Covariance | Discriminant | Boundary |
|------------|--------------|----------|
| $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2$ | Linear | Linear (hyperplane) |
| $\boldsymbol{\Sigma}_1 \neq \boldsymbol{\Sigma}_2$ | Quadratic | Quadratic |

This is the key distinction between **LDA** (Linear Discriminant Analysis) and **QDA** (Quadratic Discriminant Analysis).

---

## 11. Maximum Likelihood Parameter Estimation

### 11.1 Training Data

$$\mathcal{D} = \{(\mathbf{x}_n, t_n)\}_{n=1}^N$$

where $t_n$ indicates the class of $\mathbf{x}_n$.

### 11.2 Parameters to Estimate

$$\boldsymbol{\theta} = \{\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K, \boldsymbol{\Sigma}, \pi_1, \ldots, \pi_K\}$$

where $\pi_k = p(C_k)$.

### 11.3 Maximum Likelihood Principle

$$\boldsymbol{\theta}_{ML} = \arg\max_{\boldsymbol{\theta}} p(\mathcal{D} | \boldsymbol{\theta})$$

Equivalently, maximize log-likelihood:
$$\boldsymbol{\theta}_{ML} = \arg\max_{\boldsymbol{\theta}} \ln p(\mathcal{D} | \boldsymbol{\theta})$$

### 11.4 Likelihood of Labeled Data

Assume conditional independence:
$$p(\mathcal{D} | \boldsymbol{\theta}) = \prod_{n=1}^N p(\mathbf{x}_n, t_n | \boldsymbol{\theta})$$

For $\mathbf{x}_n \in C_k$:
$$p(\mathbf{x}_n, t_n = C_k) = p(\mathbf{x}_n | C_k) \pi_k$$

Log-likelihood:
$$\ln p(\mathcal{D} | \boldsymbol{\theta}) = \sum_{n=1}^N [\ln p(\mathbf{x}_n | C_{t_n}) + \ln \pi_{t_n}]$$

### 11.5 Class Indicator Variables

Define:
$$t_{nk} = \begin{cases} 1, & \text{if } \mathbf{x}_n \in C_k \\ 0, & \text{otherwise} \end{cases}$$

Properties:
$$\sum_k t_{nk} = 1 \quad \text{(each point belongs to one class)}$$

Number of points in class $k$:
$$N_k = \sum_{n=1}^N t_{nk}, \quad \sum_k N_k = N$$

Compact log-likelihood:
$$\ln p(\mathcal{D} | \boldsymbol{\theta}) = \sum_{n=1}^N \sum_{k=1}^K t_{nk} [\ln p(\mathbf{x}_n | C_k) + \ln \pi_k]$$

---

## 11.6 MLE of Class Means

### Terms Involving $\boldsymbol{\mu}_k$

$$-\frac{1}{2} \sum_{n=1}^N t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)$$

Minimize:
$$E_k = \sum_n t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)$$

### Derivative

$$\frac{\partial E_k}{\partial \boldsymbol{\mu}_k} = -2\boldsymbol{\Sigma}^{-1} \sum_n t_{nk} \mathbf{x}_n + 2 N_k \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k$$

Set to zero:
$$\boldsymbol{\Sigma}^{-1} N_k \boldsymbol{\mu}_k = \boldsymbol{\Sigma}^{-1} \sum_n t_{nk} \mathbf{x}_n$$

Since $\boldsymbol{\Sigma}^{-1}$ is nonsingular:

$$\boldsymbol{\mu}_{k,ML} = \frac{1}{N_k} \sum_{n: t_{nk}=1} \mathbf{x}_n$$

**The MLE of the class mean is the sample mean of that class.**

---

## 11.7 MLE of Shared Covariance Matrix

### Covariance-Dependent Log-Likelihood

$$\ell(\boldsymbol{\Sigma}) = -\frac{N}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2} \text{Tr}(\boldsymbol{\Sigma}^{-1} \mathbf{S})$$

where scatter matrix:
$$\mathbf{S} = \sum_{n,k} t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

### Matrix Calculus Identities

1. $\frac{\partial \ln |\boldsymbol{\Sigma}|}{\partial \boldsymbol{\Sigma}} = \boldsymbol{\Sigma}^{-1}$
2. $\frac{\partial \text{Tr}(\boldsymbol{\Sigma}^{-1} \mathbf{S})}{\partial \boldsymbol{\Sigma}} = -\boldsymbol{\Sigma}^{-1} \mathbf{S} \boldsymbol{\Sigma}^{-1}$

### Derivative

$$\frac{\partial \ell}{\partial \boldsymbol{\Sigma}} = -\frac{N}{2} \boldsymbol{\Sigma}^{-1} + \frac{1}{2} \boldsymbol{\Sigma}^{-1} \mathbf{S} \boldsymbol{\Sigma}^{-1}$$

Set to zero and multiply by $\boldsymbol{\Sigma}$:
$$-\frac{N}{2} \boldsymbol{\Sigma} + \frac{1}{2} \mathbf{S} = 0 \implies \mathbf{S} = N \boldsymbol{\Sigma}$$

### MLE

$$\boldsymbol{\Sigma}_{ML} = \frac{1}{N} \sum_{k=1}^K \sum_{n=1}^N t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

**Interpretation:** Average of outer products of deviations from class means.

---

## 11.8 MLE of Class Priors

### Log-Likelihood for Priors

$$\ell(\boldsymbol{\pi}) = \sum_{k=1}^K N_k \ln \pi_k$$

Subject to: $\sum_k \pi_k = 1$

### Lagrange Multiplier

$$\mathcal{L} = \sum_k N_k \ln \pi_k + \lambda \left(\sum_k \pi_k - 1\right)$$

$$\frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{N_k}{\pi_k} + \lambda = 0 \implies \pi_k = -\frac{N_k}{\lambda}$$

Sum to 1:
$$\sum_k \pi_k = -\frac{1}{\lambda} \sum_k N_k = -\frac{N}{\lambda} = 1 \implies \lambda = -N$$

### MLE

$$\pi_{k,ML} = \frac{N_k}{N}$$

**The MLE of a class prior is its observed fraction in the training data.**

---

## 11.9 Summary of MLE Estimates

| Parameter | MLE Estimate |
|-----------|--------------|
| **Mean** $\boldsymbol{\mu}_k$ | $\displaystyle \frac{1}{N_k} \sum_{n: t_{nk}=1} \mathbf{x}_n$ |
| **Covariance** $\boldsymbol{\Sigma}$ | $\displaystyle \frac{1}{N} \sum_{n,k} t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$ |
| **Prior** $\pi_k$ | $\displaystyle \frac{N_k}{N}$ |

---

## 11.10 Complete Pipeline

1. **Learn:** Estimate $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}, \pi_k$ from training data
2. **Compute:** $p(\mathbf{x} | C_k) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma})$
3. **Posterior:** $p(C_k | \mathbf{x}) = \frac{p(\mathbf{x} | C_k) \pi_k}{\sum_j p(\mathbf{x} | C_j) \pi_j}$
4. **Classify:** $\hat{C} = \arg\max_k p(C_k | \mathbf{x})$

Because $\boldsymbol{\Sigma}$ is shared, posteriors form softmax of linear discriminants.

---

## 12. Singular Covariance Matrix: Issues and Solutions

### 12.1 Problem

Gaussian density requires:
$$|\boldsymbol{\Sigma}| > 0 \quad \text{and} \quad \boldsymbol{\Sigma}^{-1} \text{ exists}$$

If $|\boldsymbol{\Sigma}| = 0$, then at least one eigenvalue $\lambda_i = 0$.

### 12.2 Geometric Interpretation

Zero variance in at least one direction $\implies$ Gaussian ellipsoid collapses to lower-dimensional object.

Example: 3D data lying entirely on a plane $\implies$ no spread perpendicular to plane.

### 12.3 Solutions

| Method | Description |
|--------|-------------|
| **Regularization** | $\boldsymbol{\Sigma}_{reg} = \boldsymbol{\Sigma} + \lambda \mathbf{I},\ \lambda > 0$ |
| **Remove Redundant Features** | Eliminate perfectly/nearly linearly dependent features |
| **Dimensionality Reduction** | Use PCA to work in lower-dimensional space |
| **Pseudoinverse** | Use $\boldsymbol{\Sigma}^+$ (Moore-Penrose) with degenerate Gaussian |

---

## Problem Sets and Examples

### Problem Set 1: Two-Class Logistic Regression

**Problem 1.1:** Derive the decision boundary for a two-class problem where $p(C_1) = 0.7$, $p(C_2) = 0.3$, and class-conditional densities are $\mathcal{N}(\mu_1, \sigma^2)$ and $\mathcal{N}(\mu_2, \sigma^2)$.

*Hint: Use $a = \ln \frac{p(\mathbf{x}|C_1)p(C_1)}{p(\mathbf{x}|C_2)p(C_2)}$ and set $a = 0$.*

**Problem 1.2:** Show that for the two-class case, $\frac{\partial \sigma(a)}{\partial a} = \sigma(a)(1 - \sigma(a))$.

**Problem 1.3:** If $\mathbf{w} = [2, -1]^T$ and $w_0 = 0.5$, classify the point $\mathbf{x} = [1, 2]^T$. What is $p(C_1 | \mathbf{x})$?

---

### Problem Set 2: Softmax and Multiclass

**Problem 2.1:** For $K=3$ classes with scores $a_1 = 1, a_2 = 0, a_3 = -1$, compute $p(C_1 | \mathbf{x}), p(C_2 | \mathbf{x}), p(C_3 | \mathbf{x})$.

**Problem 2.2:** Prove that adding a constant $c$ to all scores $a_k$ does not change the softmax probabilities.

**Problem 2.3:** In a 3-class problem, the discriminant functions are:
- $a_1(\mathbf{x}) = 2x_1 + x_2 - 1$
- $a_2(\mathbf{x}) = -x_1 + 3x_2 + 2$
- $a_3(\mathbf{x}) = x_1 - 2x_2$

Find the pairwise decision boundaries between each pair of classes.

---

### Problem Set 3: Gaussian Generative Models

**Problem 3.1:** Given two classes with shared covariance:
- $\boldsymbol{\mu}_1 = [0, 0]^T$, $\boldsymbol{\mu}_2 = [2, 2]^T$
- $\boldsymbol{\Sigma} = \begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix}$
- $p(C_1) = p(C_2) = 0.5$

Derive $\mathbf{w}$ and $w_0$ for the linear discriminant. Plot the decision boundary.

**Problem 3.2:** Suppose class $C_1$ has covariance $\boldsymbol{\Sigma}_1 = \mathbf{I}$ and $C_2$ has $\boldsymbol{\Sigma}_2 = 2\mathbf{I}$. Write the quadratic discriminant function $a_1(\mathbf{x}) - a_2(\mathbf{x})$.

**Problem 3.3:** For a dataset with $N=100$ points, $N_1 = 60$, $N_2 = 40$, and sample means $\hat{\boldsymbol{\mu}}_1 = [1, 2]^T$, $\hat{\boldsymbol{\mu}}_2 = [3, 1]^T$, compute the MLE estimates of $\pi_1, \pi_2, \boldsymbol{\mu}_1, \boldsymbol{\mu}_2$ and $\boldsymbol{\Sigma}$ (assume shared).

---

### Problem Set 4: Maximum Likelihood Derivations

**Problem 4.1:** Derive the MLE of the mean $\boldsymbol{\mu}_k$ without using the indicator $t_{nk}$, directly from:
$$\ln p(\mathcal{D} | \boldsymbol{\theta}) = \sum_{n: t_n = k} \ln p(\mathbf{x}_n | C_k) + \text{const}$$
Show it equals the sample mean.

**Problem 4.2:** Verify the matrix identity: $\frac{\partial \ln |\boldsymbol{\Sigma}|}{\partial \boldsymbol{\Sigma}} = \boldsymbol{\Sigma}^{-1}$ for a $2 \times 2$ matrix.

**Problem 4.3:** The scatter matrix is defined as $\mathbf{S} = \sum_{n,k} t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$. Show that $\text{Tr}(\boldsymbol{\Sigma}^{-1} \mathbf{S}) = \sum_{n,k} t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k)$.

---

### Problem Set 5: Edge Cases and Practical Issues

**Problem 5.1:** In a 2D problem, all data points for class $C_1$ lie exactly on the line $x_2 = 2x_1$. What is the rank of the sample covariance matrix? What happens when you try to compute the Gaussian density?

**Problem 5.2:** Given a singular covariance matrix $\boldsymbol{\Sigma} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ (rank 1), apply regularization with $\lambda = 0.1$ and verify the resulting matrix is invertible.

**Problem 5.3:** Consider a 3-class problem where one class has only 1 training sample ($N_3 = 1$). What is the MLE of $\boldsymbol{\mu}_3$? What is the MLE of $\boldsymbol{\Sigma}$? Discuss the issue and suggest a solution.

---

## Key Takeaways Summary

1. **Generative classification** models $p(\mathbf{x}, C_k)$ and uses Bayes' theorem
2. **Two classes** $\to$ Logistic sigmoid $\sigma(a)$ where $a = \text{log posterior odds}$
3. **K classes** $\to$ Softmax of linear discriminants $a_k(\mathbf{x})$
4. **Shared Gaussian covariance** $\to$ Linear decision boundaries (LDA)
5. **Different covariances** $\to$ Quadratic boundaries (QDA)
6. **MLE estimates** have simple closed forms: sample means, average scatter, class frequencies
7. **Singular covariance** requires regularization, feature removal, or dimensionality reduction

---

*End of Lecture Notes*