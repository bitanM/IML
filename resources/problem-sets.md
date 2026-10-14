---
layout: default
title: Problem Sets Index
nav_order: 3
permalink: /resources/problem-sets/
---

# Problem Sets Index

This page provides a consolidated index of all problem sets across the lecture notes.

---

## Lecture 2: Convex Functions & Optimization

### Problem Set 1: Definitions and Basic Properties
- [1.1] Sum of convex functions / Product of convex functions
- [1.2] Affine transformation of convex functions
- [1.3] Convexity of $f(x) = |x|$ using definition
- [1.4] Jensen's inequality for 3 points / generalization to n points

### Problem Set 2: Second Derivative and Hessian Tests
- [2.1] 1D convexity checks: $x^4$, $\ln(x)$, $e^x + e^{-x}$, $\sin(x)$, $-\sqrt{x}$
- [2.2] 2D Hessian: $f(x_1, x_2) = x_1^2 + 2x_1x_2 + 3x_2^2$
- [2.3] Saddle point analysis: $f(x_1, x_2) = x_1^2 - x_2^2$
- [2.4] Quadratic forms: $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$

### Problem Set 3: $z^T H z$ Condition
- [3.1] $f(x_1, x_2) = x_1^4 + x_2^4$ - position-dependent Hessian
- [3.2] $f(x_1, x_2) = (x_1 - x_2)^2$ - null directions
- [3.3] Quadratic function properties
- [3.4] Condition number analysis

### Problem Set 4: Convexity vs Minima
- [4.1] Convexity + minimum existence for 5 functions
- [4.2] Strictly convex ⇒ at most one minimum
- [4.3] Examples of convex functions with various minimum properties
- [4.4] $f(x) = -\log(x)$ analysis

### Problem Set 5: Non-Convex Functions with Minima
- [5.1] $f(x) = x^4 - x^2$ complete analysis
- [5.2] $f(x) = \sin(x) + 0.1 x^2$
- [5.3] Construct function with 3 local minima
- [5.4] Piecewise quadratic convexity

### Problem Set 6: Operations Preserving Convexity
- [6.1] Nonnegative weighted sum
- [6.2] Pointwise maximum
- [6.3] Affine transformation
- [6.4] Composition with nondecreasing convex function

### Problem Set 7: Worked Examples
- [7.1] Quadratic form convexity check
- [7.2] Logistic loss convexity
- [7.3] Least squares objective
- [7.4] Entropy function

### Problem Set 8: Advanced Topics
- [8.1] Strong convexity
- [8.2] Convex conjugate
- [8.3] Bregman divergence
- [8.4] Gradient descent convergence

---

## Lecture 3-5: Linear Regression (Probabilistic View)

### Problem Set 1: Linear Regression Basics
- [1.1] Basis functions: cubic, Fourier, Gaussian
- [1.2] Explicit form and evaluation
- [1.3] Linearity in parameters vs input

### Problem Set 2: Gaussian Noise & Likelihood
- [2.1] Single observation likelihood
- [2.2] Log-likelihood derivation steps
- [2.3] Behavior as $\beta \to \infty, 0$

### Problem Set 3: Least Squares & Normal Equations
- [3.1] Numerical computation with given design matrix
- [3.2] Scalar derivation of normal equations
- [3.3] Singularity of $\Phi^T\Phi$ and regularization
- [3.4] $\beta_{ML}$ computation

### Problem Set 4: Assumptions & Diagnostics
- [4.1] Residual pattern identification
- [4.2] Heteroscedasticity simulation
- [4.3] Durbin-Watson interpretation
- [4.4] Outlier vs leverage vs influence

### Problem Set 5: Extensions
- [5.1] Bias-variance tradeoff in polynomial regression
- [5.2] Ridge regression from Gaussian prior
- [5.3] MLE vs MAP comparison
- [5.4] Bayesian linear regression posterior
- [5.5] Predictive distribution

### Problem Set 6: Worked Examples
- [6.1] Simple linear regression by hand
- [6.2] Polynomial regression comparison
- [6.3] Assumption checking

---

## Lecture 6-7: Linear Models for Classification (Discriminant Functions)

### Problem Set 1: Binary Linear Discriminants
- [1.1] Decision boundary, normal vector, distance, classification
- [1.2] 3D hyperplane geometry
- [1.3] Signed distance derivation
- [1.4] Point positions and perpendicular distance

### Problem Set 2: Multiclass Discriminants
- [2.1] Classification with 3 discriminants
- [2.2] Pairwise boundary derivation
- [2.3] Algebraic dependence verification
- [2.4] Number of boundaries for K classes

### Problem Set 3: OvO vs K-Class
- [3.1] Ambiguous regions in OvO
- [3.2] OvR ambiguous regions
- [3.3] Parameter count comparison
- [3.4] Imbalanced data effects

### Problem Set 4: Geometry
- [4.1] 3D intersection line
- [4.2] Decision boundary check
- [4.3] Boundary consistency check
- [4.4] Convexity of decision regions

### Problem Set 5: Least Squares for Classification
- [5.1] Squared error limitations
- [5.2] Least squares approximates class probability
- [5.3] Limitations of least squares
- [5.4] One-hot encoding problems

---

## Lecture 7-8: Linear Models for Classification (Generative Models)

### Problem Set 1: Two-Class Logistic Regression
- [1.1] Decision boundary with priors
- [1.2] Sigmoid derivative
- [1.3] Classification and probability computation

### Problem Set 2: Softmax & Multiclass
- [2.1] Softmax computation for 3 classes
- [2.2] Translation invariance of softmax
- [2.3] 3-class boundary equations
- [2.4] Softmax temperature effects

### Problem Set 3: Gaussian Generative Models
- [3.1] LDA derivation for 2D case
- [3.2] QDA with different covariances
- [3.3] MLE computation for given data

### Problem Set 4: MLE Derivations
- [4.1] Mean MLE without indicator variables
- [4.2] 2x2 log-determinant derivative
- [4.3] Trace identity verification

### Problem Set 5: Edge Cases
- [5.1] Singular covariance from collinear data
- [5.2] Regularization of singular matrix
- [5.3] Single-sample class issues

---

## Lecture 9: Logistic Regression

### Problem Set 1: Logistic Regression Fundamentals
- [1.1] Sigmoid properties and interpretation
- [1.2] Log-likelihood derivation
- [1.3] Gradient and Hessian

### Problem Set 2: Optimization
- [2.1] Gradient descent derivation
- [2.2] Newton's method / IRLS
- [2.3] Convergence analysis

### Problem Set 3: Regularization
- [3.1] L2 regularization effect
- [3.2] L1 regularization (sparsity)
- [3.3] Regularization path

### Problem Set 4: Multiclass
- [4.1] Softmax regression
- [4.2] One-vs-rest comparison
- [4.3] Hierarchical softmax

---

## Lecture 11: Support Vector Machines

### Problem Set 1: Basic SVM Concepts
- [1.1] Hyperplane parameters and classification
- [1.2] Correct classification equivalence
- [1.3] Linear separability check
- [1.4] Label encoding rationale

### Problem Set 2: Margin & Distance
- [2.1] Distance computation with normalization
- [2.2] Scaling invariance proof
- [2.3] Functional vs geometric margin
- [2.4] General distance derivation

### Problem Set 3: Hard-SVM Formulation
- [3.1] Max-margin to min-norm conversion
- [3.2] Convexity, Lagrangian, KKT
- [3.3] Support vector count
- [3.4] $\frac{1}{2}$ factor rationale

### Problem Set 4: Soft-SVM & Slack
- [4.1] Slack variable computation
- [4.2] Constrained ↔ unconstrained equivalence
- [4.3] Outlier handling with $\lambda$
- [4.4] $C$ vs $\lambda$ relationship

### Problem Set 5: Hinge Loss
- [5.1] Loss function comparison
- [5.2] Convexity and non-differentiability
- [5.3] Single example limits
- [5.4] Hinge bounds zero-one loss

### Problem Set 6: Support Vectors
- [6.1] Support vector definition and $\mathbf{w}$ representation
- [6.2] Minimum/maximum support vectors
- [6.3] Soft-SVM support vector types
- [6.4] Non-separable bullseye dataset

### Problem Set 7: Dual Formulation
- [7.1] Hard-SVM dual derivation
- [7.2] Kernel trick explanation
- [7.3] Soft-SVM box constraints
- [7.4] Sparsity and prediction

### Problem Set 8: Worked Examples
- [8.1] Hard-SVM in 2D
- [8.2] Soft-SVM with outlier
- [8.3] Hinge loss computation
- [8.4] Regularization path

### Problem Set 9: Generalization Theory
- [9.1] VC dimension and margin
- [9.2] Margin bound statement
- [9.3] Margin and generalization
- [9.4] SVM vs Logistic Regression

### Problem Set 10: Extensions
- [10.1] Kernel SVM for XOR
- [10.2] Multi-class SVM approaches
- [10.3] $\nu$-SVM parameters
- [10.4] Structured SVM hinge loss

---

## Fisher's LDA Assignment

### Mathematical Tasks
- [T1] Data loading, inspection, class means
- [T2] Two-species Fisher reconstruction
- [T3] Within-class scatter matrix
- [T4] Between-class scatter matrix
- [T5] Generalized eigenvalue problem
- [T6] 2D projection and visualization
- [T7] Comparison with Fisher (1936)
- [T8] Interpretation questions

### Viva Extensions (Optional)
- [V1] Singular/ill-conditioned $S_W$
- [V2] Feature standardization effects
- [V3] Unequal class sizes
- [V4] Two-class $S_W^{-1}d$ connection
- [V5] Verification without sklearn
- [V6] Eigenvector sign/scale ambiguity
- [V7] Information loss in projection
- [V8] Classification in discriminant space
- [V9] Robustness to resampling
- [V10] Numerical issues with $S_W^{-1}S_B$

---

## How to Use Problem Sets

1. **Sequential Practice**: Work through problems in order as you study each lecture
2. **Targeted Review**: Jump to specific problem sets for exam preparation
3. **Deep Dive**: The "Worked Examples" and "Advanced Topics" sets provide deeper understanding
4. **Assignment Prep**: Fisher's LDA problems directly prepare you for the assignment and viva

---

## Difficulty Levels

| Symbol | Meaning |
|--------|---------|
| 🟢 | Basic - Direct application of definitions |
| 🟡 | Intermediate - Requires combining concepts |
| 🔴 | Advanced - Proofs, derivations, extensions |
| ⚫ | Research-level - Open-ended exploration |

Most problem sets progress from 🟢 to 🔴 within each set.