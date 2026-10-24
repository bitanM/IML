---
layout: default
title: "Lecture 11: Support Vector Machines"
nav_order: 6
parent: Lectures
has_toc: true
permalink: /lectures/Lecture_11_SVM_Support_Vector_Machines/
---

# Lecture 11: Support Vector Machines – Margin and Hard/Soft SVM

**Instructor:** Dr. Anirban Sen  
**Date:** October 12, 2026  

---

## Table of Contents

1. [What is an SVM?](#1-what-is-an-svm)
2. [Why Large Margin?](#2-why-large-margin)
3. [Setting Up the Classification Problem](#3-setting-up-the-classification-problem)
4. [Distance from Point to Hyperplane](#4-distance-from-point-to-hyperplane)
5. [Hard-SVM: Maximizing the Margin](#5-hard-svm-maximizing-the-margin)
6. [Scaling Problem and Its Resolution](#6-scaling-problem-and-its-resolution)
7. [From Maximum Margin to Minimum Norm](#7-from-maximum-margin-to-minimum-norm)
8. [Hard-SVM Final Formulation](#8-hard-svm-final-formulation)
9. [From Hard Margin to Soft Margin](#9-from-hard-margin-to-soft-margin)
10. [Slack Variables and Hinge Loss](#10-slack-variables-and-hinge-loss)
11. [Soft-SVM as Regularized Loss Minimization](#11-soft-svm-as-regularized-loss-minimization)
12. [Hard-SVM vs Soft-SVM Comparison](#12-hard-svm-vs-soft-svm-comparison)
13. [The Complete Conceptual Chain](#13-the-complete-conceptual-chain)
14. [Problem Sets and Examples](#14-problem-sets-and-examples)

---

## 1. What is an SVM?

### 1.1 Basic Idea

An **SVM (Support Vector Machine)** is a method for learning a linear classifier by looking for a **large-margin separator**.

### 1.2 Linear Classifier

A linear classifier is defined by:
$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

**Decision boundary** (hyperplane):
$$\mathbf{w}^T \mathbf{x} + b = 0$$

**Prediction:**
$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$

### 1.3 SVM Question

> Among all separating hyperplanes, **which one has the largest margin?**

---

## 2. Why Large Margin?

### 2.1 The Problem

Multiple hyperplanes can separate the training data:

```
       +1
      ╱ separator 2
     ╱
    ╱  separator 1
   ╱
──╱──────────────── x₁
 ╱  -1
╱
```

Both separate correctly, but which should we prefer?

### 2.2 Margin Definition

**Margin** = minimum distance between any training point and the separating hyperplane.

### 2.3 Intuition

- **Large margin** ⇒ points are comfortably away from the boundary
- A point close to the boundary can change classification after a small perturbation
- With large margin, the classifier tolerates small input changes
- Also motivates better **generalization**

### 2.4 SVM Principle

> Among separating hyperplanes, prefer one that leaves the **largest possible margin**.

---

## 3. Setting Up the Classification Problem

### 3.1 Training Set

$$\mathcal{S} = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_m, y_m)\}$$
- $\mathbf{x}_i \in \mathbb{R}^d$
- $y_i \in \{-1, +1\}$

### 3.2 Linear Classifier

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

**Decision boundary:** $\mathbf{w}^T \mathbf{x} + b = 0$

**Prediction:** $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$

### 3.3 Why Labels $\{-1, +1\}$?

They allow the classification condition for **both classes** in one expression:

$$y_i (\mathbf{w}^T \mathbf{x}_i + b) > 0$$

### 3.4 Proof: $y_i (\mathbf{w}^T \mathbf{x}_i + b) > 0 \iff$ Correct Classification

| Case | Correct Classification Requires | Multiply by $y_i$ |
|------|--------------------------------|-------------------|
| $y_i = +1$ | $\mathbf{w}^T \mathbf{x}_i + b > 0$ | $y_i(\mathbf{w}^T \mathbf{x}_i + b) > 0$ |
| $y_i = -1$ | $\mathbf{w}^T \mathbf{x}_i + b < 0$ | $-(\mathbf{w}^T \mathbf{x}_i + b) > 0 \iff y_i(\mathbf{w}^T \mathbf{x}_i + b) > 0$ |

### 3.5 Linear Separability

A training set is **linearly separable** if $\exists (\mathbf{w}, b)$ such that:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) > 0 \quad \forall i$$

Geometrically: there exists a hyperplane with all $+1$ on one side, all $-1$ on the other.

---

## 4. Distance from Point to Hyperplane

### 4.1 Hyperplane

$$\mathbf{w}^T \mathbf{x} + b = 0$$

The vector $\mathbf{w}$ is **normal** to the hyperplane.

### 4.2 Distance Formula

If $\|\mathbf{w}\| = 1$, then distance of point $\mathbf{x}$ from hyperplane:
$$\text{dist}(\mathbf{x}, \text{hyperplane}) = |\mathbf{w}^T \mathbf{x} + b|$$

For a training set, distance of closest point:
$$\min_i |\mathbf{w}^T \mathbf{x}_i + b|$$

### 4.3 Meaningful Interpretation

If we **require** $\|\mathbf{w}\| = 1$, the objective has a direct geometric meaning: it **is** the actual margin.

---

## 5. Hard-SVM: Maximizing the Margin

### 5.1 Optimization Problem

For separating hyperplane with $\|\mathbf{w}\| = 1$, margin is $\min_i |\mathbf{w}^T \mathbf{x}_i + b|$.

**Hard-SVM solves:**
$$\arg\max_{(\mathbf{w},b): \|\mathbf{w}\|=1} \min_i |\mathbf{w}^T \mathbf{x}_i + b|$$
subject to correct classification:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) > 0 \quad \forall i$$

### 5.2 Removing the Absolute Value

For correctly classified points:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) > 0$$

- If $y_i = +1$: $\mathbf{w}^T \mathbf{x}_i + b > 0$
- If $y_i = -1$: $\mathbf{w}^T \mathbf{x}_i + b < 0$

Thus:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) = |\mathbf{w}^T \mathbf{x}_i + b|$$

**Hard-SVM becomes:**
$$\arg\max_{(\mathbf{w},b): \|\mathbf{w}\|=1} \min_i y_i (\mathbf{w}^T \mathbf{x}_i + b)$$

---

## 6. Scaling Problem and Its Resolution

### 6.1 The Scaling Problem

Consider hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$.

Multiply both $\mathbf{w}$ and $b$ by $c > 0$:
$$(c\mathbf{w})^T \mathbf{x} + cb = 0$$

This is the **same hyperplane**, but:
$$y_i ((c\mathbf{w})^T \mathbf{x} + cb) = c \, y_i (\mathbf{w}^T \mathbf{x} + b)$$

The numerical value of the objective can be made **arbitrarily large** by scaling $(\mathbf{w}, b)$.

### 6.2 Solution

Fix the scale by imposing:
$$\|\mathbf{w}\| = 1$$

Now the objective has a meaningful geometric interpretation: **it is the actual margin**.

---

## 7. From Maximum Margin to Minimum Norm

### 7.1 Let Optimal Margin Be $\gamma^*$

$$\gamma^* = \min_i y_i ((\mathbf{w}^*)^T \mathbf{x}_i + b^*)$$
where $\|\mathbf{w}^*\| = 1$.

Since $\gamma^*$ is the largest possible margin:
$$y_i ((\mathbf{w}^*)^T \mathbf{x}_i + b^*) \geq \gamma^* \quad \forall i$$

### 7.2 Rescale

Define:
$$\tilde{\mathbf{w}} = \frac{\mathbf{w}^*}{\gamma^*}, \quad \tilde{b} = \frac{b^*}{\gamma^*}$$

Then:
$$y_i (\tilde{\mathbf{w}}^T \mathbf{x}_i + \tilde{b}) = \frac{1}{\gamma^*} y_i ((\mathbf{w}^*)^T \mathbf{x}_i + b^*) \geq 1$$

Therefore:
$$y_i (\tilde{\mathbf{w}}^T \mathbf{x}_i + \tilde{b}) \geq 1 \quad \forall i$$

### 7.3 Why Minimize $\|\mathbf{w}\|$?

Since $\|\mathbf{w}^*\| = 1$:
$$\|\tilde{\mathbf{w}}\| = \left\|\frac{\mathbf{w}^*}{\gamma^*}\right\| = \frac{1}{\gamma^*}$$

Thus:
$$\text{larger margin } \gamma^* \iff \text{smaller } \|\mathbf{w}\|$$

### 7.4 Equivalent Formulation

Instead of: **maximize margin** (with $\|\mathbf{w}\|=1$)

We can equivalently: **fix the margin to 1** and **minimize $\|\mathbf{w}\|$**

---

## 8. Hard-SVM Final Formulation

### 8.1 Quadratic Program

$$\min_{\mathbf{w},b} \|\mathbf{w}\|^2$$

subject to:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i$$

### 8.2 Core SVM Intuition

> **Maximize geometric margin $\iff$ Minimize $\|\mathbf{w}\|$ after fixing the scale.**

### 8.3 Final Output

Let $(\mathbf{w}_0, b_0) = \arg\min \|\mathbf{w}\|^2$ subject to constraints.

Then output:
$$\hat{\mathbf{w}} = \frac{\mathbf{w}_0}{\|\mathbf{w}_0\|}, \quad \hat{b} = \frac{b_0}{\|\mathbf{w}_0\|}$$

### 8.4 Why Dual Formulation?

The inner $\min_i$ makes the objective **non-differentiable**. Ordinary differentiation doesn't directly give a convenient optimization problem. The **dual formulation** resolves this.

---

## 9. From Hard Margin to Soft Margin

### 9.1 Hard-SVM Assumption

Hard-SVM assumes **linearly separable** data.

Constraints:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i$$

### 9.2 Real Data Problems

Real data may contain:
- Overlapping classes
- Noisy observations
- Mislabeled examples
- Examples that lie inside the margin

### 9.3 Soft-SVM Idea

> Instead of requiring every example to satisfy the margin constraint, **allow some violations, but penalize them.**

---

## 10. Slack Variables and Hinge Loss

### 10.1 Relaxing the Hard Constraint

Hard-SVM: $y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1$

Introduce **slack variable** $\xi_i \geq 0$:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$$

Rearranging:
$$\xi_i \geq 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)$$

### 10.2 Interpretation of $\xi_i$

| $\xi_i$ | Meaning |
|---------|---------|
| $\xi_i = 0$ | No violation (point satisfies margin) |
| $\xi_i > 0$ | Margin constraint is violated |

### 10.3 What $\xi_i$ Measures

Recall: $y_i (\mathbf{w}^T \mathbf{x}_i + b)$ tells where the example lies relative to classifier.

| $y_i (\mathbf{w}^T \mathbf{x}_i + b)$ | $\xi_i$ | Interpretation |
|----------------------------------------|---------|----------------|
| $\geq 1$ | 0 | Correct, outside margin |
| $0.5$ | $0.5$ | Correct, inside margin |
| $0$ | $1$ | On decision boundary |
| $-2$ | $3$ | Misclassified |
| $-7$ | $8$ | Strongly misclassified |

### 10.4 Smallest Feasible Slack

$$\xi_i = \max\{0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\}$$

There is **no upper bound** on $\xi_i$.

---

## 10.5 Hinge Loss

Define $z = y_i (\mathbf{w}^T \mathbf{x}_i + b)$.

**Hinge loss:**
$$\ell_{\text{hinge}}(z) = \max(0, 1 - z) = \begin{cases}
1 - z, & z < 1 \\
0, & z \geq 1
\end{cases}$$

### Properties

- **Zero** once the example satisfies the desired margin ($z \geq 1$)
- **Linear increase** for violations ($z < 1$)
- Larger violation $\Rightarrow$ larger hinge loss

### 10.6 Examples

| $z = y_i(\mathbf{w}^T \mathbf{x}_i + b)$ | Hinge Loss | Interpretation |
|------------------------------------------|------------|----------------|
| $2$ | $0$ | Correct, outside margin |
| $1$ | $0$ | Exactly at margin |
| $0.5$ | $0.5$ | Correct, inside margin |
| $0$ | $1$ | On decision boundary |
| $-2$ | $3$ | Misclassified |
| $-7$ | $8$ | Strongly misclassified |

---

## 11. Soft-SVM as Regularized Loss Minimization

### 11.1 Two Competing Objectives

1. **Large margin** $\iff$ small $\|\mathbf{w}\|$
2. **Small constraint violations** $\iff$ small $\xi_i$

### 11.2 Soft-SVM Optimization Problem

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \left[ \lambda \|\mathbf{w}\|^2 + \frac{1}{m} \sum_{i=1}^m \xi_i \right]$$

subject to:
$$y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Parameter:** $\lambda > 0$ controls the trade-off.

### 11.3 What Does $\lambda$ Do?

Objective: $\lambda \|\mathbf{w}\|^2 + \frac{1}{m} \sum_i \xi_i$

| Term | Preference |
|------|------------|
| $\lambda \|\mathbf{w}\|^2$ | Large margin |
| $\frac{1}{m} \sum_i \xi_i$ | Fewer violations |

- **Large $\lambda$** $\Rightarrow$ strong preference for large margin
- **Small $\lambda$** $\Rightarrow$ more tolerant of violations

### 11.4 Why Allow Violations?

**Outlier example:** A hard-margin classifier may move the decision boundary substantially just to classify one unusual point. Soft-SVM says: "I will tolerate this violation, but I will pay a penalty for it."

### 11.5 Eliminating Slack Variables

At optimum:
$$\xi_i = \max\{0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\} = \ell_{\text{hinge}}(y_i (\mathbf{w}^T \mathbf{x}_i + b))$$

Thus:
$$\frac{1}{m} \sum_{i=1}^m \xi_i = \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\}$$

**Soft-SVM without slack variables:**
$$\min_{\mathbf{w}, b} \left[ \lambda \|\mathbf{w}\|^2 + \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\} \right]$$

---

## 12. The Big Picture

### 12.1 Soft-SVM = Regularized Loss Minimization

Objective:
$$\lambda \|\mathbf{w}\|^2 + L_{\text{hinge}, \mathcal{S}}(\mathbf{w}, b)$$

where:
$$L_{\text{hinge}, \mathcal{S}}(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\}$$

| Term | Name | Effect |
|------|------|--------|
| $\lambda \|\mathbf{w}\|^2$ | **Regularization** | Prefer small $\|\mathbf{w}\| \iff$ large margin |
| $L_{\text{hinge}}$ | **Average hinge loss** | Penalize margin violations |

### 12.2 General Form

$$\text{loss} + \lambda \cdot \text{complexity penalty}$$

Here:
- Loss = Hinge loss
- Complexity penalty = $\|\mathbf{w}\|^2$

### 12.3 Final Formulation

$$\min_{\mathbf{w}, b} \left[ \lambda \|\mathbf{w}\|^2 + L_{\text{hinge}, \mathcal{S}}(\mathbf{w}, b) \right]$$

> **Large margin when possible; tolerate violations when necessary.**

---

## 13. Hard-SVM vs Soft-SVM Comparison

| Aspect | Hard-SVM | Soft-SVM |
|--------|----------|----------|
| **Assumption** | Data linearly separable | Does not require separability |
| **Objective** | $\min \|\mathbf{w}\|^2$ | $\min \left[ \lambda \|\mathbf{w}\|^2 + \frac{1}{m} \sum \xi_i \right]$ |
| **Constraints** | $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ | $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \xi_i \geq 0$ |
| **Violations** | Not allowed | Allowed, but penalized |
| **Real-world use** | Rarely (data rarely perfectly separable) | Standard choice |

---

## 14. The Complete Conceptual Chain

```
Hard-SVM: maximize margin
    ↓
minimize ‖w‖² subject to yᵢ(wᵀxᵢ + b) ≥ 1
    ↓
When data not perfectly separable:
    ↓
Allow constraint violations
    ↓
Introduce slack ξᵢ
    ↓
Penalize ∑ξᵢ
    ↓
ξᵢ = max{0, 1 - yᵢ(wᵀxᵢ + b)}
    ↓
Hinge loss
    ↓
Soft-SVM = regularization + hinge loss
    ↓
min λ‖w‖² + L_hinge(w, b)
```

---

## 15. Problem Sets and Examples

### Problem Set 1: Basic SVM Concepts

**Problem 1.1:** For a linear classifier $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$:
- (a) If $\mathbf{w} = [2, -1]^T$ and $b = 1$, what is the decision boundary equation?
- (b) What is the normal vector to this hyperplane?
- (c) Classify points $\mathbf{x}_1 = [1, 1]^T$, $\mathbf{x}_2 = [0, 2]^T$, $\mathbf{x}_3 = [2, 0]^T$

**Problem 1.2:** Prove that $y_i(\mathbf{w}^T\mathbf{x}_i + b) > 0$ is equivalent to correct classification for $y_i \in \{-1, +1\}$.

**Problem 1.3:** Given three points: $\mathbf{x}_1 = [1, 2]^T$ ($y=+1$), $\mathbf{x}_2 = [2, 1]^T$ ($y=+1$), $\mathbf{x}_3 = [0, 0]^T$ ($y=-1$).
- (a) Is this dataset linearly separable?
- (b) If yes, find one separating hyperplane (any $\mathbf{w}, b$ that works).

**Problem 1.4:** Why do we use labels $\{-1, +1\}$ instead of $\{0, 1\}$ in SVM? What would change?

---

### Problem Set 2: Margin and Distance

**Problem 2.1:** For hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ with $\mathbf{w} = [3, 4]^T$, $b = -5$:
- (a) Normalize $\mathbf{w}$ so that $\|\mathbf{w}\| = 1$
- (b) What is the distance from point $\mathbf{x} = [2, 1]^T$ to this hyperplane?
- (c) What is the distance from point $\mathbf{x} = [1, 1]^T$ to this hyperplane?

**Problem 2.2:** Show that for any hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$, if we scale $\mathbf{w} \to c\mathbf{w}$, $b \to cb$ with $c > 0$, the hyperplane doesn't change but the functional margin $y_i(\mathbf{w}^T\mathbf{x}_i + b)$ scales by $c$.

**Problem 2.3:** Given a dataset with minimum functional margin $\gamma = 2$ and $\|\mathbf{w}\| = 3$, what is the geometric margin? If we rescale so $\|\mathbf{w}\| = 1$, what is the new functional margin?

**Problem 2.4:** Derive the distance from point $\mathbf{x}$ to hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ without assuming $\|\mathbf{w}\| = 1$. (Hint: project $\mathbf{x}$ onto the normal direction)

---

### Problem Set 3: Hard-SVM Formulation

**Problem 3.1:** Convert the max-margin problem to the min-norm problem step by step:
1. Start with $\max_{\|\mathbf{w}\|=1} \min_i y_i(\mathbf{w}^T\mathbf{x}_i + b)$
2. Let optimal margin be $\gamma^*$, define $\tilde{\mathbf{w}} = \mathbf{w}^*/\gamma^*$
3. Show $y_i(\tilde{\mathbf{w}}^T\mathbf{x}_i + \tilde{b}) \geq 1$
4. Show $\|\tilde{\mathbf{w}}\| = 1/\gamma^*$
5. Conclude: maximizing $\gamma^* \iff$ minimizing $\|\mathbf{w}\|$ with margin fixed to 1

**Problem 3.2:** For the Hard-SVM QP:
$$\min \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \quad \forall i$$
- (a) Why is this a convex optimization problem?
- (b) What is the Lagrangian?
- (c) Write the KKT conditions.

**Problem 3.3:** Suppose we have a 2D dataset with 2 support vectors at the margin. How many constraints are active (tight) at the solution? What does this tell us about $\mathbf{w}$?

**Problem 3.4:** Hard-SVM can be written as:
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$
Why is the $\frac{1}{2}$ factor included? Does it change the solution?

---

### Problem Set 4: Soft-SVM and Slack Variables

**Problem 4.1:** For a given $(\mathbf{w}, b)$, what is the smallest feasible $\xi_i$ for each point? Verify the table:

| $z = y_i(\mathbf{w}^T\mathbf{x}_i + b)$ | Smallest $\xi_i$ |
|------------------------------------------|------------------|
| 2.0 | 0 |
| 1.0 | 0 |
| 0.5 | 0.5 |
| 0 | 1 |
| -1 | 2 |
| -3 | 4 |

**Problem 4.2:** Show that the Soft-SVM problem with slack variables:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum \xi_i$$
$$\text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$
is equivalent to the unconstrained hinge loss formulation:
$$\min_{\mathbf{w}, b} \lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$$

**Problem 4.3:** Consider a dataset with one extreme outlier. Explain qualitatively how the decision boundary changes as $\lambda$ varies from very large to very small.

**Problem 4.4:** What is the relationship between the regularization parameter $C$ (often used in SVM libraries like scikit-learn) and $\lambda$ in our formulation? (Hint: $C = 1/(2\lambda)$ or similar)

---

### Problem Set 5: Hinge Loss Analysis

**Problem 5.1:** Plot the hinge loss $\ell(z) = \max(0, 1-z)$ and compare with:
- (a) Zero-one loss: $\mathbb{I}(z \leq 0)$
- (b) Squared loss: $(1-z)^2$
- (c) Logistic loss: $\log(1 + e^{-z})$
Discuss the differences in how they treat correctly classified points with $z > 1$.

**Problem 5.2:** Show that hinge loss is convex but not strictly convex. Where is it not differentiable?

**Problem 5.3:** For a single training example $(\mathbf{x}, y)$, the SVM objective is:
$$\lambda\|\mathbf{w}\|^2 + \max(0, 1 - y(\mathbf{w}^T\mathbf{x} + b))$$
- (a) Find the optimal $\mathbf{w}, b$ when $\lambda \to \infty$
- (b) Find the optimal $\mathbf{w}, b$ when $\lambda \to 0$
- (c) What happens at intermediate values?

**Problem 5.4:** Prove that hinge loss is an upper bound on the zero-one loss:
$$\max(0, 1 - z) \geq \mathbb{I}(z \leq 0) \quad \forall z$$

---

### Problem Set 6: Geometry and Support Vectors

**Problem 6.1:** In Hard-SVM, support vectors are points with $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$.
- (a) Why are they called "support vectors"?
- (b) Show that $\mathbf{w} = \sum_{i \in \text{SV}} \alpha_i y_i \mathbf{x}_i$ for some $\alpha_i \geq 0$ (using KKT).
- (c) If there are only 2 support vectors in 2D, what does the decision boundary look like?

**Problem 6.2:** For a linearly separable 2D dataset with 100 points, how many support vectors can there be at minimum? At maximum?

**Problem 6.3:** In Soft-SVM, points with $\xi_i > 0$ are also considered support vectors. Classify:
- $\xi_i = 0, y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$: margin support vector
- $\xi_i = 0, y_i(\mathbf{w}^T\mathbf{x}_i + b) > 1$: not a support vector
- $0 < \xi_i < 1$: margin error support vector
- $\xi_i = 1$: on decision boundary
- $\xi_i > 1$: misclassified support vector

**Problem 6.4:** Consider a 2D dataset where $+1$ class forms a small cluster inside the $-1$ class (like a bullseye). Is this linearly separable? What would Hard-SVM do? What would Soft-SVM do with different $\lambda$ values?

---

### Problem Set 7: Dual Formulation (Advanced)

**Problem 7.1:** Derive the dual of Hard-SVM:
1. Write Lagrangian: $\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_i \alpha_i (y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1)$
2. Minimize over $\mathbf{w}, b$ to get dual function
3. Show dual is: $\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$
4. Constraints: $\alpha_i \geq 0$, $\sum_i \alpha_i y_i = 0$

**Problem 7.2:** What is the kernel trick? Show how the dual formulation enables kernelization by replacing $\mathbf{x}_i^T \mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$.

**Problem 7.3:** For Soft-SVM, the dual has box constraints $0 \leq \alpha_i \leq C$. Derive this and explain the relationship between $C$ and $\lambda$.

**Problem 7.4:** In the dual, only support vectors have $\alpha_i > 0$. Explain how this leads to sparse solutions and efficient prediction.

---

### Problem Set 8: Worked Numerical Examples

**Example 1: Hard-SVM in 2D**

Dataset: $(x_1, x_2, y) = \{(1, 2, +1), (2, 1, +1), (0, 0, -1)\}$

1. Plot the points
2. Find the maximum margin separator by inspection
3. Compute $\mathbf{w}, b$ for this separator
4. Verify margin = 1 for the support vectors
5. Check $\|\mathbf{w}\|$

**Example 2: Soft-SVM with One Outlier**

Base dataset (separable): $(1, 1, +1), (2, 2, +1), (-1, -1, -1), (-2, -2, -1)$
Add outlier: $(10, 10, -1)$

1. What does Hard-SVM do? (Try to solve)
2. For Soft-SVM with $\lambda = 1$, what happens to the outlier?
3. For Soft-SVM with $\lambda = 0.01$, what happens?
4. Discuss the trade-off.

**Example 3: Computing Hinge Loss**

For $\mathbf{w} = [1, -1]^T, b = 0$:
| $\mathbf{x}_i$ | $y_i$ | $z = y_i(\mathbf{w}^T\mathbf{x}_i + b)$ | $\ell_{\text{hinge}}$ |
|----------------|-------|-----------------------------------------|---------------------|
| $[2, 0]^T$ | $+1$ | | |
| $[0, 2]^T$ | $+1$ | | |
| $[-1, -1]^T$ | $-1$ | | |
| $[0.5, 0.5]^T$ | $+1$ | | |

**Example 4: Regularization Path**

Consider 1D data: $x \in \{-2, -1, 1, 2\}$ with $y = \text{sign}(x)$.
Solve Soft-SVM: $\min_w \lambda w^2 + \frac{1}{4}\sum \max(0, 1 - y_i(w x_i))$ (no bias for simplicity)

1. Find $w^*$ as function of $\lambda$
2. What happens as $\lambda \to 0$? As $\lambda \to \infty$?
3. Plot $w^*$ vs $\lambda$

---

### Problem Set 9: Generalization and Theory

**Problem 9.1:** (VC Dimension) The VC dimension of linear classifiers in $\mathbb{R}^d$ is $d+1$. How does the margin affect the effective capacity? (Hint: margin bounds)

**Problem 9.2:** (Margin Bound) State and explain the generalization bound for SVMs:
$$\text{Test error} \leq \text{Training error} + O\left(\frac{R^2}{\gamma^2 m}\right)$$
where $R$ is the radius of the data and $\gamma$ is the margin.

**Problem 9.3:** Why does maximizing margin improve generalization? Connect to:
- Compression argument
- PAC-Bayes
- Stability

**Problem 9.4:** (SVM vs Logistic Regression) Both use linear classifiers. Compare:
- Loss functions (hinge vs logistic)
- Probabilistic outputs
- Sparsity of solutions
- Sensitivity to outliers

---

### Problem Set 10: Extensions and Variations

**Problem 10.1:** (Kernel SVM) For the XOR problem (not linearly separable in 2D):
- Map $\phi(x_1, x_2) = (x_1, x_2, x_1 x_2)$
- Show it becomes linearly separable in 3D
- What kernel corresponds to this mapping?

**Problem 10.2:** (Multi-class SVM) How to extend binary SVM to $K$ classes?
- One-vs-Rest
- One-vs-One
- Crammer-Singer formulation

**Problem 10.3:** (ν-SVM) In ν-SVM, the parameter $\nu \in (0,1]$ controls:
- Upper bound on fraction of margin errors
- Lower bound on fraction of support vectors
Explain the intuition.

**Problem 10.4:** (Structured SVM) For structured output (e.g., sequences, trees), the hinge loss generalizes to:
$$\max_{\hat{y} \neq y} [\Delta(y, \hat{y}) + \mathbf{w}^T \phi(\mathbf{x}, \hat{y})] - \mathbf{w}^T \phi(\mathbf{x}, y)$$
Explain the components and how this reduces to standard hinge loss for binary classification.

---

## Key Formulas Reference

| Concept | Formula |
|---------|---------|
| **Linear classifier** | $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ |
| **Decision boundary** | $\mathbf{w}^T \mathbf{x} + b = 0$ |
| **Prediction** | $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$ |
| **Correct classification** | $y_i(\mathbf{w}^T \mathbf{x}_i + b) > 0$ |
| **Distance to hyperplane** ($\|\mathbf{w}\|=1$) | $|\mathbf{w}^T \mathbf{x} + b|$ |
| **Hard-SVM (primal)** | $\min \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ |
| **Slack variable** | $\xi_i = \max\{0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)\}$ |
| **Hinge loss** | $\ell_{\text{hinge}}(z) = \max(0, 1 - z)$ |
| **Soft-SVM (primal)** | $\min \lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$ |
| **Soft-SVM (unconstrained)** | $\min \lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$ |
| **Regularization view** | $\lambda \|\mathbf{w}\|^2 + L_{\text{hinge}}(\mathbf{w}, b)$ |
| **Margin** | $\gamma = 1/\|\mathbf{w}\|$ (when functional margin = 1) |

---

## Navigation

| Previous | Next |
|----------|------|
| [Lecture 9: Logistic Regression]({{ site.baseurl }}/lectures/Lecture_9_Logistic_Regression_Notes/) | — |

*End of Lecture Notes*