# Lecture 6-7: Linear Models for Classification

**Instructor:** Dr. Anirban Sen  
**Date:** September 14, 2026  

---

## Table of Contents

1. [Introduction: Why Classification?](#1-introduction-why-classification)
2. [Regression vs Classification](#2-regression-vs-classification)
3. [Why Linear Models?](#3-why-linear-models)
4. [Discriminative vs Generative Approaches](#4-discriminative-vs-generative-approaches)
5. [Discriminant Functions and Decision Boundaries](#5-discriminant-functions-and-decision-boundaries)
6. [Geometry of Linear Decision Boundaries](#6-geometry-of-linear-decision-boundaries)
7. [Signed Distance from Decision Boundary](#7-signed-distance-from-decision-boundary)
8. [Binary Classification with Linear Discriminants](#8-binary-classification-with-linear-discriminants)
9. [Multiclass Discriminant Functions](#9-multiclass-discriminant-functions)
10. [One-vs-One vs One-vs-Rest vs K-Class Discriminant](#10-one-vs-one-vs-one-vs-rest-vs-k-class-discriminant)
11. [Decision Boundaries in Multiclass Case](#11-decision-boundaries-in-multiclass-case)
12. [Algebraic Constraints on Decision Boundaries](#12-algebraic-constraints-on-decision-boundaries)
13. [Why K-Class Construction Avoids Ambiguity](#13-why-k-class-construction-avoids-ambiguity)
14. [Classroom Exercise](#14-classroom-exercise)
15. [Problem Sets and Examples](#15-problem-sets-and-examples)

---

## 1. Introduction: Why Classification?

### 1.1 Goal
Assign an input vector **x** to one of a finite set of classes.

### 1.2 Examples
| Domain | Classes |
|--------|---------|
| Email | Spam / Not Spam |
| Medical Diagnosis | Healthy / Diseased |
| Digit Recognition | 0–9 |
| Face Recognition | Person A / Person B / ... |

---

## 2. Regression vs Classification

| Aspect | Regression | Classification |
|--------|------------|----------------|
| **Output** | Continuous | Discrete labels |
| **Goal** | Predict value | Predict class |
| **Loss Function** | Squared error | Decision boundary |

---

## 3. Why Linear Models?

- Simple mathematical form
- Interpretable parameters
- Foundation for Logistic Regression and SVM
- Can be extended using basis functions and kernels

---

## 4. Discriminative vs Generative Approaches

| Approach | Description |
|----------|-------------|
| **Discriminative** | Learn $p(C_k | \mathbf{x})$ or a decision function directly |
| **Generative** | Learn $p(\mathbf{x} | C_k)$ and $p(C_k)$, then use Bayes' theorem |

---

## 5. Discriminant Functions and Decision Boundaries

### 5.1 What is a Discriminant Function?

A discriminant function assigns a score to every input vector:
$$y(\mathbf{x}) = f(\mathbf{x})$$

For **linear discriminants**:
$$y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$$

### 5.2 Decision Rule
$$\hat{C}(\mathbf{x}) = \begin{cases}
C_1, & y(\mathbf{x}) > 0 \\
C_2, & y(\mathbf{x}) < 0
\end{cases}$$

### 5.3 Meaning of Parameters

| Parameter | Role |
|-----------|------|
| **Weight vector** $\mathbf{w}$ | Determines the **orientation** of the decision boundary |
| **Bias** $w_0$ | **Shifts** the boundary without changing its orientation |

**Decision boundary:**
$$\mathbf{w}^T \mathbf{x} + w_0 = 0$$

---

## 6. Geometry of Linear Decision Boundaries

### 6.1 Hyperplanes

| Dimension | Equation |
|-----------|----------|
| 2D | $ax + by + c = 0$ |
| 3D | $ax + by + cz + d = 0$ |
| $D$-dimensions | $\mathbf{w}^T \mathbf{x} + w_0 = 0$ |

A hyperplane has dimension $D - 1$.

### 6.2 Vector $\mathbf{w}$ is Perpendicular to the Hyperplane

![Geometry](geometry_figure.png)

The vector $\mathbf{w}$ is the **normal vector** to the hyperplane.

---

## 7. Signed Distance from Decision Boundary

### 7.1 Closest Point on Hyperplane to Origin

Let $\mathbf{x}^*$ be the point on the hyperplane closest to the origin.

Since $\mathbf{x}^*$ must be parallel to $\mathbf{w}$:
$$\mathbf{x}^* = \lambda \mathbf{w}$$

Substitute into hyperplane equation:
$$\mathbf{w}^T (\lambda \mathbf{w}) + w_0 = 0$$
$$\lambda \|\mathbf{w}\|^2 + w_0 = 0$$
$$\lambda = -\frac{w_0}{\|\mathbf{w}\|^2}$$

**Closest point:**
$$\mathbf{P} = -\frac{w_0}{\|\mathbf{w}\|^2} \mathbf{w}$$

**Distance from origin:**
$$d = \|\mathbf{P}\| = \frac{|w_0|}{\|\mathbf{w}\|}$$

### 7.2 Signed Distance of Any Point $\mathbf{x}$

Let $\mathbf{x}_\perp$ be the orthogonal projection of $\mathbf{x}$ onto the hyperplane.

$$\mathbf{x} = \mathbf{x}_\perp + r \frac{\mathbf{w}}{\|\mathbf{w}\|}$$

where:
- $\mathbf{x}_\perp$: projection of $\mathbf{x}$ onto the hyperplane
- $\frac{\mathbf{w}}{\|\mathbf{w}\|}$: unit normal vector
- $r$: signed perpendicular distance from hyperplane to $\mathbf{x}$

### 7.3 Derivation of Signed Distance

Multiply both sides by $\mathbf{w}^T$:
$$\mathbf{w}^T \mathbf{x} = \mathbf{w}^T \mathbf{x}_\perp + r \frac{\mathbf{w}^T \mathbf{w}}{\|\mathbf{w}\|}$$

Since $\mathbf{w}^T \mathbf{w} = \|\mathbf{w}\|^2$:
$$\mathbf{w}^T \mathbf{x} = \mathbf{w}^T \mathbf{x}_\perp + r \|\mathbf{w}\|$$

Add $w_0$ to both sides:
$$\mathbf{w}^T \mathbf{x} + w_0 = \mathbf{w}^T \mathbf{x}_\perp + w_0 + r \|\mathbf{w}\|$$

But $\mathbf{x}_\perp$ lies on the decision surface, so $\mathbf{w}^T \mathbf{x}_\perp + w_0 = 0$.

Therefore:
$$\mathbf{w}^T \mathbf{x} + w_0 = r \|\mathbf{w}\|$$

Recall $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$.

### 7.4 Key Result

$$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|}$$

**Interpretation:**
- The discriminant value $y(\mathbf{x})$ is **proportional to the signed perpendicular distance** from the decision boundary.
- $r > 0$: positive side (class $C_1$)
- $r < 0$: negative side (class $C_2$)
- $r = 0$: on the hyperplane

---

## 8. Binary Classification with Linear Discriminants

### 8.1 Classification Rule

Given two classes $C_1, C_2$:

$$\text{Predict } \begin{cases}
C_1, & \text{if } y(\mathbf{x}) > 0 \\
C_2, & \text{if } y(\mathbf{x}) < 0
\end{cases}$$

### 8.2 Decision Regions

Each side of the boundary corresponds to one class:
- Region $y(\mathbf{x}) > 0$ → Class $C_1$
- Region $y(\mathbf{x}) < 0$ → Class $C_2$

---

## 9. Multiclass Discriminant Functions

### 9.1 From Two Classes to K Classes

For two classes, linear discriminant:
$$y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$$
Decision boundary: $y(\mathbf{x}) = 0$

For $K$ classes, one might try:
1. **One-versus-the-rest**
2. **One-versus-one**

Both can produce **ambiguous regions**.

### 9.2 K-Class Linear Discriminant Construction

Define $K$ linear discriminant functions:
$$y_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}, \quad k = 1, \ldots, K$$

**Classification rule:**
$$\hat{C}(\mathbf{x}) = \arg\max_k y_k(\mathbf{x})$$

Equivalently:
$$\mathbf{x} \in \mathcal{R}_k \iff y_k(\mathbf{x}) > y_j(\mathbf{x}) \quad \forall j \neq k$$

**Key idea:** We do **not** ask whether $y_k(\mathbf{x})$ is positive or negative. We ask **which discriminant has the largest value**.

---

## 10. One-vs-One vs One-vs-Rest vs K-Class Discriminant

### 10.1 One-vs-One (OvO)

For $K=3$, construct three binary classifiers:
- $C_1$ vs $C_2$: $y_{12}(\mathbf{x}) = \mathbf{w}_{12}^T \mathbf{x} + w_{120}$
- $C_1$ vs $C_3$: $y_{13}(\mathbf{x}) = \mathbf{w}_{13}^T \mathbf{x} + w_{130}$
- $C_2$ vs $C_3$: $y_{23}(\mathbf{x}) = \mathbf{w}_{23}^T \mathbf{x} + w_{230}$

Decision boundaries: $y_{12}=0, y_{13}=0, y_{23}=0$

**Problem:** These three classifiers are trained **independently**. Their three decision planes are **not algebraically constrained** to pass through a common point/line.

### 10.2 One-vs-One Problem

The three pairwise classifiers can create **regions where pairwise decisions are inconsistent** (ambiguous regions). Such regions cannot be assigned unambiguously to one of the three classes.

### 10.3 K-Class Discriminant (Single Construction)

One coherent set of discriminant functions: $y_1, y_2, y_3$

Decision boundaries: $y_i(\mathbf{x}) = y_j(\mathbf{x})$

---

## 11. Decision Boundaries in Multiclass Case

### 11.1 Pairwise Decision Boundary

Consider classes $C_1$ and $C_2$:
$$y_1(\mathbf{x}) = y_2(\mathbf{x})$$

Substitute:
$$\mathbf{w}_1^T \mathbf{x} + w_{10} = \mathbf{w}_2^T \mathbf{x} + w_{20}$$

Rearrange:
$$(\mathbf{w}_1 - \mathbf{w}_2)^T \mathbf{x} + (w_{10} - w_{20}) = 0$$

**This is a linear equation in $\mathbf{x}$.** For $\mathbf{x} \in \mathbb{R}^D$, the decision boundary is a $(D-1)$-dimensional hyperplane.

### 11.2 Two Different Types of Planes (Don't Confuse!)

| Equation | Meaning |
|----------|---------|
| $y_1(\mathbf{x}) = 0$ | Zero-plane of discriminant 1 |
| $y_2(\mathbf{x}) = 0$ | Zero-plane of discriminant 2 |
| $y_1(\mathbf{x}) = y_2(\mathbf{x})$ | **Decision boundary** between $C_1, C_2$ |

**Example:** A point can have $y_1(\mathbf{x}) = 5, y_2(\mathbf{x}) = 5, y_3(\mathbf{x}) = -2$. It lies on the $C_1/C_2$ decision boundary **even though it lies on neither $y_1=0$ nor $y_2=0$**.

---

## 12. Algebraic Constraints on Decision Boundaries

### 12.1 Three Classes: Three Decision Planes

$$H_{12}: y_1 = y_2 \implies (\mathbf{w}_1 - \mathbf{w}_2)^T \mathbf{x} + (w_{10} - w_{20}) = 0$$
$$H_{23}: y_2 = y_3 \implies (\mathbf{w}_2 - \mathbf{w}_3)^T \mathbf{x} + (w_{20} - w_{30}) = 0$$
$$H_{13}: y_1 = y_3 \implies (\mathbf{w}_1 - \mathbf{w}_3)^T \mathbf{x} + (w_{10} - w_{30}) = 0$$

### 12.2 Crucial Constraint: Three Planes Are Not Independent

Suppose $\mathbf{x}$ lies on both $H_{12}$ and $H_{23}$:
$$y_1(\mathbf{x}) = y_2(\mathbf{x}) \quad \text{and} \quad y_2(\mathbf{x}) = y_3(\mathbf{x})$$

Then:
$$y_1(\mathbf{x}) = y_2(\mathbf{x}) = y_3(\mathbf{x}) \implies y_1(\mathbf{x}) = y_3(\mathbf{x})$$

So $\mathbf{x}$ also lies on $H_{13}$.

### 12.3 Algebraic Proof

Add $H_{12}$ and $H_{23}$:
$$(\mathbf{w}_1 - \mathbf{w}_2)^T \mathbf{x} + (\mathbf{w}_2 - \mathbf{w}_3)^T \mathbf{x} + (w_{10} - w_{20}) + (w_{20} - w_{30}) = 0$$

Simplify:
$$(\mathbf{w}_1 - \mathbf{w}_3)^T \mathbf{x} + (w_{10} - w_{30}) = 0$$

**This is exactly $H_{13}$.**

### 12.4 Contrast with One-vs-One

For independently trained pairwise classifiers (OvO), the three boundary equations **need not satisfy this relationship**. For a single $K$-class discriminant, they **necessarily do**.

---

## 13. Why K-Class Construction Avoids Ambiguity

| Property | One-vs-One (OvO) | Single K-Class Discriminant |
|----------|------------------|----------------------------|
| **Classifiers** | Three independently trained | One coherent set of scores $y_1, y_2, y_3$ |
| **Boundaries** | Independent | $y_i = y_j$ |
| **Pairwise boundaries** | Algebraically unconstrained | Algebraically linked |
| **Ambiguous regions** | Can create | None |
| **Decision regions** | May be non-convex/multi-connected | **Convex and singly connected** |
| **Classification** | Requires voting | $\hat{C}(\mathbf{x}) = \arg\max_k y_k(\mathbf{x})$ |

### 13.1 Main Takeaway

> The important difference is not simply "three planes versus three planes." It is the **relationship between the discriminant functions** that generate those planes.

---

## 14. Classroom Exercise

**Given:**
$$y(\mathbf{x}) = 4x_1 + 2x_2 - 8$$

**Find:**
1. Decision boundary
2. Normal vector
3. Distance from origin
4. Class of $(2, 1)$

### Solution:

1. **Decision boundary:** $4x_1 + 2x_2 - 8 = 0 \implies 2x_1 + x_2 = 4$

2. **Normal vector:** $\mathbf{w} = [4, 2]^T$

3. **Distance from origin:**
   $$d = \frac{|w_0|}{\|\mathbf{w}\|} = \frac{|-8|}{\sqrt{4^2 + 2^2}} = \frac{8}{\sqrt{20}} = \frac{8}{2\sqrt{5}} = \frac{4}{\sqrt{5}} \approx 1.79$$

4. **Class of $(2, 1)$:**
   $$y(2, 1) = 4(2) + 2(1) - 8 = 8 + 2 - 8 = 2 > 0 \implies \text{Class } C_1$$

---

## 15. Problem Sets and Examples

### Problem Set 1: Binary Linear Discriminants

**Problem 1.1:** Given $y(\mathbf{x}) = 3x_1 - 5x_2 + 2$:
- (a) Write the decision boundary equation
- (b) Find the normal vector
- (c) Compute distance from origin to decision boundary
- (d) Classify points: $(1, 1), (0, 0), (2, 1)$

**Problem 1.2:** For a binary classifier with $\mathbf{w} = [2, -1, 3]^T$ and $w_0 = -4$:
- (a) What is the dimension of the decision boundary?
- (b) Find the point on the decision boundary closest to the origin
- (c) If $\mathbf{x} = [1, 2, 0]^T$, what is the signed distance to the boundary?

**Problem 1.3:** Prove that for any point $\mathbf{x}$, the signed distance to the hyperplane $\mathbf{w}^T \mathbf{x} + w_0 = 0$ is $r = \frac{\mathbf{w}^T \mathbf{x} + w_0}{\|\mathbf{w}\|}$.

**Problem 1.4:** Given two points $\mathbf{x}_A = [1, 2]^T$ and $\mathbf{x}_B = [3, 1]^T$ with discriminant $y(\mathbf{x}) = x_1 - 2x_2 + 3$:
- (a) Which side of the boundary is each point on?
- (b) What is the perpendicular distance between these two points along the normal direction?

---

### Problem Set 2: Multiclass Discriminants

**Problem 2.1:** For a 3-class problem with:
- $y_1(\mathbf{x}) = 2x_1 + x_2 - 1$
- $y_2(\mathbf{x}) = -x_1 + 3x_2 + 2$
- $y_3(\mathbf{x}) = x_1 - 2x_2$

Find the classification for $\mathbf{x} = [1, 1]^T, [0, 0]^T, [2, -1]^T$.

**Problem 2.2:** For the same discriminants above, derive the equations of all three pairwise decision boundaries: $H_{12}, H_{23}, H_{13}$.

**Problem 2.3:** Verify that the three boundaries $H_{12}, H_{23}, H_{13}$ from Problem 2.2 intersect at a single line (are algebraically dependent).

**Problem 2.4:** In a 4-class problem, how many pairwise decision boundaries are there? Write the general equation for the boundary between $C_i$ and $C_j$.

---

### Problem Set 3: One-vs-One vs K-Class Construction

**Problem 3.1:** Explain why One-vs-One classification with 3 classes can produce ambiguous regions. Draw a diagram showing a possible ambiguous region.

**Problem 3.2:** Suppose we use One-vs-Rest (OvR) for a 3-class problem:
- Classifier 1: $C_1$ vs $\{C_2, C_3\}$
- Classifier 2: $C_2$ vs $\{C_1, C_3\}$
- Classifier 3: $C_3$ vs $\{C_1, C_2\}$

Can OvR produce ambiguous regions? If so, describe what they look like.

**Problem 3.3:** Compare the number of parameters in:
- (a) OvO for $K$ classes
- (b) OvR for $K$ classes
- (c) Single K-class discriminant

Assume input dimension $D$. Which has the fewest parameters?

**Problem 3.4:** In OvO, each binary classifier is trained independently. What happens to the decision boundaries if the training data is highly imbalanced (e.g., 90% in $C_1$, 5% in $C_2$, 5% in $C_3$)?

---

### Problem Set 4: Geometry and Visualization

**Problem 4.1:** In $\mathbb{R}^3$, consider three classes with discriminant functions:
- $y_1(\mathbf{x}) = x_1 + 2x_2 - x_3 + 1$
- $y_2(\mathbf{x}) = -x_1 + x_2 + 2x_3 - 2$
- $y_3(\mathbf{x}) = 2x_1 - x_2 + x_3$

Show that the three decision planes $y_1=y_2$, $y_2=y_3$, $y_1=y_3$ intersect in a single line. Find the parametric equation of this line.

**Problem 4.2:** A point $\mathbf{x}$ in $\mathbb{R}^2$ has discriminant scores $y_1 = 3, y_2 = 5, y_3 = 1$. Which class is it assigned to? Is it on any decision boundary?

**Problem 4.3:** Consider a 3-class problem in $\mathbb{R}^2$ where the decision boundaries are:
- $H_{12}: x_1 + x_2 = 0$
- $H_{23}: x_1 - x_2 = 0$
- $H_{13}: 2x_1 = 0$

Can these boundaries come from a single K-class discriminant construction? Why or why not?

**Problem 4.4:** Show that the decision regions $\mathcal{R}_k = \{\mathbf{x} : y_k(\mathbf{x}) > y_j(\mathbf{x}) \ \forall j \neq k\}$ are **convex** sets.

---

### Problem Set 5: Least Squares for Classification (Bishop 4.1.3)

**Problem 5.1:** For a binary classification problem with targets $t \in \{0, 1\}$, we can use linear regression to fit $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$. Why is squared error loss inappropriate for classification?

**Problem 5.2:** In least-squares classification, we assign $t=1$ for $C_1$ and $t=0$ for $C_2$. Show that the least-squares solution approximates the class conditional mean $\mathbb{E}[t|\mathbf{x}] = p(C_1|\mathbf{x})$.

**Problem 5.3:** What is the main limitation of using least squares for classification? (Hint: Consider outliers and the fact that the target is not truly continuous.)

**Problem 5.4:** For a 3-class problem, we could encode targets as $\mathbf{t} \in \{(1,0,0), (0,1,0), (0,0,1)\}$ and fit three linear regression models. What are the problems with this approach?

---

## Key Takeaways Summary

1. **Linear discriminant function:** $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$
2. **Decision boundary:** $\mathbf{w}^T \mathbf{x} + w_0 = 0$ (hyperplane, dimension $D-1$)
3. **Normal vector:** $\mathbf{w}$ is perpendicular to the decision boundary
4. **Signed distance:** $r = \frac{\mathbf{w}^T \mathbf{x} + w_0}{\|\mathbf{w}\|}$ — discriminant value is proportional to signed distance
5. **Binary classification:** $y(\mathbf{x}) > 0 \to C_1$, $y(\mathbf{x}) < 0 \to C_2$
6. **Multiclass (K-class):** Define $K$ discriminants $y_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}$
7. **Classification rule:** $\hat{C}(\mathbf{x}) = \arg\max_k y_k(\mathbf{x})$
8. **Pairwise boundary:** $y_i(\mathbf{x}) = y_j(\mathbf{x}) \iff (\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (w_{i0} - w_{j0}) = 0$
9. **Zero-plane $\neq$ decision boundary:** $y_k(\mathbf{x}) = 0$ is not the class boundary
10. **Algebraic constraint:** Three pairwise boundaries $H_{12}, H_{23}, H_{13}$ always intersect in a common line (not independent)
11. **K-class avoids ambiguity:** Decision regions are convex and singly connected
12. **OvO/OvR problems:** Independent training creates ambiguous regions; K-class uses one coherent set of scores

---

*End of Lecture Notes*