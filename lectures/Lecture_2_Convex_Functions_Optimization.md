---
layout: default
title: "Lecture 2: Convex Functions & Optimization"
nav_order: 1
parent: Lectures
has_toc: true
permalink: /lectures/Lecture_2_Convex_Functions_Optimization/
---

# Lecture 2: Convex Functions and Convex Optimization

**Instructor:** Dr. Anirban Sen  
**Institution:** Ashoka University  

---

## Table of Contents

1. [Definition of a Convex Function](#1-definition-of-a-convex-function)
2. [Geometric Interpretation](#2-geometric-interpretation)
3. [Intuitive Understanding](#3-intuitive-understanding)
4. [The Chord Between Two Points](#4-the-chord-between-two-points)
5. [Why Convexity Matters for Optimization](#5-why-convexity-matters-for-optimization)
6. [Gradient and Hessian](#6-gradient-and-hessian)
7. [Detecting Convexity: The zᵀHz Condition](#7-detecting-convexity-the-zthz-condition)
8. [Why zᵀHz Measures Curvature](#8-why-zthz-measures-curvature)
9. [Why zᵀHz ≥ 0 Guarantees Convexity](#9-why-zthz--0-guarantees-convexity)
10. [Examples of Convex Functions](#10-examples-of-convex-functions)
11. [Important Counterexamples and Misconceptions](#11-important-counterexamples-and-misconceptions)
12. [Summary: The Big Picture](#12-summary-the-big-picture)
13. [Problem Sets and Examples](#13-problem-sets-and-examples)

---

## 1. Definition of a Convex Function

### 1.1 Formal Definition

A function $f$ is **convex** on a **convex domain** if, for any $x_1, x_2$ in the domain and any $0 \leq \theta \leq 1$:

$$f(\theta x_1 + (1 - \theta)x_2) \leq \theta f(x_1) + (1 - \theta)f(x_2)$$

### 1.2 In Words

> The function evaluated at a weighted average of two inputs is no larger than the corresponding weighted average of the two function values.

### 1.3 Key Requirements

| Requirement | Description |
|-------------|-------------|
| **Convex Domain** | The set of all $\theta x_1 + (1-\theta)x_2$ must be in the domain |
| **For ALL $x_1, x_2$** | The inequality must hold for every pair of points in the domain |
| **For ALL $\theta \in [0,1]$** | The inequality must hold for every convex combination |

---

## 2. Geometric Interpretation

### 2.1 Graph Below the Chord

For a convex function, the **graph lies below the chord joining any two points** on the graph.

```
f(x)
  │
  │    ● (x₂, f(x₂))
  │   /|
  │  / |
  │ /  |  chord
  │/   |
  ●────●─────── x
(x₁, f(x₁))
```

The function curve is always **below or on** the straight line (chord) connecting any two points on the graph.

---

## 3. Intuitive Understanding

### 3.1 "Bends Upward"

Think of a convex function as a function that:
- **"Bends upward"**
- Equivalently: **its slope does not decrease** as we move from left to right

### 3.2 One-Dimensional Condition

For a one-dimensional function $f(x)$:
$$f''(x) \geq 0 \iff \text{slope } f'(x) \text{ is increasing}$$

### 3.3 Example: $f(x) = x^2$

- $f'(x) = 2x$ (slope)
- $f''(x) = 2 > 0$ (slope increases everywhere)
- Slope progression: $-10 \to -4 \to 0 \to 4 \to 10$

---

## 4. The Chord Between Two Points

### 4.1 Two Points on the Graph

$$P_1 = (x_1, f(x_1)), \quad P_2 = (x_2, f(x_2))$$

### 4.2 Points on the Line Segment (Chord)

A point on the line segment joining them:
$$P_\theta = \theta P_1 + (1 - \theta)P_2, \quad 0 \leq \theta \leq 1$$

### 4.3 Coordinates Explicitly

$$\begin{bmatrix} x_\theta \\ y_\theta \end{bmatrix} = \begin{bmatrix} \theta x_1 + (1 - \theta)x_2 \\ \theta f(x_1) + (1 - \theta)f(x_2) \end{bmatrix}$$

- $x_\theta = \theta x_1 + (1 - \theta)x_2$ (input)
- $y_\theta = \theta f(x_1) + (1 - \theta)f(x_2)$ (chord height)

### 4.4 Convexity Condition

At $x_\theta$, the actual function height is $f(x_\theta) = f(\theta x_1 + (1 - \theta)x_2)$.

Convexity requires:
$$f(x_\theta) \leq y_{\text{chord}}$$

or:
$$f(\theta x_1 + (1 - \theta)x_2) \leq \theta f(x_1) + (1 - \theta)f(x_2)$$

---

## 5. Why Convexity Matters for Optimization

### 5.1 The Key Guarantee

> **For a convex function: every local minimum is a global minimum.**

### 5.2 Practical Implication

If an optimization algorithm finds a point where:
$$\nabla f(x^*) = 0$$

we **do not have to worry** that there is a better minimum somewhere else.

### 5.3 Why This Matters

This is one reason **convex optimization problems are particularly well-behaved** — no need for multiple restarts, no worry about getting stuck in poor local minima.

---

## 6. Gradient and Hessian

### 6.1 Gradient (Multivariate)

For $f(x_1, \ldots, x_n)$, the gradient is a **vector** pointing in the direction of steepest increase:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### 6.2 Hessian (Multivariate)

The **Hessian** is the derivative of the gradient (matrix of second derivatives):

$$H = \nabla^2 f$$

For two variables:
$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} \end{bmatrix}$$

The Hessian describes the **curvature** of the function.

### 6.3 Convexity and Hessian

For a twice continuously differentiable function:
- $H \succeq 0$ (positive semidefinite) $\implies$ $f$ is convex
- $H \succ 0$ (positive definite) $\implies$ $f$ is **strictly** convex

---

## 7. Detecting Convexity: The $z^T H z$ Condition

### 7.1 The Key Test

Let $z$ be **any direction** (any vector). The quantity:

$$z^T H z$$

measures the **second-order curvature** of the function in direction $z$.

### 7.2 Condition for Convexity

If
$$z^T H z \geq 0 \quad \text{for every } z$$

then the function curves upward in **every direction**.

This is precisely the condition:
$$H \succeq 0 \quad \text{(Hessian is positive semidefinite)}$$

---

## 8. Why $z^T H z$ Measures Curvature

### 8.1 Reduction to One Dimension

Pick a point $x$ and a direction $z$. Move along the one-dimensional line:

$$x(t) = x + t z$$

Define the one-dimensional function:
$$g(t) = f(x + t z)$$

### 8.2 First Derivative

$$g'(t) = \nabla f(x + t z)^T z$$

### 8.3 Second Derivative

$$g''(t) = z^T H(x + t z) z$$

### 8.4 Conclusion

> **$z^T H z$ is exactly the second derivative when we move in direction $z$.**

---

## 9. Why $z^T H z \geq 0$ Guarantees Convexity

### 9.1 Chain of Logic

1. Suppose $z^T H(x) z \geq 0$ for every $x$ and every $z$
2. Then $g''(t) \geq 0$ for the one-dimensional function $g(t) = f(x + t z)$
3. For a 1D function, $g''(t) \geq 0$ means $g$ is **convex**
4. This is true **along every possible line** through the domain
5. Therefore, the original multivariate function $f$ is convex

$$H \succeq 0 \implies f \text{ is convex}$$

### 9.2 One-Dimensional Argument

Why does $g''(t) \geq 0$ imply convexity?

- $g''(t) \geq 0 \implies g'(t)$ is **non-decreasing**
- As we move left to right, **slope never decreases**
- Graph **bends upward**
- Equivalently: for any $t_1, t_2$:
  $$g(\theta t_1 + (1-\theta)t_2) \leq \theta g(t_1) + (1-\theta)g(t_2)$$
- Since this holds along **every direction** $z$, the multivariate function is convex

---

## 10. Examples of Convex Functions

### Example 1: $f(x) = x^2$ (1D)

| Derivative | Expression | Sign |
|------------|------------|------|
| $f'(x)$ | $2x$ | — |
| $f''(x)$ | $2$ | $> 0$ |

**Conclusion:** $x^2$ is **strictly convex** on $\mathbb{R}$.

---

### Example 2: $f(x_1, x_2) = x_1^2 + x_2^2$ (2D)

**Gradient:**
$$\nabla f = \begin{bmatrix} 2x_1 \\ 2x_2 \end{bmatrix}$$

**Hessian:**
$$H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} = 2I$$

**Test $z^T H z$:**
Take arbitrary $z = \begin{bmatrix} z_1 \\ z_2 \end{bmatrix}$

$$z^T H z = \begin{bmatrix} z_1 & z_2 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} z_1 \\ z_2 \end{bmatrix} = 2z_1^2 + 2z_2^2 = 2\|z\|^2 \geq 0$$

**Conclusion:** Strictly convex (bowl-shaped surface).

**Unique minimum:** At $\nabla f = 0 \implies x_1 = 0, x_2 = 0$.

---

### Example 3: $f(x) = e^{ax}$ (1D, parameterized)

**First derivative:** $f'(x) = a e^{ax}$

**Second derivative:** $f''(x) = a^2 e^{ax}$

Since $a^2 \geq 0$ and $e^{ax} > 0$ for all $x$ and all $a \in \mathbb{R}$:
$$f''(x) \geq 0 \quad \text{for every } x \text{ and every } a \in \mathbb{R}$$

**Conclusion:** $e^{ax}$ is **convex for every $a \in \mathbb{R}$**. For $a \neq 0$, strictly convex.

---

## 11. Important Counterexamples and Misconceptions

### Misconception 1: "Convex $\implies$ Has a Minimum"

**FALSE!** Convexity tells us: **IF a local minimum exists, it is global.** It does **not** guarantee a minimum is actually attained.

#### Counterexample: $f(x) = e^{ax}$

| Case | Behavior | Minimum? |
|------|----------|----------|
| $a > 0$ | $e^{ax} \to 0$ as $x \to -\infty$, but $e^{ax} > 0$ for all finite $x$ | **No minimum**, infimum = 0 |
| $a < 0$ | $e^{ax} \to 0$ as $x \to +\infty$, but $e^{ax} > 0$ for all finite $x$ | **No minimum**, infimum = 0 |
| $a = 0$ | $f(x) = 1$ (constant) | **Every** $x \in \mathbb{R}$ is a global minimum |

---

### Misconception 2: "Has a Minimum $\implies$ Convex"

**FALSE!** Having a global minimum does not imply convexity.

#### Counterexample: $f(x) = x^4 - x^2$

**Analysis:**
- $f'(x) = 4x^3 - 2x = 2x(2x^2 - 1)$
- Critical points: $x = 0, \pm \frac{1}{\sqrt{2}}$
- $f''(x) = 12x^2 - 2$
- $f''(0) = -2 < 0$ (local maximum at $x=0$)
- $f''(\pm 1/\sqrt{2}) = 12(1/2) - 2 = 4 > 0$ (local minima)

**Global minima:** $x = \pm \frac{1}{\sqrt{2}}$ with value $f = -\frac{1}{4}$

**But:** $f''(x) = 12x^2 - 2$ is **negative** for $|x| < \frac{1}{\sqrt{6}}$
$\implies$ Function is **not convex** (it's concave near $x=0$)

---

## 12. Summary: The Big Picture

```
Convexity
    ↓
Chord lies above graph
    ↓
Slope does not decrease
    ↓
f''(x) ≥ 0          (1D)
    ↓
zᵀHz ≥ 0  ∀z      (multi-D)
    ↓
H ⪰ 0
```

### What to Remember

| # | Concept | Key Takeaway |
|---|---------|--------------|
| **1** | **Definition** | $f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta)f(x_2)$ — graph below chord |
| **2** | **1D Test** | $f''(x) \geq 0 \implies$ convex |
| **3** | **Multi-D Test** | $z^T H z \geq 0 \ \forall z \iff H \succeq 0 \implies$ convex |
| **4** | **Optimization** | Convexity $\not\implies$ minimum exists. But **if** minimum exists, it's **global** |

---

## 13. Problem Sets and Examples

### Problem Set 1: Definitions and Basic Properties

**Problem 1.1:** Prove that the sum of two convex functions is convex. Is the product of two convex functions necessarily convex? (Hint: consider $f(x)=x^2$, $g(x)=x^2$)

**Problem 1.2:** Let $f$ be convex and $g(x) = f(Ax + b)$ for some matrix $A$ and vector $b$. Prove that $g$ is convex.

**Problem 1.3:** Show that $f(x) = |x|$ is convex using the definition (not the second derivative, since it's not differentiable at 0).

**Problem 1.4:** Prove that if $f$ is convex, then for any $x_1, x_2, x_3$ and $\theta_1, \theta_2, \theta_3 \geq 0$ with $\sum \theta_i = 1$:
$$f(\theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3) \leq \theta_1 f(x_1) + \theta_2 f(x_2) + \theta_3 f(x_3)$$
(This is Jensen's inequality for 3 points; generalize to $n$ points.)

---

### Problem Set 2: Second Derivative and Hessian Tests

**Problem 2.1:** Determine convexity of the following 1D functions using $f''(x)$:
- (a) $f(x) = x^4$
- (b) $f(x) = \ln(x)$ for $x > 0$
- (c) $f(x) = e^x + e^{-x}$
- (d) $f(x) = \sin(x)$ on $[0, \pi]$
- (e) $f(x) = -\sqrt{x}$ for $x > 0$

**Problem 2.2:** For the function $f(x_1, x_2) = x_1^2 + 2x_1x_2 + 3x_2^2$:
- (a) Compute the Hessian
- (b) Compute $z^T H z$ for arbitrary $z = [z_1, z_2]^T$
- (c) Is the function convex? Strictly convex?
- (d) Find the global minimum

**Problem 2.3:** Consider $f(x_1, x_2) = x_1^2 - x_2^2$.
- (a) Compute the Hessian
- (b) Find directions $z$ where $z^T H z > 0$, $z^T H z < 0$, and $z^T H z = 0$
- (c) Is this function convex? Concave? Neither?
- (d) What is the geometric shape? (Hint: saddle point)

**Problem 2.4:** For $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ where $A$ is a symmetric matrix:
- (a) What is the Hessian?
- (b) Prove $f$ is convex iff $A \succeq 0$
- (c) Prove $f$ is strictly convex iff $A \succ 0$

---

### Problem Set 3: The $z^T H z$ Condition and Directional Curvature

**Problem 3.1:** Let $f(x_1, x_2) = x_1^4 + x_2^4$.
- (a) Compute the Hessian $H(x_1, x_2)$
- (b) Evaluate $z^T H z$ for arbitrary $z = [z_1, z_2]^T$
- (c) Is $f$ convex? (Note: Hessian depends on $x$!)

**Problem 3.2:** For $f(x_1, x_2) = (x_1 - x_2)^2$:
- (a) Compute the Hessian
- (b) Find all directions $z$ where $z^T H z = 0$
- (c) Is the function strictly convex? Why or why not?

**Problem 3.3:** Prove that for a quadratic function $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c$:
- The Hessian is constant (independent of $\mathbf{x}$)
- $f$ is convex iff $A \succeq 0$
- The condition $z^T H z \geq 0$ reduces to checking eigenvalues of $A$

**Problem 3.4:** Consider $f(x_1, x_2) = \frac{1}{2}(x_1^2 + \gamma x_2^2)$ for $\gamma > 0$.
- (a) What is the condition number of the Hessian?
- (b) How does the condition number affect gradient descent convergence?
- (c) What happens as $\gamma \to \infty$ or $\gamma \to 0$?

---

### Problem Set 4: Convexity vs. Existence of Minima

**Problem 4.1:** For each function below, state whether it is convex and whether it has a global minimum:
- (a) $f(x) = e^x$
- (b) $f(x) = e^{-x}$
- (c) $f(x) = x^2$
- (d) $f(x) = 1$
- (e) $f(x) = \max(0, x)$ (ReLU)

**Problem 4.2:** Prove that a strictly convex function on $\mathbb{R}^n$ can have **at most one** global minimum.

**Problem 4.3:** Give an example of a convex function that:
- (a) Has a unique global minimum
- (b) Has infinitely many global minima
- (c) Has no global minimum (only an infimum)
- (d) Is bounded below but has no minimum

**Problem 4.4:** Let $f(x) = -\log(x)$ for $x > 0$.
- (a) Is it convex?
- (b) Does it have a minimum on $(0, \infty)$?
- (c) What is its infimum?
- (d) What happens if we restrict to $x \in [1, \infty)$?

---

### Problem Set 5: Non-Convex Functions with Minima

**Problem 5.1:** For $f(x) = x^4 - x^2$:
- (a) Find all critical points
- (b) Classify each as local min, local max, or saddle
- (c) Find global minimum(s)
- (d) Show explicitly where $f''(x) < 0$ (non-convex region)

**Problem 5.2:** Consider $f(x) = \sin(x) + 0.1 x^2$.
- (a) How many local minima does it have?
- (b) Is it convex?
- (c) Does it have a unique global minimum?

**Problem 5.3:** Construct a function $f: \mathbb{R} \to \mathbb{R}$ that:
- Has exactly 3 local minima
- Has a unique global minimum
- Is differentiable everywhere

**Problem 5.4:** Let $f(x) = x^2$ for $x \leq 0$ and $f(x) = 2x^2$ for $x > 0$.
- (a) Is this function convex?
- (b) Is it strictly convex?
- (c) Is it differentiable at $x=0$?
- (d) Does it have a unique minimum?

---

### Problem Set 6: Operations Preserving Convexity

**Problem 6.1:** (Nonnegative weighted sum) Let $f_1, \ldots, f_m$ be convex and $w_1, \ldots, w_m \geq 0$. Prove $f = \sum w_i f_i$ is convex.

**Problem 6.2:** (Pointwise maximum) Let $f_1, \ldots, f_m$ be convex. Prove $f(x) = \max_i f_i(x)$ is convex.

**Problem 6.3:** (Affine transformation) Let $f$ be convex. Prove $g(x) = f(Ax + b)$ is convex.

**Problem 6.4:** (Composition with nondecreasing convex function) Let $f$ be convex and $g: \mathbb{R} \to \mathbb{R}$ be convex and nondecreasing. Prove $h(x) = g(f(x))$ is convex. Give an example where $g$ is convex but NOT nondecreasing, and $h$ is not convex.

---

### Problem Set 7: Worked Examples

**Example 1: Quadratic Form Convexity Check**

Function: $f(x_1, x_2) = 2x_1^2 + 4x_1x_2 + 5x_2^2$

1. Write in matrix form: $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$
2. Find matrix $A$
3. Compute eigenvalues of $A$
4. Determine convexity from eigenvalues
5. Find minimum

**Example 2: Logistic Loss Convexity**

Function: $f(w) = \log(1 + e^{-y w^T x})$ for fixed $x, y$ (where $y \in \{\pm 1\}$)

1. Compute first and second derivatives w.r.t $w$
2. Show $f''(w) \geq 0$ for all $w$
3. Conclude logistic loss is convex

**Example 3: Least Squares Objective**

Function: $E(\mathbf{w}) = \frac{1}{2} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$

1. Express as quadratic form
2. Compute Hessian
3. Show it's positive semidefinite
4. When is it positive definite?

**Example 4: Entropy Function**

Function: $H(\mathbf{p}) = -\sum_i p_i \log p_i$ for $\mathbf{p} \in \Delta^{n-1}$ (probability simplex)

1. Compute Hessian (diagonal matrix with entries $1/p_i$)
2. Show $z^T H z = \sum z_i^2 / p_i \geq 0$
3. Conclude entropy is convex (wait, negative entropy is convex!)

---

### Problem Set 8: Advanced Topics

**Problem 8.1:** (Strong Convexity) A function $f$ is $\mu$-strongly convex if $f(x) - \frac{\mu}{2}\|x\|^2$ is convex. Show that for a twice differentiable function, this is equivalent to $H \succeq \mu I$ (all eigenvalues $\geq \mu$).

**Problem 8.2:** (Convex Conjugate) The convex conjugate of $f$ is $f^*(y) = \sup_x \{y^T x - f(x)\}$. Compute the convex conjugate of:
- (a) $f(x) = \frac{1}{2}x^2$
- (b) $f(x) = e^x$
- (c) $f(x) = -\log x$ (for $x > 0$)

**Problem 8.3:** (Bregman Divergence) For a strictly convex differentiable $f$, the Bregman divergence is $D_f(x, y) = f(x) - f(y) - \nabla f(y)^T(x - y)$. Show $D_f(x, y) \geq 0$ with equality iff $x = y$.

**Problem 8.4:** (Gradient Descent on Convex Functions) For a convex function with $L$-Lipschitz gradient ($\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$), show that gradient descent with step size $1/L$ satisfies:
$$f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$$

---

## Key Formulas Reference

| Concept | Formula |
|---------|---------|
| **Convex Definition** | $f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta)f(x_2)$ |
| **1D Test** | $f''(x) \geq 0$ |
| **Gradient** | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^T$ |
| **Hessian** | $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ |
| **Curvature in direction $z$** | $z^T H z$ |
| **Convexity (multi-D)** | $H \succeq 0 \iff z^T H z \geq 0 \ \forall z$ |
| **Strict Convexity** | $H \succ 0 \iff z^T H z > 0 \ \forall z \neq 0$ |
| **Line restriction** | $g(t) = f(x + tz) \implies g''(t) = z^T H(x+tz)z$ |

---

## Navigation

| Previous | Next |
|----------|------|
| — | [Lecture 3-5: Linear Regression (Probabilistic View)]({{ site.baseurl }}/lectures/Lecture_3_5_Linear_Regression_Probabilistic/) |

*End of Lecture Notes*