---
layout: default
title: "Lecture 9: Logistic Regression"
nav_order: 5
parent: Lectures
has_toc: true
permalink: /lectures/Lecture_9_Logistic_Regression_Notes/
---

# Logistic Regression — Complete Derivation Notes

**Instructor:** Dr. Anirban Sen  
**Date:** October 9, 2026  

---

## Table of Contents

1. [Notation & Conventions](#1-notation--conventions)
2. [Binary Logistic Regression: What Does the Model Predict?](#2-binary-logistic-regression-what-does-the-model-predict)
3. [Likelihood of the Observed Label](#3-likelihood-of-the-observed-label)
4. [Maximum Likelihood & Log-Likelihood](#4-maximum-likelihood--log-likelihood)
5. [Parameter Count: Gaussian PGM vs. Logistic Regression](#5-parameter-count-gaussian-pgm-vs-logistic-regression)
6. [Linear Separability and the Overfitting Problem](#6-linear-separability-and-the-overfitting-problem)
7. [From Binary to Multiclass: the Softmax](#7-from-binary-to-multiclass-the-softmax)
8. [Softmax Reduces to Sigmoid (K = 2)](#8-softmax-reduces-to-sigmoid-k--2)
9. [Parameter Redundancy and the Reference Class](#9-parameter-redundancy-and-the-reference-class)
10. [Log-Odds Interpretation](#10-log-odds-interpretation)
11. [Decision Boundaries](#11-decision-boundaries)
12. [Multiclass Likelihood and Log-Likelihood](#12-multiclass-likelihood-and-log-likelihood)
13. [Substituting the Softmax into the Log-Likelihood](#13-substituting-the-softmax-into-the-log-likelihood)
14. [Gradient of the Log-Likelihood](#14-gradient-of-the-log-likelihood)
15. [The MLE Condition and Why There Is No Closed Form](#15-the-mle-condition-and-why-there-is-no-closed-form)
16. [Numerical Optimization (Gradient Ascent)](#16-numerical-optimization-gradient-ascent)
17. [Compact View of the Whole Derivation](#17-compact-view-of-the-whole-derivation)
18. [Summary](#18-summary)
19. [Practice Problems with Solutions](#19-practice-problems-with-solutions)

---

## 1. Notation & Conventions

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_n$ | $n$-th input vector, $\mathbf{x}\in\mathbb{R}^M$ |
| $\phi(\mathbf{x}_n)=\phi_n$ | feature vector (basis-function transform) of $\mathbf{x}_n$ |
| $C_1, C_2,\dots,C_K$ | the classes |
| $t_n$ (binary) | observed label: $1$ if $\mathbf{x}_n\in C_1$, $0$ if $\mathbf{x}_n\in C_2$ |
| $t_{nk}$ (multiclass) | one-hot target: $1$ if $\mathbf{x}_n\in C_k$, else $0$ |
| $y_n$ or $y_{nk}$ | the model's **predicted probability** |
| $\mathbf{w}$ or $\mathbf{w}_k$ | weight (parameter) vectors |
| $a$ or $a_k$ | "score" / activation, a linear function of features |
| $N$ | number of training examples |
| $M$ | input dimension |
| $K$ | number of classes |

> **Key distinction:** $y_n$ = what the model *predicts*; $t_n$ = what *actually happened*.

---

## 2. Binary Logistic Regression: What Does the Model Predict?

For binary classification, logistic regression models the posterior probability of class $C_1$:

$$
y_n = P(C_1\mid \mathbf{x}_n) = \sigma\!\left(\mathbf{w}^T\phi(\mathbf{x}_n)\right)
$$

where the **sigmoid (logistic) function** is

$$
\sigma(a)=\frac{1}{1+e^{-a}}
$$

The **training target** is encoded as

$$
t_n=\begin{cases}1, & \mathbf{x}_n\in C_1\\[2pt] 0, & \mathbf{x}_n\in C_2\end{cases}
$$

**Properties of the sigmoid** (used later):

- $\sigma(a)\in(0,1)$ for every real $a$ → a valid probability.
- $\sigma(a)\to 1$ as $a\to+\infty$, and $\sigma(a)\to0$ as $a\to-\infty$.
- $\sigma(0)=0.5$ → the decision boundary is $\mathbf{w}^T\phi(\mathbf{x})=0$.
- $\sigma(-a)=1-\sigma(a)$.

**Example.** Let $\mathbf{w}^T\phi(\mathbf{x}_n)=2$. Then $y_n=\dfrac{1}{1+e^{-2}}=\dfrac{1}{1+0.1353}\approx 0.881$, so the model says "88.1 % likely to be $C_1$."

---

## 3. Likelihood of the Observed Label

### 3.1 One training example

The probability the model assigns to the **observed** label is

$$
p(t_n\mid \mathbf{x}_n)=y_n^{\,t_n}(1-y_n)^{1-t_n}
$$

**Why does this work?** Check both cases:

- If $t_n=1$: $\;y_n^{1}(1-y_n)^{0}=y_n$
- If $t_n=0$: $\;y_n^{0}(1-y_n)^{1}=1-y_n$

So the formula automatically "selects" the probability of whichever class actually occurred.

### 3.2 All $N$ examples (independence)

For independent training examples, probabilities multiply:

$$
L(\mathbf{w})=\prod_{n=1}^{N} y_n^{\,t_n}(1-y_n)^{1-t_n}
$$

This is the **likelihood**: the probability the model assigns to the labels that actually occurred.

**Example.** Three points with predictions $y=(0.9,\,0.2,\,0.7)$ and labels $t=(1,0,1)$:

$$
L = 0.9\times(1-0.2)\times 0.7 = 0.9\times0.8\times0.7 = 0.504
$$

---

## 4. Maximum Likelihood & Log-Likelihood

**Maximum likelihood** chooses the parameters that make the observed labels as probable as possible:

$$
\mathbf{w}_{ML}=\arg\max_{\mathbf{w}} L(\mathbf{w})
$$

Since $\ln$ is monotonically increasing (same maximizer) and turns products into sums, maximize the **log-likelihood**:

$$
\ln L(\mathbf{w})=\sum_{n=1}^{N}\Big[t_n\ln y_n+(1-t_n)\ln(1-y_n)\Big]
$$

(The negative of this is the familiar **binary cross-entropy loss**.)

**Intuition**

- If the true label is $C_1$, we want $y_n=P(C_1\mid\mathbf{x}_n)$ close to **1**.
- If the true label is $C_2$, we want $1-y_n=P(C_2\mid\mathbf{x}_n)$ close to **1**.

**Example — continuing above:**

$$
\ln L=\ln0.9+\ln0.8+\ln0.7=-0.105-0.223-0.357=-0.685\quad(=\ln 0.504 ✓)
$$

---

## 5. Parameter Count: Gaussian PGM vs. Logistic Regression

### 5.1 Setup (Gaussian generative model)

Binary classification, input $\mathbf{x}\in\mathbb{R}^M$:

$$
\mathbf{x}\mid C_1\sim\mathcal{N}(\boldsymbol\mu_1,\Sigma),\qquad \mathbf{x}\mid C_2\sim\mathcal{N}(\boldsymbol\mu_2,\Sigma)
$$

(shared covariance $\Sigma$). Each mean is an $M$-vector:

$$
\boldsymbol\mu_k=\begin{bmatrix}\mu_{k1}\\ \mu_{k2}\\ \vdots\\ \mu_{kM}\end{bmatrix}\in\mathbb{R}^M
$$

### 5.2 Means

$$
\#\text{params}(\boldsymbol\mu_1)=M,\quad \#\text{params}(\boldsymbol\mu_2)=M\;\;\Rightarrow\;\;\boxed{2M}
$$

### 5.3 Covariance matrix — counting independent parameters

$\Sigma$ is $M\times M$:

$$
\Sigma=\begin{bmatrix}\sigma_{11}&\sigma_{12}&\cdots&\sigma_{1M}\\ \sigma_{21}&\sigma_{22}&\cdots&\sigma_{2M}\\ \vdots&\vdots&\ddots&\vdots\\ \sigma_{M1}&\sigma_{M2}&\cdots&\sigma_{MM}\end{bmatrix}
$$

A general $M\times M$ matrix has $M^2$ entries. **But** a covariance matrix is **symmetric**, $\sigma_{ij}=\sigma_{ji}$, so $\sigma_{12}$ and $\sigma_{21}$ are *one* parameter. We need only one triangle (including the diagonal).

**Step 1 — diagonal entries** $\sigma_{11},\dots,\sigma_{MM}$: there are $M$ variances.

**Step 2 — off-diagonal entries** (one per pair, above the diagonal):

$$
1+2+\cdots+(M-1)=\frac{M(M-1)}{2}
$$

**Step 3 — add them:**

$$
M+\frac{M(M-1)}{2}=\frac{2M+M^2-M}{2}=\boxed{\frac{M(M+1)}{2}}
$$

### 5.4 Example: $M=3$

$$
\Sigma=\begin{bmatrix}\sigma_{11}&\sigma_{12}&\sigma_{13}\\ \sigma_{12}&\sigma_{22}&\sigma_{23}\\ \sigma_{13}&\sigma_{23}&\sigma_{33}\end{bmatrix}
$$

Independent parameters: $\underbrace{\sigma_{11},\sigma_{22},\sigma_{33}}_{3\text{ variances}}$ and $\underbrace{\sigma_{12},\sigma_{13},\sigma_{23}}_{3\text{ covariances}}$.

$$
3+3=6\qquad\text{and}\qquad \frac{3(3+1)}{2}=\boxed{6}\;✓
$$

> **Key idea:** A symmetric $M\times M$ covariance matrix has $\dfrac{M(M+1)}{2}$ independent parameters.

### 5.5 Total comparison

| Model component | # parameters |
|---|---|
| Mean $\boldsymbol\mu_1$ | $M$ |
| Mean $\boldsymbol\mu_2$ | $M$ |
| Shared covariance $\Sigma$ | $\dfrac{M(M+1)}{2}$ |
| Class prior $\pi_1$ (then $\pi_2=1-\pi_1$) | $1$ |
| **Gaussian PGM total** | $\;2M+\dfrac{M(M+1)}{2}+1$ |

In contrast, logistic regression directly learns

$$
p(C_1\mid\mathbf{x})=\sigma(\mathbf{w}^T\mathbf{x}+w_0)
$$

with only

$$
\boxed{M+1}\ \text{parameters.}
$$

**How the gap grows:**

| $M$ | Gaussian PGM: $2M+\frac{M(M+1)}{2}+1$ | Logistic regression: $M+1$ |
|---|---|---|
| 3 | $6+6+1=13$ | 4 |
| 10 | $20+55+1=76$ | 11 |
| 100 | $200+5050+1=5251$ | 101 |

The generative model grows **quadratically** in $M$, logistic regression only **linearly**.

---

## 6. Linear Separability and the Overfitting Problem

### 6.1 What does "linearly separable" mean?

The decision boundary is $\mathbf{w}^T\phi(\mathbf{x})=0$. The data are **linearly separable** if there exists some $\mathbf{w}$ with

$$
\mathbf{w}^T\phi(\mathbf{x}_n)>0\ \text{for every } C_1 \text{ point},\qquad
\mathbf{w}^T\phi(\mathbf{x}_n)<0\ \text{for every } C_2 \text{ point}.
$$

In words: **one hyperplane can correctly separate all training points.**

### 6.2 Why does the weight magnitude matter?

Suppose the data are separable with a separating $\mathbf{w}$. Multiply by a constant $c>1$:

$$
\mathbf{w}'=c\,\mathbf{w}
$$

**The boundary does not move:**

$$
\mathbf{w}'^T\phi(\mathbf{x})=c\,\mathbf{w}^T\phi(\mathbf{x})=0\iff \mathbf{w}^T\phi(\mathbf{x})=0
$$

**But the scores grow in magnitude** (and keep their signs):

$$
C_1:\ \mathbf{w}'^T\phi(\mathbf{x})\to+\infty,\qquad C_2:\ \mathbf{w}'^T\phi(\mathbf{x})\to-\infty
$$

### 6.3 What happens to the probabilities?

Recall $y=\sigma(\mathbf{w}^T\phi(\mathbf{x}))$, and $\sigma(a)\to1$ as $a\to+\infty$, $\sigma(a)\to0$ as $a\to-\infty$. Therefore, for separable data as the scale grows:

$$
P(C_1\mid\mathbf{x})\to1\ \text{ for every } C_1\text{ point},\qquad
P(C_1\mid\mathbf{x})\to0\ \text{ for every } C_2\text{ point}.
$$

Every training example gets probability approaching **1 for its correct label**. Hence the likelihood **keeps increasing** as

$$
\|\mathbf{w}\|\to\infty.
$$

### 6.4 The problem

Increasing $\|\mathbf{w}\|$ keeps every training point correctly classified, so the likelihood keeps rising even though classification is already perfect:

- **There is no finite maximum-likelihood solution:** $\|\mathbf{w}\|\to\infty$.
- The sigmoid becomes extremely **steep**.
- The model becomes almost **infinitely confident**.
- Predictions near the decision boundary become highly **sensitive** to small input changes.

> **Perfect training classification, but no finite MLE.**

**Worked example (1-D, separable).** Data: $x=+1\in C_1$, $x=-1\in C_2$. Model: $y=\sigma(wx)$.

- Point 1 ($t=1$): probability $=\sigma(w)$.
- Point 2 ($t=0$): probability $=1-\sigma(-w)=\sigma(w)$.

$$
L(w)=\sigma(w)^2\;\longrightarrow\;1\ \text{ as } w\to\infty,\ \text{but never reaches it.}
$$

| $w$ | $\sigma(w)$ | $L(w)$ |
|---|---|---|
| 1 | 0.731 | 0.534 |
| 3 | 0.953 | 0.907 |
| 5 | 0.9933 | 0.9866 |
| 10 | 0.99995 | 0.9999 |

$L$ is strictly increasing in $w$ → there is **no maximizer**.

### 6.5 Why is linear separability necessary for this behaviour?

Suppose the data are **not** linearly separable. Then no hyperplane classifies every training point correctly: at least one point is on the wrong side.

For example, take a $C_1$ point with $\mathbf{w}^T\phi(\mathbf{x})<0$. If we increase the magnitude of $\mathbf{w}$:

$$
\mathbf{w}^T\phi(\mathbf{x})\to-\infty\;\Rightarrow\;P(C_1\mid\mathbf{x})=\sigma(\mathbf{w}^T\phi(\mathbf{x}))\to0
$$

But the point **actually belongs to $C_1$**, so its likelihood contribution $\to0$, which makes the total likelihood **worse, not better**. So increasing $\|\mathbf{w}\|$ cannot keep helping; the likelihood has a finite maximizer.

**Worked example (1-D, not separable).** Two points both at $x=1$: one labelled $C_1$, one labelled $C_2$.

$$
L(w)=\sigma(w)\,[1-\sigma(w)]
$$

This is maximized at $w=0$ with $L=0.25$, and $L\to0$ as $|w|\to\infty$. A **finite** MLE exists.

### 6.6 Separable vs. non-separable summary

| | Linearly separable | Not separable |
|---|---|---|
| All training points can be correct? | Yes | No |
| Increase $\|\mathbf{w}\|$ | Helps all points | Hurts some point |
| Probabilities $\to 0,1$ | All can | Cannot all |
| MLE solution | $\|\mathbf{w}\|\to\infty$ | Finite $\mathbf{w}$ |

> **Key reason:** Separability lets us make every training prediction increasingly confident *and still correct*. Without separability, increasing confidence eventually makes at least one training prediction **confidently wrong**.

### 6.7 The overfitting problem

$$
\text{separability}\;\Rightarrow\;\text{likelihood keeps increasing}\;\Rightarrow\;\|\mathbf{w}\|\to\infty
$$

- The **direction** of $\mathbf{w}$ determines the decision boundary: replacing $\mathbf{w}$ by $c\mathbf{w}$ gives the same boundary $(c\mathbf{w})^T\phi(\mathbf{x})=0$.
- The **magnitude** of $\mathbf{w}$ only controls how confident (steep) the sigmoid is.

This is why maximum-likelihood logistic regression can suffer **severe overfitting** on linearly separable data.

*(Standard remedy: add a regularizer such as $-\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the log-likelihood, which penalizes large weights and gives a finite solution.)*

---

## 7. From Binary to Multiclass: the Softmax

### 7.1 Goal

Binary case: $P(C_1\mid\mathbf{x})=\sigma(a)=\dfrac{1}{1+e^{-a}}$ with $a=\mathbf{w}^T\phi(\mathbf{x})$.

For $K>2$ classes we need $P(C_k\mid\mathbf{x})$, $k=1,\dots,K$, satisfying

$$
\sum_{k=1}^{K}P(C_k\mid\mathbf{x})=1\qquad\text{and}\qquad P(C_k\mid\mathbf{x})\ge0.
$$

We need a **multiclass generalization of the sigmoid**.

### 7.2 Class-dependent linear functions

Give each class its own linear score:

$$
a_k=\mathbf{w}_k^T\phi(\mathbf{x}),\qquad k=1,\dots,K
$$

where $\mathbf{w}_k$ is the parameter vector of class $C_k$. This yields $K$ real numbers $a_1,\dots,a_K$ that we must convert into $K$ probabilities with $0\le P(C_k\mid\mathbf{x})\le1$ and $\sum_kP(C_k\mid\mathbf{x})=1$.

### 7.3 Construction of the softmax

**Step 1 — exponentiate** (makes everything positive): $e^{a_1},e^{a_2},\dots,e^{a_K}$.

**Step 2 — normalize** by the sum:

$$
\boxed{P(C_k\mid\mathbf{x})=\frac{e^{a_k}}{\sum_{j=1}^{K}e^{a_j}}}\qquad\text{(softmax)}
$$

**Check 1 (non-negative):** every $e^{a_k}>0$ ⇒ $P(C_k\mid\mathbf{x})\ge0$.

**Check 2 (sums to 1):**

$$
\sum_{k=1}^{K}P(C_k\mid\mathbf{x})=\sum_{k=1}^{K}\frac{e^{a_k}}{\sum_je^{a_j}}=\frac{\sum_ke^{a_k}}{\sum_je^{a_j}}=1
$$

So the outputs form a valid probability distribution.

**Numerical example ($K=3$).** Scores $\mathbf{a}=(2,1,0)$:

$$
e^2=7.389,\; e^1=2.718,\; e^0=1,\qquad \text{sum}=11.107
$$

$$
P=\left(\tfrac{7.389}{11.107},\tfrac{2.718}{11.107},\tfrac{1}{11.107}\right)=(0.665,\;0.245,\;0.090)
$$

Sum $=1.000$ ✓. The largest score gets the largest probability.

---

## 8. Softmax Reduces to Sigmoid ($K=2$)

For two classes:

$$
P(C_1\mid\mathbf{x})=\frac{e^{a_1}}{e^{a_1}+e^{a_2}}
$$

Divide numerator and denominator by $e^{a_2}$:

$$
P(C_1\mid\mathbf{x})=\frac{e^{a_1-a_2}}{e^{a_1-a_2}+1}
$$

Divide numerator and denominator by $e^{a_1-a_2}$:

$$
P(C_1\mid\mathbf{x})=\frac{1}{1+e^{-(a_1-a_2)}}=\sigma(a_1-a_2)
$$

Hence **the binary sigmoid is a special case of softmax**, applied to the score difference $a_1-a_2$.

---

## 9. Parameter Redundancy and the Reference Class

### 9.1 The redundancy

Add the same function $c(\mathbf{x})$ to every score: $a_k'=a_k+c(\mathbf{x})$. Then

$$
P(C_k\mid\mathbf{x})=\frac{e^{a_k+c(\mathbf{x})}}{\sum_je^{a_j+c(\mathbf{x})}}
=\frac{e^{c(\mathbf{x})}\,e^{a_k}}{e^{c(\mathbf{x})}\sum_je^{a_j}}
=\frac{e^{a_k}}{\sum_je^{a_j}}
$$

The probabilities are **unchanged**. So $(a_1,\dots,a_K)$ and $(a_1+c,\dots,a_K+c)$ represent *exactly the same* probabilities ⇒ the model has **redundant parameters**.

**Check.** $\mathbf{a}=(3,2,1)$ is $(2,1,0)$ shifted by $c=1$:

$$
\frac{e^3}{e^3+e^2+e^1}=\frac{20.086}{30.193}=0.665
$$

— identical to the earlier result ✓.

### 9.2 Removing the redundancy

Choose one class as a **reference class**, say $C_K$, and set

$$
\mathbf{w}_K=\mathbf{0}\;\Rightarrow\;a_K=0.
$$

Then for $k=1,\dots,K-1$:

$$
P(C_k\mid\mathbf{x})=\frac{e^{a_k}}{1+\sum_{j=1}^{K-1}e^{a_j}},\qquad
P(C_K\mid\mathbf{x})=\frac{1}{1+\sum_{j=1}^{K-1}e^{a_j}}
$$

(because $e^{a_K}=e^0=1$). This leaves only **$K-1$ independent parameter vectors**.

---

## 10. Log-Odds Interpretation

Using the reference class $C_K$ from §9.2, divide the two expressions:

$$
\frac{P(C_k\mid\mathbf{x})}{P(C_K\mid\mathbf{x})}=e^{a_k}
$$

Take logarithms:

$$
\ln\frac{P(C_k\mid\mathbf{x})}{P(C_K\mid\mathbf{x})}=a_k=\mathbf{w}_k^T\phi(\mathbf{x})
$$

> Multiclass logistic regression models the **log posterior odds relative to a reference class** as a **linear function** of the input.

**Example.** With $\mathbf{a}=(2,1,0)$ and reference $C_3$: $\ln\frac{P(C_1)}{P(C_3)}=2$, so $\frac{P(C_1)}{P(C_3)}=e^2=7.39$. Check: $0.665/0.090=7.39$ ✓.

---

## 11. Decision Boundaries

The most probable class is

$$
\hat C=\arg\max_kP(C_k\mid\mathbf{x}).
$$

The softmax denominator is the same for all classes, and $P(C_k\mid\mathbf{x})\propto e^{a_k}$ with $e^{(\cdot)}$ increasing, so

$$
\hat C=\arg\max_ka_k.
$$

**Boundary between $C_k$ and $C_j$:** where the probabilities are equal.

$$
P(C_k\mid\mathbf{x})=P(C_j\mid\mathbf{x})\;\Rightarrow\;e^{a_k}=e^{a_j}\;\Rightarrow\;a_k=a_j
$$

Substituting $a_k=\mathbf{w}_k^T\phi(\mathbf{x})$:

$$
\boxed{(\mathbf{w}_k-\mathbf{w}_j)^T\phi(\mathbf{x})=0}
$$

So the **pairwise decision boundaries are linear in feature space.**

**Example.** $\phi=(1,x_1,x_2)^T$, $\mathbf{w}_1=(0,1,0)^T$, $\mathbf{w}_2=(0,0,1)^T$. Boundary between $C_1$ and $C_2$: $(\mathbf{w}_1-\mathbf{w}_2)^T\phi=x_1-x_2=0$ → the line $x_1=x_2$. Class 1 wins when $x_1>x_2$.

---

## 12. Multiclass Likelihood and Log-Likelihood

### 12.1 One-hot targets

$$
t_{nk}=\begin{cases}1,&\mathbf{x}_n\in C_k\\0,&\text{otherwise}\end{cases}
$$

Example ($K=3$): class $C_2\Rightarrow(t_{n1},t_{n2},t_{n3})=(0,1,0)$.

The model predicts $y_{nk}=P(C_k\mid\mathbf{x}_n)$.

### 12.2 One observation

$$
p(\mathbf{t}_n\mid\mathbf{x}_n)=\prod_{k=1}^{K}y_{nk}^{\,t_{nk}}
$$

Only the observed class has $t_{nk}=1$; all other factors are $y^0=1$. So this simply picks out $y_{nk}$ for the true class (the multiclass analogue of $y^t(1-y)^{1-t}$).

### 12.3 All $N$ observations

$$
L=\prod_{n=1}^{N}\prod_{k=1}^{K}y_{nk}^{\,t_{nk}}
$$

### 12.4 Log-likelihood

$$
\ln L=\ln\Big[\prod_{n,k}y_{nk}^{\,t_{nk}}\Big]=\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\ln y_{nk}
$$

We want

$$
\mathbf{w}_{ML}=\arg\max_{\mathbf{w}}\ln L.
$$

(The negative is the **categorical cross-entropy**.)

### 12.5 The multiclass model, written out

With $a_{nk}=\mathbf{w}_k^T\phi_n$ and $\phi_n=\phi(\mathbf{x}_n)$:

$$
y_{nk}=P(C_k\mid\mathbf{x}_n)=\frac{\exp(a_{nk})}{\sum_{j=1}^{K}\exp(a_{nj})}=\frac{\exp(\mathbf{w}_k^T\phi_n)}{\sum_{j=1}^{K}\exp(\mathbf{w}_j^T\phi_n)}
$$

Parameters to estimate: $\mathbf{w}_1,\dots,\mathbf{w}_K$.

**Example.** Two training points, $K=3$:

| $n$ | true class | $\mathbf{y}_n$ |
|---|---|---|
| 1 | $C_1$ | $(0.665, 0.245, 0.090)$ |
| 2 | $C_3$ | $(0.2, 0.3, 0.5)$ |

$$
\ln L=\ln0.665+\ln0.5=-0.408-0.693=-1.101
$$

---

## 13. Substituting the Softmax into the Log-Likelihood

**Step 1 — log of the softmax.** From $y_{nk}=\dfrac{e^{a_{nk}}}{\sum_je^{a_{nj}}}$ and $\ln(a/b)=\ln a-\ln b$:

$$
\ln y_{nk}=a_{nk}-\ln\Big(\sum_je^{a_{nj}}\Big)
$$

**Step 2 — insert $a_{nk}=\mathbf{w}_k^T\phi_n$:**

$$
\ln y_{nk}=\mathbf{w}_k^T\phi_n-\ln\Big[\sum_je^{\mathbf{w}_j^T\phi_n}\Big]
$$

**Step 3 — plug into $\ln L=\sum_{n,k}t_{nk}\ln y_{nk}$:**

$$
\ln L=\sum_{n,k}t_{nk}\Big[\mathbf{w}_k^T\phi_n-\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)\Big]
$$

**Step 4 — split into two sums:**

$$
\ln L=\sum_{n,k}t_{nk}\mathbf{w}_k^T\phi_n-\sum_{n,k}t_{nk}\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)
$$

**Step 5 — use $\sum_kt_{nk}=1$** (exactly one class is observed per $n$). The log-sum-exp term does not depend on $k$, so it factors out of the $k$-sum:

$$
\sum_{n,k}t_{nk}\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)=\sum_n\Big(\underbrace{\sum_kt_{nk}}_{=1}\Big)\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)=\sum_n\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)
$$

Therefore:

$$
\boxed{\ln L=\sum_{n,k}t_{nk}\mathbf{w}_k^T\phi_n-\sum_{n}\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)}
$$

---

## 14. Gradient of the Log-Likelihood

We differentiate with respect to $\mathbf{w}_k$:

$$
\frac{\partial\ln L}{\partial\mathbf{w}_k}=\frac{\partial}{\partial\mathbf{w}_k}\Big[\sum_{n,j}t_{nj}\mathbf{w}_j^T\phi_n-\sum_n\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)\Big]
$$

### 14.1 First term

Only the $j=k$ term depends on $\mathbf{w}_k$, and $\frac{\partial}{\partial\mathbf{w}_k}\mathbf{w}_k^T\phi_n=\phi_n$:

$$
\frac{\partial}{\partial\mathbf{w}_k}\sum_{n,j}t_{nj}\mathbf{w}_j^T\phi_n=\sum_nt_{nk}\phi_n
$$

### 14.2 Second term (log-sum-exp)

Use $\frac{d}{dx}\ln f(x)=\frac{1}{f(x)}\frac{df(x)}{dx}$. Inside the sum, only $j=k$ depends on $\mathbf{w}_k$:

$$
\frac{\partial}{\partial\mathbf{w}_k}\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)=\frac{e^{\mathbf{w}_k^T\phi_n}\,\phi_n}{\sum_je^{\mathbf{w}_j^T\phi_n}}
$$

The fraction $\dfrac{e^{\mathbf{w}_k^T\phi_n}}{\sum_je^{\mathbf{w}_j^T\phi_n}}$ is exactly the softmax probability $y_{nk}$, so

$$
\frac{\partial}{\partial\mathbf{w}_k}\ln\Big(\sum_je^{\mathbf{w}_j^T\phi_n}\Big)=y_{nk}\,\phi_n
$$

### 14.3 Combine

$$
\frac{\partial\ln L}{\partial\mathbf{w}_k}=\sum_nt_{nk}\phi_n-\sum_ny_{nk}\phi_n
$$

$$
\boxed{\frac{\partial\ln L}{\partial\mathbf{w}_k}=\sum_{n=1}^{N}(t_{nk}-y_{nk})\,\phi_n}
$$

> **Interpretation:** gradient $=$ (observed target $-$ model prediction), weighted by the feature vector.

**Binary special case.** With $K=2$ (or directly from the binary log-likelihood using $\sigma'(a)=\sigma(a)(1-\sigma(a))$) the gradient becomes $\nabla_{\mathbf{w}}\ln L=\sum_n(t_n-y_n)\phi_n$ — the same form.

**Numerical example.** One data point, $K=3$, $\phi=(1,2)^T$, true class $C_2$ so $\mathbf{t}=(0,1,0)$, current prediction $\mathbf{y}=(0.665,0.245,0.090)$:

| class | $t-y$ | gradient $(t-y)\phi$ |
|---|---|---|
| $\mathbf{w}_1$ | $-0.665$ | $(-0.665,\,-1.330)$ |
| $\mathbf{w}_2$ | $+0.755$ | $(+0.755,\,+1.510)$ |
| $\mathbf{w}_3$ | $-0.090$ | $(-0.090,\,-0.180)$ |

Gradient ascent pushes $\mathbf{w}_2$ *toward* $\phi$ (raising $P(C_2)$) and pushes the others *away*. Note the three gradients sum to $\mathbf{0}$ because $\sum_kt_k=\sum_ky_k=1$.

---

## 15. The MLE Condition and Why There Is No Closed Form

### 15.1 MLE condition

At the maximum, $\dfrac{\partial\ln L}{\partial\mathbf{w}_k}=0$, i.e. for each class $k$:

$$
\sum_{n=1}^{N}(t_{nk}-y_{nk})\phi_n=\mathbf{0}
\quad\Longleftrightarrow\quad
\sum_nt_{nk}\phi_n=\sum_ny_{nk}\phi_n
$$

- **Left side:** feature contribution from the **observed** class labels.
- **Right side:** feature contribution **predicted by the model**.

At the optimum, the model's expected feature totals match the observed ones.

**Insight.** If $\phi$ contains a constant bias component $1$, this condition includes $\sum_nt_{nk}=\sum_ny_{nk}$: the model's predicted total probability for each class equals the actual number of examples in that class.

### 15.2 Can we solve for $\mathbf{w}_k$ directly?

For **Gaussian generative models**, the MLE equations can be rearranged into closed form, e.g.

$$
\boldsymbol\mu_k=\frac{\sum_nt_{nk}\,\mathbf{x}_n}{\sum_nt_{nk}}
$$

(the class-$k$ sample mean).

For **logistic regression**,

$$
y_{nk}=\frac{e^{\mathbf{w}_k^T\phi_n}}{\sum_je^{\mathbf{w}_j^T\phi_n}}
$$

depends **nonlinearly on all** $\mathbf{w}_j$. So

$$
\sum_n(t_{nk}-y_{nk})\phi_n=\mathbf{0}
$$

cannot generally be rearranged to isolate $\mathbf{w}_k$. **The MLE parameters must be found numerically.**

---

## 16. Numerical Optimization (Gradient Ascent)

Maximize $\ln L$ iteratively. Start from initial parameters $\mathbf{w}_k^{(0)}$. At each iteration compute the gradient

$$
\mathbf{g}_k=\sum_n(t_{nk}-y_{nk})\phi_n
$$

and update

$$
\mathbf{w}_k^{(\text{new})}=\mathbf{w}_k^{(\text{old})}+\eta\,\mathbf{g}_k,\qquad\eta>0\ \text{(learning rate)}
$$

**Each iteration:**

$$
\mathbf{w}\;\longrightarrow\;y_{nk}\;\longrightarrow\;\text{gradient}\;\longrightarrow\;\mathbf{w}^{\text{new}}
$$

Repeat until the log-likelihood converges.

**One update.** Continue the example of §14 with $\eta=0.1$ and $\mathbf{w}_2^{(\text{old})}=\mathbf{0}$:

$$
\mathbf{w}_2^{(\text{new})}=\mathbf{0}+0.1\,(0.755,\,1.510)=(0.0755,\,0.151)
$$

**Pseudo-code**

```python
w = init_weights()                  # shape (K, D)
for it in range(max_iters):
    A = Phi @ w.T                   # scores a_nk, shape (N, K)
    A -= A.max(axis=1, keepdims=True)   # numerical stability (shift invariance!)
    Y = np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)   # softmax
    G = (T - Y).T @ Phi             # g_k = sum_n (t_nk - y_nk) phi_n
    w = w + eta * G
```

(The max-subtraction trick works precisely because of the shift-invariance property in §9.1.)

---

## 17. Compact View of the Whole Derivation

$$
y_{nk}=\frac{e^{\mathbf{w}_k^T\phi_n}}{\sum_je^{\mathbf{w}_j^T\phi_n}}
\;\;\Downarrow\;\;
\ln L=\sum_{n,k}t_{nk}\ln y_{nk}
\;\;\Downarrow\;\;
\frac{\partial\ln L}{\partial\mathbf{w}_k}=\sum_n(t_{nk}-y_{nk})\phi_n
$$

$$
\Downarrow\quad\sum_n(t_{nk}-y_{nk})\phi_n=\mathbf{0}
\quad\Downarrow\quad
\textbf{No closed-form solution}\Rightarrow\textbf{numerical optimization}
$$

---

## 18. Summary

For $K$ classes:

$$
a_k=\mathbf{w}_k^T\phi(\mathbf{x}),\qquad P(C_k\mid\mathbf{x})=\frac{e^{a_k}}{\sum_je^{a_j}}\ \ \text{(softmax)}
$$

- **Predicted class:** $\hat C=\arg\max_ka_k$
- **Boundary between $C_k$ and $C_j$:** $a_k=a_j$, i.e. $(\mathbf{w}_k-\mathbf{w}_j)^T\phi(\mathbf{x})=0$
- **Log-likelihood:** $\ln L=\sum_{n=1}^N\sum_{k=1}^Kt_{nk}\ln P(C_k\mid\mathbf{x}_n)$
- **Gradient:** $\sum_n(t_{nk}-y_{nk})\phi_n$ — solved numerically
- **Binary case:** sigmoid $=$ softmax with $K=2$
- **Redundancy:** only $K-1$ weight vectors needed (reference class)
- **Parameter efficiency:** $M+1$ (logistic) vs. $2M+\frac{M(M+1)}{2}+1$ (Gaussian PGM)
- **Pitfall:** on linearly separable data, the MLE has $\|\mathbf{w}\|\to\infty$ → overfitting

---

## 19. Practice Problems with Solutions

### Problem 1 — Sigmoid & likelihood
A logistic regression model gives scores $\mathbf{w}^T\phi_n=(1,\,-1,\,0)$ for three points with labels $t=(1,0,1)$. Compute $y_n$, $L$, and $\ln L$.

<details><summary>Solution</summary>

$y_1=\sigma(1)=0.731$, $y_2=\sigma(-1)=0.269$, $y_3=\sigma(0)=0.5$.

Probabilities of observed labels: $0.731$, $1-0.269=0.731$, $0.5$.

$L=0.731\times0.731\times0.5=0.267$, $\ln L=-1.32$.
</details>

### Problem 2 — Counting parameters
For $M=20$ features, how many parameters does (a) the shared-covariance Gaussian PGM need, (b) logistic regression?

<details><summary>Solution</summary>

(a) $2(20)+\frac{20\cdot21}{2}+1=40+210+1=251$.
(b) $20+1=21$.
</details>

### Problem 3 — Separability
Data in 1-D: $x=-2,-1\in C_2$ and $x=1,3\in C_1$. Is the model $y=\sigma(wx)$ well-behaved under MLE as $w\to\infty$? What if we add a point $x=2\in C_2$?

<details><summary>Solution</summary>

Original data: separable at $x=0$ ($w>0$). Every point's probability $\to1$ as $w\to\infty$, so $L\to1$ with no finite maximizer.

After adding $x=2\in C_2$: no threshold separates $\{1,3\}$ (class $C_1$) from $\{-2,-1,2\}$ (class $C_2$) using a boundary at 0 — and no 1-D threshold works at all, because $2$ lies between $1$ and $3$. The data is **not separable**, so the likelihood eventually decreases as $|w|\to\infty$ and a finite MLE exists.
</details>

### Problem 4 — Softmax
Compute the softmax of $\mathbf{a}=(1,\,1,\,1,\,1)$ and of $\mathbf{a}=(0,\,\ln2,\,\ln3)$.

<details><summary>Solution</summary>

First: all equal ⇒ $(0.25,0.25,0.25,0.25)$.

Second: $e^{a}=(1,2,3)$, sum $=6$ ⇒ $(\tfrac16,\tfrac13,\tfrac12)$.
</details>

### Problem 5 — Shift invariance
Show that softmax of $(1000,\,1001)$ equals softmax of $(0,\,1)$, and compute it.

<details><summary>Solution</summary>

Subtract $c=1000$ from both (§9.1): $(0,1)$. Then $P=\left(\frac{1}{1+e},\frac{e}{1+e}\right)=(0.269,\,0.731)$. This is also $\sigma(\mp1)$ — the binary case from §8.
</details>

### Problem 6 — Reference class
For $K=3$ with $C_3$ as the reference, $a_1=\ln 4$, $a_2=\ln 2$. Find all three probabilities and verify the log-odds relation.

<details><summary>Solution</summary>

$e^{a_1}=4$, $e^{a_2}=2$, denominator $=1+4+2=7$.

$P=(\tfrac47,\tfrac27,\tfrac17)$.

Log odds: $\ln\frac{4/7}{1/7}=\ln4=a_1$ ✓.
</details>

### Problem 7 — Decision boundary
Using $\phi=(1,x)^T$, $\mathbf{w}_1=(0,2)^T$, $\mathbf{w}_2=(3,0)^T$. Where is the boundary between $C_1$ and $C_2$, and which class wins for $x=2$?

<details><summary>Solution</summary>

$(\mathbf{w}_1-\mathbf{w}_2)^T\phi=(-3,2)\cdot(1,x)=-3+2x=0\Rightarrow x=1.5$.

At $x=2$: $a_1=4$, $a_2=3$ ⇒ $C_1$ wins.
</details>

### Problem 8 — Gradient step
One sample with $\phi=(1,\,3)^T$, true class $C_1$ out of $K=2$, current $\mathbf{y}=(0.4,0.6)$. Compute $\mathbf{g}_1$, $\mathbf{g}_2$, and the updated $\mathbf{w}_1$ from $\mathbf{w}_1=\mathbf{0}$ with $\eta=0.5$.

<details><summary>Solution</summary>

$\mathbf{t}=(1,0)$. $\mathbf{g}_1=(1-0.4)\phi=(0.6,\,1.8)$; $\mathbf{g}_2=(0-0.6)\phi=(-0.6,\,-1.8)$.

$\mathbf{w}_1^{\text{new}}=0.5\,(0.6,1.8)=(0.3,\,0.9)$.
</details>

### Problem 9 — Derivation drill
Starting from $\ln L=\sum_{n,k}t_{nk}\ln y_{nk}$ with softmax outputs, re-derive $\frac{\partial\ln L}{\partial\mathbf{w}_k}=\sum_n(t_{nk}-y_{nk})\phi_n$ without looking at §13–14. *(Hint: expand $\ln y_{nk}$, use $\sum_kt_{nk}=1$, then differentiate the log-sum-exp.)*

### Problem 10 — Conceptual
(a) Why does scaling $\mathbf{w}\to c\mathbf{w}$ change confidence but not the decision boundary?
(b) Why can a Gaussian generative model be fit in closed form while logistic regression cannot?
(c) Why does logistic regression need only $K-1$ weight vectors?

<details><summary>Solution</summary>

(a) The boundary is the set $\mathbf{w}^T\phi=0$, unchanged by positive scaling; but $\sigma(c\,a)$ is steeper for larger $c$.
(b) Gaussian MLE equations decouple into sample means/covariances; softmax couples all $\mathbf{w}_j$ nonlinearly.
(c) Softmax is invariant to adding a common term to all scores, so one class's weights can be fixed at $\mathbf{0}$.
</details>

---

*End of notes.*