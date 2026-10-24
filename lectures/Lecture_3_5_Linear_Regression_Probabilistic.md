---
layout: default
title: "Lecture 3-5: Linear Regression (Probabilistic View)"
nav_order: 2
parent: Lectures
has_toc: true
permalink: /lectures/Lecture_3_5_Linear_Regression_Probabilistic/
---

# Lecture 3-5: Linear Regression – A Probabilistic Interpretation

**Instructor:** Dr. Anirban Sen  
**Institution:** Ashoka University  
**Date:** September 10, 2026  

---

## Table of Contents

1. [The Basic Idea of Linear Regression](#1-the-basic-idea-of-linear-regression)
2. [Why "Linear" Regression?](#2-why-linear-regression)
3. [Gaussian Noise Model](#3-gaussian-noise-model)
4. [Likelihood and Log-Likelihood](#4-likelihood-and-log-likelihood)
5. [Maximum Likelihood and Least Squares Equivalence](#5-maximum-likelihood-and-least-squares-equivalence)
6. [Minimizing the Error: Normal Equations](#6-minimizing-the-error-normal-equations)
7. [Maximum Likelihood Estimate of β](#7-maximum-likelihood-estimate-of-β)
8. [Assumptions of Simple Linear Regression](#8-assumptions-of-simple-linear-regression)
9. [Problem Sets and Examples](#9-problem-sets-and-examples)

---

## 1. The Basic Idea of Linear Regression

### 1.1 Training Data

Suppose we observe training data:
$$\mathcal{D} = \{(x_1, t_1), (x_2, t_2), \ldots, (x_N, t_N)\}$$

- Each $x_i$ is an **input** (can be a vector)
- Each $t_i$ is the corresponding **target value**
- Goal: Find a function that predicts $t$ from $x$

### 1.2 Linear Regression Model

For linear regression, we write:
$$y(\mathbf{x}, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$$

where the **feature vector** is:
$$\boldsymbol{\phi}(\mathbf{x}) = \begin{bmatrix} \phi_0(\mathbf{x}) \\ \phi_1(\mathbf{x}) \\ \vdots \\ \phi_{M-1}(\mathbf{x}) \end{bmatrix}$$

and the **weight vector** is:
$$\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_{M-1} \end{bmatrix}$$

Thus:
$$y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x})$$

---

## 2. Why "Linear" Regression?

### 2.1 Key Point

> The model is called **linear** because it is **linear in its parameters** $\mathbf{w}$, not necessarily in the input $\mathbf{x}$.

### 2.2 Examples

**Example 1: Polynomial regression**
$$\boldsymbol{\phi}(x) = \begin{bmatrix} 1 \\ x \\ x^2 \end{bmatrix} \implies y(x, \mathbf{w}) = w_0 + w_1 x + w_2 x^2$$
- Nonlinear in $x$
- **Linear in $\mathbf{w}$**

**Example 2: Simple linear regression**
$$\boldsymbol{\phi}(x) = \begin{bmatrix} 1 \\ x \end{bmatrix} \implies y = w_0 + w_1 x$$
- Straight line
- $w_0$ = intercept, $w_1$ = slope

**Example 3: Basis functions**
$$\boldsymbol{\phi}(\mathbf{x}) = \begin{bmatrix} 1 \\ \phi_1(\mathbf{x}) \\ \phi_2(\mathbf{x}) \\ \vdots \end{bmatrix}$$
Can use any basis: polynomials, radial basis functions, Fourier features, etc.

### 2.3 Geometric Meaning (Straight Line)

For $y = w_0 + w_1 x$:
- $w_0$ = **intercept** (where line crosses y-axis)
- $w_1$ = **slope** = $\frac{\Delta y}{\Delta x}$

$$\mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix} \iff \text{intercept and slope of the regression line}$$

Finding the regression line $\iff$ Finding the weight vector $\mathbf{w}$.

---

## 3. Gaussian Noise Model

### 3.1 Generative Assumption

Assume the observed target is generated as:
$$t = y(\mathbf{x}, \mathbf{w}) + \epsilon$$

where $\epsilon$ is **Gaussian noise**:
$$\epsilon \sim \mathcal{N}(0, \beta^{-1})$$

$\beta$ is the **precision** (inverse variance): $\beta = \frac{1}{\sigma^2}$

### 3.2 Probabilistic Interpretation

Equivalently:
$$p(t | \mathbf{x}, \mathbf{w}, \beta) = \mathcal{N}\big(t \mid y(\mathbf{x}, \mathbf{w}), \beta^{-1}\big)$$

**Interpretation:**
- $y(\mathbf{x}, \mathbf{w})$: Model prediction (mean of distribution)
- $t$: Observed target value
- $t - y(\mathbf{x}, \mathbf{w})$: Difference modeled as Gaussian noise
- Noise has mean 0 and variance $\beta^{-1}$

**Observed Target = Prediction + Noise**

### 3.3 Derivation of Mean and Variance

**Mean:**
$$\mathbb{E}[t | \mathbf{x}] = \mathbb{E}[y(\mathbf{x}, \mathbf{w}) + \epsilon | \mathbf{x}] = \mathbb{E}[y(\mathbf{x}, \mathbf{w}) | \mathbf{x}] + \mathbb{E}[\epsilon | \mathbf{x}] = y(\mathbf{x}, \mathbf{w}) + 0 = y(\mathbf{x}, \mathbf{w})$$

**Variance:**
$$\text{Var}(t | \mathbf{x}) = \text{Var}(y(\mathbf{x}, \mathbf{w}) + \epsilon | \mathbf{x}) = \text{Var}(\epsilon) = \beta^{-1}$$
(since $y(\mathbf{x}, \mathbf{w})$ is deterministic for fixed $\mathbf{x}$)

**Therefore:**
$$t | \mathbf{x}, \mathbf{w}, \beta \sim \mathcal{N}\big(y(\mathbf{x}, \mathbf{w}), \beta^{-1}\big)$$

---

## 4. Likelihood and Log-Likelihood

### 4.1 Likelihood of the Dataset

Given $N$ independent observations:
$$\{(\mathbf{x}_n, t_n)\}_{n=1}^N$$

Assuming independence:
$$p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \beta) = \prod_{n=1}^N p(t_n | \mathbf{x}_n, \mathbf{w}, \beta)$$

Using Gaussian noise model:
$$p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \beta) = \prod_{n=1}^N \mathcal{N}\big(t_n \mid y(\mathbf{x}_n, \mathbf{w}), \beta^{-1}\big)$$

### 4.2 Why Take the Logarithm?

The likelihood contains a **product** over all observations:
$$p(\mathbf{t} | \mathbf{w}, \beta) = \prod_{n=1}^N p(t_n | \mathbf{x}_n, \mathbf{w}, \beta)$$

**Products are difficult to optimize.** Taking logarithms converts products into sums:
$$\ln \left( \prod_n a_n \right) = \sum_n \ln a_n$$

Since logarithm is monotonically increasing:
$$\arg\max p = \arg\max \ln p$$

**Maximizing the likelihood $\iff$ Maximizing the log-likelihood.**

### 4.3 Log-Likelihood Derivation

$$\ln p(\mathbf{t} | \mathbf{w}, \beta) = \ln \left( \prod_{n=1}^N p(t_n | \mathbf{x}_n, \mathbf{w}, \beta) \right)$$

Substitute Gaussian:
$$= \sum_{n=1}^N \ln \left[ \left(\frac{\beta}{2\pi}\right)^{1/2} \exp\left(-\frac{\beta}{2}(t_n - y_n)^2\right) \right]$$

where $y_n = y(\mathbf{x}_n, \mathbf{w})$.

Using $\ln(a^{1/2}) = \frac{1}{2}\ln a$ and $\ln(e^x) = x$:
$$= \sum_{n=1}^N \left[ \frac{1}{2} \ln \left(\frac{\beta}{2\pi}\right) - \frac{\beta}{2}(t_n - y_n)^2 \right]$$

$$= \frac{N}{2} \ln \left(\frac{\beta}{2\pi}\right) - \frac{\beta}{2} \sum_{n=1}^N (t_n - y_n)^2$$

### 4.4 Define the Error Function

Define the **data error function** (sum of squared errors):
$$E_D(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - y_n)^2$$

The factor $\frac{1}{2}$ is for mathematical convenience when differentiating.

Substitute:
$$\ln p(\mathbf{t} | \mathbf{w}, \beta) = \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2\pi) - \beta E_D(\mathbf{w})$$

---

## 5. Maximum Likelihood and Least Squares Equivalence

### 5.1 Key Observation

$$\ln p(\mathbf{t} | \mathbf{w}, \beta) = \text{constant} - \beta E_D(\mathbf{w})$$

For fixed $\beta$, the first two terms are constants with respect to $\mathbf{w}$.

### 5.2 Equivalence

$$\arg\max_{\mathbf{w}} \ln p(\mathbf{t} | \mathbf{w}, \beta) = \arg\min_{\mathbf{w}} E_D(\mathbf{w})$$

### 5.3 Conclusion

> **Minimizing the Sum of Squared Errors (Least Squares) is exactly equivalent to Maximum Likelihood Estimation under the Gaussian noise assumption.**

This is a fundamental result connecting classical least squares to probabilistic modeling.

---

## 6. Minimizing the Error: Normal Equations

### 6.1 Error Function in Matrix Form

Linear model:
$$y(\mathbf{x}, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$$

Define the **design matrix** $\boldsymbol{\Phi}$:
$$\boldsymbol{\Phi} = \begin{bmatrix} \boldsymbol{\phi}(\mathbf{x}_1)^T \\ \vdots \\ \boldsymbol{\phi}(\mathbf{x}_N)^T \end{bmatrix} \in \mathbb{R}^{N \times M}$$

Target vector:
$$\mathbf{t} = \begin{bmatrix} t_1 \\ \vdots \\ t_N \end{bmatrix}$$

Vector of predictions:
$$\boldsymbol{\Phi}\mathbf{w} = \begin{bmatrix} \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_1) \\ \vdots \\ \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_N) \end{bmatrix}$$

### 6.2 Sum of Squared Errors

$$E(\mathbf{w}) = \frac{\beta}{2} \sum_{i=1}^N \big(t_i - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_i)\big)^2 = \frac{\beta}{2} \|\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}\|^2$$

### 6.3 Gradient and Normal Equations

$$\nabla_{\mathbf{w}} E = \frac{\beta}{2} \nabla_{\mathbf{w}} \|\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}\|^2$$

Using chain rule:
$$\nabla_{\mathbf{w}} E = -\beta \sum_{i=1}^N \big(t_i - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_i)\big) \boldsymbol{\phi}(\mathbf{x}_i) = -\beta \boldsymbol{\Phi}^T (\mathbf{t} - \boldsymbol{\Phi}\mathbf{w})$$

Set gradient to zero ($\beta > 0$):
$$\boldsymbol{\Phi}^T (\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}) = 0$$

### 6.4 Normal Equations

$$\boldsymbol{\Phi}^T \boldsymbol{\Phi} \mathbf{w} = \boldsymbol{\Phi}^T \mathbf{t}$$

When $\boldsymbol{\Phi}^T \boldsymbol{\Phi}$ is invertible:
$$\mathbf{w}_{ML} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t}$$

This is the **maximum likelihood estimate** of the weights.

---

## 7. Maximum Likelihood Estimate of β

### 7.1 Log-Likelihood with Respect to β

$$\ln p(\mathbf{t} | \mathbf{w}, \beta) = \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2\pi) - \frac{\beta}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n))^2$$

### 7.2 Differentiate and Set to Zero

$$\frac{\partial}{\partial \beta} \ln p = \frac{N}{2\beta} - \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n))^2 = 0$$

### 7.3 MLE of β

$$\beta_{ML} = \frac{N}{\sum_{n=1}^N (t_n - \mathbf{w}_{ML}^T \boldsymbol{\phi}(\mathbf{x}_n))^2} = \frac{N}{\sum_{n=1}^N (t_n - y_n)^2}$$

**The ML estimate of the precision is the inverse of the mean squared residual.**

Variance estimate:
$$\sigma_{ML}^2 = \frac{1}{\beta_{ML}} = \frac{1}{N} \sum_{n=1}^N (t_n - y_n)^2$$

---

## 8. Assumptions of Simple Linear Regression

For the simple linear model:
$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$

We typically assume:

| # | Assumption | Mathematical Form | What It Means |
|---|------------|-------------------|---------------|
| **1** | **Linearity** | $\mathbb{E}[y | x] = \beta_0 + \beta_1 x$ | Mean of $y$ is linear in $x$ |
| **2** | **Independence** | $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$ | Errors don't depend on each other |
| **3** | **Homoscedasticity** | $\text{Var}(\epsilon | x) = \sigma^2$ | Constant error variance |
| **4** | **No extreme influential observations** | — | No single point dominates the fit |
| **5** | **Normality of errors** | $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ | Errors are Gaussian |

**Important:** These assumptions concern the **errors $\epsilon$**, not the distribution of $x$.

### 8.1 Detailed Assumptions

#### 1. Linearity
$$\mathbb{E}[y | x] = \beta_0 + \beta_1 x$$
Individual observations deviate: $y = \beta_0 + \beta_1 x + \epsilon$

Visual check: Scatter plot of $y$ vs $x$ should look approximately linear.

#### 2. Independence of Errors
$$\text{Cov}(\epsilon_i, \epsilon_j) = 0 \quad (i \neq j)$$

**Problem:** Time series data often has correlated errors (autocorrelation).

Visual check: Residuals vs observation order should show no pattern.

#### 3. Homoscedasticity
$$\text{Var}(\epsilon | x) = \sigma^2$$

**Heteroscedasticity:** Variance changes with $x$ (e.g., "funnel" pattern in residuals).

Visual check: Residuals vs fitted values should show constant spread.

#### 4. No Extreme Influential Observations
- **Outlier:** Unusual in $y$ (large residual)
- **High leverage:** Unusual in $x$ (far from $\bar{x}$)
- **Influential:** Substantially changes fitted line

Visual check: Cook's distance, leverage plots.

#### 5. Normality of Errors
$$\epsilon_i \sim \mathcal{N}(0, \sigma^2) \implies t_i | x_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2)$$

Needed for: Confidence intervals, hypothesis tests, prediction intervals.

Visual check: Q-Q plot of residuals, histogram.

---

## 9. Problem Sets and Examples

### Problem Set 1: Linear Regression Basics

**Problem 1.1:** Write the linear regression model $y(x, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(x)$ for the following basis functions:
- (a) $\boldsymbol{\phi}(x) = [1, x, x^2, x^3]^T$ (cubic polynomial)
- (b) $\boldsymbol{\phi}(x) = [1, \sin(x), \cos(x)]^T$ (Fourier basis)
- (c) $\boldsymbol{\phi}(x) = [1, x, \exp(-x^2)]^T$ (Gaussian basis)

**Problem 1.2:** For $\boldsymbol{\phi}(x) = [1, x, x^2]^T$ and $\mathbf{w} = [2, -1, 0.5]^T$:
- (a) Write the explicit form of $y(x, \mathbf{w})$
- (b) Compute $y(0), y(1), y(2)$
- (c) Is this model linear in $x$? Linear in $\mathbf{w}$?

**Problem 1.3:** Explain why the "linear" in linear regression refers to the parameters, not the input. Give an example where the model is highly nonlinear in $x$ but still linear regression.

---

### Problem Set 2: Gaussian Noise Model and Likelihood

**Problem 2.1:** Suppose we have a single observation $t = 5.2$ with prediction $y = 5.0$ and noise precision $\beta = 4$.
- (a) What is the noise variance?
- (b) Write the probability density $p(t | x, \mathbf{w}, \beta)$
- (c) Compute the log-likelihood $\ln p(t | x, \mathbf{w}, \beta)$

**Problem 2.2:** For $N$ i.i.d. observations with Gaussian noise, derive the log-likelihood step by step:
$$\ln p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \beta) = \sum_{n=1}^N \ln \mathcal{N}(t_n | y_n, \beta^{-1})$$
Show all intermediate steps.

**Problem 2.3:** The log-likelihood is $\ln p = \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2\pi) - \beta E_D(\mathbf{w})$. 
- (a) For fixed $\beta$, what does maximizing $\ln p$ w.r.t $\mathbf{w}$ correspond to?
- (b) What happens to the optimal $\mathbf{w}$ as $\beta \to \infty$? As $\beta \to 0$?

---

### Problem Set 3: Least Squares and Normal Equations

**Problem 3.1:** Given the design matrix and targets:
$$\boldsymbol{\Phi} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{bmatrix}, \quad \mathbf{t} = \begin{bmatrix} 2 \\ 3 \\ 5 \\ 4 \end{bmatrix}$$
- (a) Compute $\boldsymbol{\Phi}^T \boldsymbol{\Phi}$ and $\boldsymbol{\Phi}^T \mathbf{t}$
- (b) Solve for $\mathbf{w}_{ML} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t}$
- (c) Compute the predictions $\boldsymbol{\Phi}\mathbf{w}_{ML}$ and residuals $\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}_{ML}$

**Problem 3.2:** Derive the normal equations from the scalar form of the sum of squared errors:
$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \boldsymbol{\phi}_n)^2$$
Show that $\frac{\partial E}{\partial w_j} = 0$ leads to $\sum_n (t_n - \mathbf{w}^T \boldsymbol{\phi}_n) \phi_{nj} = 0$ for each $j$.

**Problem 3.3:** When is $\boldsymbol{\Phi}^T \boldsymbol{\Phi}$ not invertible? What does this mean for the solution? How can we fix it (regularization)?

**Problem 3.4:** For the design matrix in Problem 3.1, compute $\beta_{ML}$ using the MLE formula.

---

### Problem Set 4: Assumptions and Diagnostics

**Problem 4.1:** You fit a linear regression and observe the following residual patterns. Identify which assumption is violated:
- (a) Residuals vs fitted values shows a clear U-shaped pattern
- (b) Residuals vs time shows a strong positive trend
- (c) Residuals vs fitted values shows a widening funnel shape
- (d) One point has Cook's distance > 1
- (e) Q-Q plot of residuals shows heavy tails

**Problem 4.2:** Generate synthetic data where the true model is $y = 2 + 3x + \epsilon$ but the errors have variance proportional to $x^2$ (heteroscedastic). What happens to the standard errors of the coefficient estimates if you ignore this?

**Problem 4.3:** In a time series regression, the Durbin-Watson statistic is 0.5. What does this indicate? How might you fix it?

**Problem 4.4:** What is the difference between an outlier, a high-leverage point, and an influential point? Give a concrete example of each.

---

### Problem Set 5: Extensions and Deep Understanding

**Problem 5.1:** (Bias-Variance Tradeoff) In polynomial regression with basis $\boldsymbol{\phi}(x) = [1, x, x^2, \ldots, x^M]^T$:
- (a) What happens to training error as $M$ increases?
- (b) What happens to test error as $M$ increases?
- (c) How does the probabilistic view (with $\beta$) help understand this?

**Problem 5.2:** (Connection to Ridge Regression) If we put a Gaussian prior $\mathbf{w} \sim \mathcal{N}(0, \alpha^{-1}\mathbf{I})$ on the weights, the posterior is proportional to likelihood $\times$ prior. Show that the MAP estimate of $\mathbf{w}$ minimizes:
$$E(\mathbf{w}) = \frac{\beta}{2} \|\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}\|^2 + \frac{\alpha}{2} \|\mathbf{w}\|^2$$
What is the solution?

**Problem 5.3:** (MLE vs MAP) Compare $\mathbf{w}_{ML} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t}$ with the MAP solution. When do they coincide?

**Problem 5.4:** (Bayesian Linear Regression) Instead of point estimates, we can compute the full posterior $p(\mathbf{w} | \mathbf{t})$. For Gaussian likelihood and Gaussian prior, show that the posterior is Gaussian. What are its mean and covariance?

**Problem 5.5:** (Predictive Distribution) For a new input $\mathbf{x}^*$, the predictive distribution is:
$$p(t^* | \mathbf{x}^*, \mathbf{t}) = \int p(t^* | \mathbf{x}^*, \mathbf{w}) p(\mathbf{w} | \mathbf{t}) d\mathbf{w}$$
Explain why this integral is tractable (Gaussian) and what the predictive variance captures that the point prediction doesn't.

---

### Problem Set 6: Worked Examples

**Example 1: Simple Linear Regression by Hand**

Given data: $(x, t) = \{(1, 2), (2, 3), (3, 5), (4, 4)\}$

Model: $y = w_0 + w_1 x$

1. Construct $\boldsymbol{\Phi}$ and $\mathbf{t}$
2. Compute $\mathbf{w}_{ML}$
3. Plot the data and fitted line
4. Compute $R^2 = 1 - \frac{\sum (t_n - y_n)^2}{\sum (t_n - \bar{t})^2}$

**Example 2: Polynomial Regression**

Data: $(x, t) = \{(0, 1), (1, 2), (2, 1), (3, 2), (4, 5)\}$

Fit models with:
- (a) Linear: $\boldsymbol{\phi} = [1, x]^T$
- (b) Quadratic: $\boldsymbol{\phi} = [1, x, x^2]^T$
- (c) Cubic: $\boldsymbol{\phi} = [1, x, x^2, x^3]^T$

Compare training error, and discuss overfitting.

**Example 3: Checking Assumptions**

For the linear fit in Example 1:
1. Plot residuals vs fitted values
2. Plot Q-Q plot of residuals
3. Compute Durbin-Watson statistic
4. Check for influential points (Cook's distance)

---

## Key Takeaways Summary

1. **Linear regression model:** $y(\mathbf{x}, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$ — linear in parameters $\mathbf{w}$
2. **Gaussian noise assumption:** $t = y(\mathbf{x}, \mathbf{w}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \beta^{-1})$
3. **Likelihood:** $p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \beta) = \prod_n \mathcal{N}(t_n | y_n, \beta^{-1})$
4. **Log-likelihood:** $\ln p = \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi) - \beta E_D(\mathbf{w})$
5. **MLE of weights $\iff$ Least Squares:** $\mathbf{w}_{ML} = \arg\min E_D(\mathbf{w})$
6. **Normal equations:** $\boldsymbol{\Phi}^T \boldsymbol{\Phi} \mathbf{w} = \boldsymbol{\Phi}^T \mathbf{t} \implies \mathbf{w}_{ML} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t}$
7. **MLE of precision:** $\beta_{ML} = \frac{N}{\sum_n (t_n - y_n)^2}$ (inverse mean squared residual)
8. **Key assumptions:** Linearity, Independence, Homoscedasticity, No influential points, Normality
9. **Diagnostics:** Residual plots, Q-Q plots, Cook's distance, Durbin-Watson statistic
10. **Probabilistic view enables:** Bayesian inference, uncertainty quantification, regularization as prior, model comparison

---

## Navigation

| Previous | Next |
|----------|------|
| [Lecture 2: Convex Functions & Optimization]({{ site.baseurl }}/lectures/Lecture_2_Convex_Functions_Optimization/) | [Lecture 6-7: Discriminant Functions]({{ site.baseurl }}/lectures/Lecture_6_7_Linear_Models_for_Classification/) |

*End of Lecture Notes*