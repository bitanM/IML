---
layout: default
title: Key Formulas Reference
nav_order: 4
permalink: /resources/formulas/
---

# Key Formulas Reference

A consolidated reference of all key mathematical formulas across the lectures.

---

## Convex Optimization (Lecture 2)

### Definitions
| Concept | Formula |
|---------|---------|
| Convex function | $f(\theta x_1 + (1-\theta)x_2) \leq \theta f(x_1) + (1-\theta)f(x_2)$ |
| Convex domain | $\theta x_1 + (1-\theta)x_2 \in \text{dom}(f)$ for all $\theta \in [0,1]$ |

### 1D Tests
| Test | Condition |
|------|-----------|
| Second derivative | $f''(x) \geq 0 \implies$ convex |
| Strict convexity | $f''(x) > 0 \implies$ strictly convex |

### Multivariate Tests
| Test | Condition |
|------|-----------|
| Hessian PSD | $H \succeq 0 \iff z^T H z \geq 0 \ \forall z \implies$ convex |
| Hessian PD | $H \succ 0 \iff z^T H z > 0 \ \forall z \neq 0 \implies$ strictly convex |
| Gradient | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^T$ |
| Hessian | $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ |
| Directional curvature | $g(t) = f(x + tz) \implies g''(t) = z^T H(x+tz)z$ |

### Key Implications
- Convex + local minimum exists $\implies$ **global minimum**
- Convex $\not\implies$ minimum exists (e.g., $e^x$)
- Minimum exists $\not\implies$ convex (e.g., $x^4 - x^2$)

---

## Linear Regression (Lectures 3-5)

### Model
| Component | Formula |
|-----------|---------|
| Linear model | $y(\mathbf{x}, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x})$ |
| Design matrix | $\Phi_{ij} = \phi_j(\mathbf{x}_i)$ |
| Predictions | $\Phi \mathbf{w}$ |

### Gaussian Noise Model
| Component | Formula |
|-----------|---------|
| Generative | $t = y(\mathbf{x}, \mathbf{w}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \beta^{-1})$ |
| Likelihood | $p(t | \mathbf{x}, \mathbf{w}, \beta) = \mathcal{N}(t | y(\mathbf{x}, \mathbf{w}), \beta^{-1})$ |
| Dataset likelihood | $p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \beta) = \prod_{n=1}^N \mathcal{N}(t_n | y_n, \beta^{-1})$ |

### Log-Likelihood
$$\ln p(\mathbf{t} | \mathbf{w}, \beta) = \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi) - \beta E_D(\mathbf{w})$$

### Error Function
$$E_D(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \boldsymbol{\phi}_n)^2 = \frac{1}{2} \|\mathbf{t} - \Phi \mathbf{w}\|^2$$

### Normal Equations
$$\Phi^T \Phi \mathbf{w} = \Phi^T \mathbf{t}$$

### MLE Solutions
| Parameter | MLE |
|-----------|-----|
| Weights | $\mathbf{w}_{ML} = (\Phi^T \Phi)^{-1} \Phi^T \mathbf{t}$ |
| Precision | $\beta_{ML} = \frac{N}{\sum_n (t_n - y_n)^2}$ |
| Variance | $\sigma_{ML}^2 = \frac{1}{\beta_{ML}} = \frac{1}{N} \sum_n (t_n - y_n)^2$ |

### Assumptions
| # | Assumption | Formula |
|---|------------|---------|
| 1 | Linearity | $\mathbb{E}[y | x] = \beta_0 + \beta_1 x$ |
| 2 | Independence | $\text{Cov}(\epsilon_i, \epsilon_j) = 0 \ (i \neq j)$ |
| 3 | Homoscedasticity | $\text{Var}(\epsilon | x) = \sigma^2$ |
| 4 | No influential points | — |
| 5 | Normality | $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ |

---

## Discriminant Functions (Lectures 6-7)

### Linear Discriminant
| Component | Formula |
|-----------|---------|
| Discriminant | $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$ |
| Decision boundary | $\mathbf{w}^T \mathbf{x} + w_0 = 0$ |
| Prediction | $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + w_0)$ |
| Normal vector | $\mathbf{w}$ (perpendicular to boundary) |
| Bias | $w_0$ (shifts boundary) |

### Geometry
| Quantity | Formula |
|----------|---------|
| Distance to origin | $d = \frac{|w_0|}{\|\mathbf{w}\|}$ |
| Signed distance | $r = \frac{\mathbf{w}^T \mathbf{x} + w_0}{\|\mathbf{w}\|}$ |
| Closest point to origin | $\mathbf{x}^* = -\frac{w_0}{\|\mathbf{w}\|^2} \mathbf{w}$ |

### Multiclass (K-class)
| Component | Formula |
|-----------|---------|
| Discriminants | $y_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}$ |
| Classification | $\hat{C}(\mathbf{x}) = \arg\max_k y_k(\mathbf{x})$ |
| Pairwise boundary | $y_i(\mathbf{x}) = y_j(\mathbf{x}) \iff (\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (w_{i0} - w_{j0}) = 0$ |
| Zero-plane | $y_k(\mathbf{x}) = 0$ (NOT the decision boundary!) |

### Algebraic Constraint (3 classes)
$$H_{12} + H_{23} = H_{13}$$
The three pairwise boundaries always intersect in a common line.

---

## Generative Classification (Lectures 7-8)

### Bayes' Theorem
| Component | Formula |
|-----------|---------|
| Two-class posterior | $p(C_1 | \mathbf{x}) = \frac{p(\mathbf{x}|C_1)p(C_1)}{p(\mathbf{x}|C_1)p(C_1) + p(\mathbf{x}|C_2)p(C_2)}$ |
| K-class posterior | $p(C_k | \mathbf{x}) = \frac{p(\mathbf{x}|C_k)p(C_k)}{\sum_j p(\mathbf{x}|C_j)p(C_j)}$ |

### Two-Class: Sigmoid
| Component | Formula |
|-----------|---------|
| Log-odds | $a = \ln \frac{p(\mathbf{x}|C_1)p(C_1)}{p(\mathbf{x}|C_2)p(C_2)}$ |
| Sigmoid | $\sigma(a) = \frac{1}{1 + e^{-a}}$ |
| Posterior | $p(C_1 | \mathbf{x}) = \sigma(a)$ |
| Decision boundary | $a = 0 \iff p(C_1|\mathbf{x}) = p(C_2|\mathbf{x}) = \frac{1}{2}$ |

### K-Class: Softmax
| Component | Formula |
|-----------|---------|
| Class scores | $a_k(\mathbf{x}) = \ln[p(\mathbf{x}|C_k)p(C_k)]$ |
| Softmax | $\text{softmax}(a_k) = \frac{e^{a_k}}{\sum_j e^{a_j}}$ |
| Posterior | $p(C_k | \mathbf{x}) = \text{softmax}_k(a_1, \ldots, a_K)$ |
| Classification | $\hat{C} = \arg\max_k a_k(\mathbf{x})$ |

### Gaussian Class-Conditionals (Shared $\Sigma$)
| Component | Formula |
|-----------|---------|
| Density | $p(\mathbf{x}|C_k) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma})$ |
| Two-class weight | $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ |
| Two-class bias | $w_0 = -\frac{1}{2}\boldsymbol{\mu}_1^T \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^T \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 + \ln \frac{p(C_1)}{p(C_2)}$ |
| K-class weights | $\mathbf{w}_k = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k$ |
| K-class biases | $w_{k0} = -\frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k + \ln p(C_k)$ |
| Boundary $C_i/C_j$ | $(\mathbf{w}_i - \mathbf{w}_j)^T \mathbf{x} + (w_{i0} - w_{j0}) = 0$ |

### QDA (Different Covariances)
- Quadratic boundary: $\mathbf{x}^T (\boldsymbol{\Sigma}_2^{-1} - \boldsymbol{\Sigma}_1^{-1}) \mathbf{x} + \cdots = 0$

---

## Maximum Likelihood Estimation (Lectures 7-8)

### Indicators
$$t_{nk} = \begin{cases} 1, & \mathbf{x}_n \in C_k \\ 0, & \text{otherwise} \end{cases}, \quad N_k = \sum_n t_{nk}$$

### MLE Estimates
| Parameter | Formula |
|-----------|---------|
| Mean | $\boldsymbol{\mu}_{k,ML} = \frac{1}{N_k} \sum_{n: t_{nk}=1} \mathbf{x}_n$ |
| Covariance | $\boldsymbol{\Sigma}_{ML} = \frac{1}{N} \sum_{n,k} t_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$ |
| Prior | $\pi_{k,ML} = \frac{N_k}{N}$ |

### Matrix Calculus Identities
| Identity | Formula |
|----------|---------|
| $\frac{\partial \ln |\boldsymbol{\Sigma}|}{\partial \boldsymbol{\Sigma}}$ | $\boldsymbol{\Sigma}^{-1}$ |
| $\frac{\partial \text{Tr}(\boldsymbol{\Sigma}^{-1} \mathbf{S})}{\partial \boldsymbol{\Sigma}}$ | $-\boldsymbol{\Sigma}^{-1} \mathbf{S} \boldsymbol{\Sigma}^{-1}$ |

---

## Logistic Regression (Lecture 9)

### Model
| Component | Formula |
|-----------|---------|
| Sigmoid | $\sigma(a) = \frac{1}{1 + e^{-a}}$ |
| Probability | $p(C_1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + w_0)$ |
| Log-odds | $\ln \frac{p(C_1|\mathbf{x})}{p(C_2|\mathbf{x})} = \mathbf{w}^T \mathbf{x} + w_0$ |

### Likelihood & Loss
| Component | Formula |
|-----------|---------|
| Likelihood | $p(\mathbf{t}|\mathbf{w}) = \prod_n y_n^{t_n} (1-y_n)^{1-t_n}$ |
| Cross-entropy | $E(\mathbf{w}) = -\sum_n [t_n \ln y_n + (1-t_n)\ln(1-y_n)]$ |
| Gradient | $\nabla E = \sum_n (y_n - t_n) \boldsymbol{\phi}_n$ |
| Hessian | $\mathbf{H} = \Phi^T \mathbf{R} \Phi$ where $R_{nn} = y_n(1-y_n)$ |

### Multiclass (Softmax Regression)
| Component | Formula |
|-----------|---------|
| Probability | $p(C_k|\mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \boldsymbol{\phi})}{\sum_j \exp(\mathbf{w}_j^T \boldsymbol{\phi})}$ |
| Cross-entropy | $E = -\sum_n \sum_k t_{nk} \ln y_{nk}$ |

---

## Support Vector Machines (Lecture 11)

### Hard-SVM
| Component | Formula |
|-----------|---------|
| Primal | $\min \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ |
| Margin | $\gamma = \frac{1}{\|\mathbf{w}\|}$ (functional margin = 1) |
| Distance | $r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|}$ |

### Soft-SVM
| Component | Formula |
|-----------|---------|
| Primal (slack) | $\min \lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \xi_i \geq 0$ |
| Slack variable | $\xi_i = \max\{0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b)\}$ |
| Unconstrained | $\min \lambda\|\mathbf{w}\|^2 + \frac{1}{m}\sum \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$ |
| Hinge loss | $\ell_{\text{hinge}}(z) = \max(0, 1-z)$ where $z = y_i(\mathbf{w}^T\mathbf{x}_i + b)$ |

### Dual (Hard-SVM)
| Component | Formula |
|-----------|---------|
| Dual objective | $\max \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$ |
| Constraints | $\alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$ |
| Weight vector | $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ |
| Bias | $b = y_k - \mathbf{w}^T \mathbf{x}_k$ for any $\alpha_k > 0$ |

### Dual (Soft-SVM)
- Box constraints: $0 \leq \alpha_i \leq C$ where $C = \frac{1}{2\lambda m}$

### Kernel Trick
Replace $\mathbf{x}_i^T \mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$

---

## Fisher's LDA

### Scatter Matrices
| Matrix | Formula |
|--------|---------|
| Within-class | $S_W = \sum_{k=1}^C \sum_{i: y_i=k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$ |
| Between-class | $S_B = \sum_{k=1}^C n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$ |
| Total | $S_T = S_W + S_B$ |

### Two-Class
| Component | Formula |
|-----------|---------|
| Difference | $\mathbf{d} = \boldsymbol{\mu}_2 - \boldsymbol{\mu}_1$ |
| Optimal direction | $S_W \mathbf{a} = \mathbf{d}$ |
| Fisher's criterion | $J(\mathbf{a}) = \frac{(\mathbf{a}^T \mathbf{d})^2}{\mathbf{a}^T S_W \mathbf{a}}$ |

### Multi-Class
| Component | Formula |
|-----------|---------|
| Generalized eigenproblem | $S_B \mathbf{w} = \lambda S_W \mathbf{w}$ |
| Standard eigenproblem | $S_W^{-1} S_B \mathbf{w} = \lambda \mathbf{w}$ |
| Discriminant directions | Top $C-1$ eigenvectors |
| Projection | $\mathbf{z}_i = W^T \mathbf{x}_i, \quad W = [\mathbf{w}_1, \ldots, \mathbf{w}_{C-1}]$ |
| Fisher's criterion | $J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$ |

### Key Properties
| Property | Formula |
|----------|---------|
| $S_B$ rank | $\text{rank}(S_B) \leq C-1$ |
| Sum constraint | $\sum_k n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu}) = 0$ |
| Two-class $S_B$ | $S_B = \frac{n_1 n_2}{N} \mathbf{d} \mathbf{d}^T$ (rank 1) |

### Verification Checks
| Check | Formula | Target |
|-------|---------|--------|
| $S_W$ symmetry | $\|S_W - S_W^T\|_F$ | $\approx 0$ |
| $S_B$ symmetry | $\|S_B - S_B^T\|_F$ | $\approx 0$ |
| $S_B$ rank | $\text{rank}(S_B)$ | $\leq 2$ |
| Eigen residual | $\|S_B \mathbf{w}_j - \lambda_j S_W \mathbf{w}_j\|_2$ | $\approx 0$ |
| Sign invariance | $\text{corr}(X\mathbf{w}, X(-\mathbf{w}))$ | $-1$ |

---

## Quick Reference Card

### Most Used Formulas

**Linear Algebra**
- $\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
- $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$
- $\frac{\partial}{\partial X} \ln |X| = X^{-T}$
- $\frac{\partial}{\partial X} \text{Tr}(X^{-1} S) = -X^{-1} S X^{-1}$

**Probability**
- $\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(x-\mu)^2}{2\sigma^2})$
- $\ln \mathcal{N}(x | \mu, \sigma^2) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$

**Optimization**
- Gradient descent: $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla E(\mathbf{w})$
- Newton: $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1} \nabla E(\mathbf{w})$

**Information Theory**
- Entropy: $H(p) = -\sum p(x) \log p(x)$
- KL divergence: $D_{KL}(p \| q) = \sum p(x) \log \frac{p(x)}{q(x)}$