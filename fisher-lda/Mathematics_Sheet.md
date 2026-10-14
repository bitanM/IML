---
layout: default
title: "Mathematics Sheet (Summary)"
nav_order: 2
parent: Fisher's LDA
has_toc: true
---

# Mathematics Sheet: Fisher's Linear Discriminant Analysis (LDA)

This sheet summarizes the mathematics presented in Fisher's 1936 paper *"The Use of Multiple Measurements in Taxonomic Problems"* and the corresponding Assignment 1 requirements.

---

## 1. Problem Setup

### Data Structure
- **Observations**: $x_i \in \mathbb{R}^4$ (4 measurements per iris flower)
- **Classes**: $C = 3$ species (setosa, versicolor, virginica)
- **Samples per class**: $n_k = 50$ for each $k \in \{1,2,3\}$
- **Total samples**: $N = 150$

### Features (in order)
1. $x_1$: sepal length (cm)
2. $x_2$: sepal width (cm)
3. $x_3$: petal length (cm)
4. $x_4$: petal width (cm)

### Class Means
$$
\mu_k = \frac{1}{n_k} \sum_{x_i \in \text{class } k} x_i \in \mathbb{R}^4
$$

### Overall Mean
$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i = \frac{1}{N} \sum_{k=1}^C n_k \mu_k \in \mathbb{R}^4
$$

---

## 2. Scatter Matrices

### 2.1 Within-Class Scatter Matrix ($S_W$)

**Definition** (sum of squares and products within species):
$$
S_W = \sum_{k=1}^C \sum_{x_i \in \text{class } k} (x_i - \mu_k)(x_i - \mu_k)^T \in \mathbb{R}^{4 \times 4}
$$

**Properties**:
- Symmetric: $S_W^T = S_W$
- Positive semi-definite
- Degrees of freedom: $N - C = 147$ (for 3 classes, 150 samples)
- In Fisher's paper: Table III shows sums of squares and products for 2 species (98 d.f.)

### 2.2 Between-Class Scatter Matrix ($S_B$)

**Definition**:
$$
S_B = \sum_{k=1}^C n_k (\mu_k - \mu)(\mu_k - \mu)^T \in \mathbb{R}^{4 \times 4}
$$

**Properties**:
- Symmetric: $S_B^T = S_B$
- Positive semi-definite
- **Rank**: $\text{rank}(S_B) \leq C - 1 = 2$
- **Reason for rank bound**: The vectors $(\mu_k - \mu)$ for $k=1,\dots,C$ sum to zero (they lie in a $(C-1)$-dimensional subspace)

---

## 3. Fisher's Criterion

For a projection direction $w \in \mathbb{R}^4$, the **discriminant ratio** is:
$$
J(w) = \frac{w^T S_B w}{w^T S_W w}
$$

- **Numerator**: between-class variance (signal)
- **Denominator**: within-class variance (noise)
- Maximizing $J(w)$ finds directions that best separate classes relative to within-class spread

---

## 4. Two-Class Case (Fisher's Original Calculation)

### Setup (Section II of paper)
- Classes: Iris setosa ($k=1$) and Iris versicolor ($k=2$)
- Difference vector: $d = \mu_2 - \mu_1 \in \mathbb{R}^4$

### Optimization
Find linear combination $X = a^T x = \sum_{p=1}^4 a_p x_p$ that maximizes:
$$
\frac{(a^T d)^2}{a^T S_W a}
$$

### Solution
Setting derivative to zero gives the **linear system**:
$$
S_W a = d \quad \text{(Equation 1 in paper)}
$$

Where $S_W$ is the $4 \times 4$ within-class scatter matrix for the two species.

### Fisher's Reported Result
After rescaling so coefficient of sepal length ($a_1$) = 1:
$$
X = x_1 + 5.9037 x_2 - 7.1299 x_3 - 10.1036 x_4
$$

**Mean values**:
- Setosa: 12.3345
- Versicolor: -21.4815
- Difference: 33.816

**Variance of compound**: 11.4269 (dividing sum of squares 1085.5522 by 95 d.f.)

**Standard error of mean difference** (50 per class): 0.6761 cm

---

## 5. Multi-Class Case (3 Classes)

### Generalized Eigenvalue Problem
Maximizing Fisher's criterion leads to:
$$
S_B w = \lambda S_W w
$$

This is equivalent to the standard eigenvalue problem:
$$
S_W^{-1} S_B w = \lambda w \quad \text{(or better: solve } S_W v = S_B w \text{ via linear solve)}
$$

### Key Properties
- At most $C-1 = 2$ **nonzero** generalized eigenvalues
- The corresponding eigenvectors $w_1, w_2$ span the **discriminant subspace**
- Remaining eigenvalues are zero (or numerically near-zero)

### Fisher's Three-Species Construction (Section VI)
Fisher constructs a specific contrast using chromosome counts:
- Setosa: diploid (38 chromosomes) → weight 25
- Versicolor: hexaploid (70 chromosomes) → weight 1
- Virginica: tetraploid (70 chromosomes) → weight 16

The weighted within-class scatter matrix:
$$
S_W^{(3)} = 16 \cdot S_W^{\text{virginica}} + 1 \cdot S_W^{\text{versicolor}} + 25 \cdot S_W^{\text{setosa}}
$$

Mean contrast vector (weighted): $d^{(3)} = 4\mu_{\text{virginica}} + \mu_{\text{versicolor}} - 5\mu_{\text{setosa}}$

Solves: $S_W^{(3)} a = d^{(3)}$

**Reported coefficients** (×100):
- Sepal length: -3.308998
- Sepal width: -2.759132
- Petal length: 8.866048
- Petal width: 9.392551

---

## 6. Projection to 2D

### Discriminant Matrix
$$
W = [w_1 \quad w_2] \in \mathbb{R}^{4 \times 2}
$$

### Projected Coordinates
For each observation $x_i \in \mathbb{R}^4$:
$$
z_i = W^T x_i \in \mathbb{R}^2
$$

This gives the mapping $\mathbb{R}^4 \to \mathbb{R}^2$.

---

## 7. Numerical Verification Requirements

| Check | Formula | Expected |
|-------|---------|----------|
| $S_W$ symmetry | $\|S_W - S_W^T\|_F$ | $\approx 0$ |
| $S_B$ symmetry | $\|S_B - S_B^T\|_F$ | $\approx 0$ |
| $S_B$ rank | $\text{rank}(S_B)$ | $\leq 2$ |
| Eigenvalue residual | $\|S_B w_j - \lambda_j S_W w_j\|_2$ | $\approx 0$ |
| Sign/scale invariance | $w_j$ vs $c w_j$ ($c \neq 0$) | Same subspace |

---

## 8. Interpretation Questions (Assignment Task 8)

### 8.1 Why within-class scatter in denominator?
It measures **within-class variability** (noise). A good discriminant direction should have:
- Large between-class separation (numerator)
- Small within-class spread (denominator)

This is analogous to signal-to-noise ratio.

### 8.2 Why at most 2 discriminant directions for 3 classes?
$S_B = \sum_{k=1}^3 n_k (\mu_k - \mu)(\mu_k - \mu)^T$ has rank $\leq C-1 = 2$ because:
$$
\sum_{k=1}^3 n_k (\mu_k - \mu) = 0
$$
The three mean-deviation vectors are linearly dependent; only 2 are independent.

### 8.3 What does a large generalized eigenvalue indicate?
A large $\lambda_j$ means the corresponding direction $w_j$ achieves **large between-class variance relative to within-class variance** — strong discriminatory power.

### 8.4 Coefficient interpretation
Large absolute coefficients in $w_j$ indicate measurements that contribute most to that discriminant axis. However:
- Coefficients depend on **scale** of measurements
- Only **relative ratios** and **signs** matter for the direction
- Standardizing features changes coefficients but not the discriminant subspace

### 8.5 Information lost in 4D → 2D reduction
- Dimensions orthogonal to the discriminant subspace
- Within-class variance structure not aligned with class separation
- Any non-linear class boundaries (LDA finds only linear discriminants)

---

## 9. Relationship Between Two-Class and Multi-Class LDA

| Aspect | Two-Class | Multi-Class ($C=3$) |
|--------|-----------|---------------------|
| Problem | $S_W a = d$ | $S_B w = \lambda S_W w$ |
| Solution | Single direction | Up to $C-1$ directions |
| Connection | $S_B = n_1 n_2 / N \cdot d d^T$ | Generalizes rank-1 $S_B$ |
| Eigenvalues | 1 non-zero | Up to 2 non-zero |

For $C=2$: $S_B \propto d d^T$ (rank 1), so $S_W^{-1} S_B$ has exactly 1 non-zero eigenvalue.

---

## 10. Sign and Scale Ambiguities

Eigenvectors satisfy $S_B w = \lambda S_W w$.

If $w$ is a solution, then for any $c \neq 0$:
- $c w$ is also a solution with same $\lambda$
- The **subspace** $\text{span}\{w\}$ is unchanged
- The **projection** $W^T x$ changes by scaling but class separation (relative distances) is preserved

**Practical implication**: Coefficients can be flipped in sign or rescaled without changing the discriminant analysis results.

---

## 11. Numerical Implementation Notes

### Preferred Solver
```python
# Instead of explicitly computing S_W^{-1}
eigvals, eigvecs = np.linalg.eig(np.linalg.solve(S_W, S_B))
# Or use generalized eigensolver:
eigvals, eigvecs = scipy.linalg.eigh(S_B, S_W)
```

### Residual Check
For each retained eigenpair $(\lambda_j, w_j)$:
$$
r_j = \|S_B w_j - \lambda_j S_W w_j\|_2
$$
Should be on order of machine precision (~1e-14 to 1e-12).

### Dimension Checks
| Quantity | Expected Shape |
|----------|----------------|
| $S_W$ | (4, 4) |
| $S_B$ | (4, 4) |
| $W$ | (4, 2) |
| $z_i$ | (2,) |
| $Z$ (all projected) | (150, 2) |

---

## 12. Key References

1. **R. A. Fisher**, "The Use of Multiple Measurements in Taxonomic Problems," *Annals of Eugenics*, 7(2), 179–188, 1936. doi:10.1111/j.1469-1809.1936.tb02137.x.
2. **Assignment**: Introduction to Machine Learning, Assignment 1: Fisher's LDA (September 2026).

---

*This sheet covers all mathematical content needed for the assignment implementation and viva preparation.*