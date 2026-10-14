# Detailed Mathematics Sheet: Fisher's Linear Discriminant Analysis

This sheet provides step-by-step derivations of all equations used in Fisher's LDA, from first principles.

---

## 1. Problem Setup and Notation

### Data Structure
- **Observations**: $x_i \in \mathbb{R}^p$ where $p=4$ (sepal length, sepal width, petal length, petal width)
- **Classes**: $C = 3$ species ($k = 1,2,3$ for setosa, versicolor, virginica)
- **Samples per class**: $n_k = 50$
- **Total samples**: $N = \sum_{k=1}^C n_k = 150$

### Class Means
For each class $k$, the mean vector is:
$$
\mu_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i \in \mathbb{R}^p
$$

### Overall Mean
$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i = \frac{1}{N} \sum_{k=1}^C n_k \mu_k
$$

---

## 2. Two-Class Fisher Discriminant: Complete Derivation

### 2.1 Objective: Maximize Separation Relative to Spread

For two classes ($C=2$), we seek a linear combination:
$$
y = w^T x = \sum_{j=1}^p w_j x_j
$$

This projects $p$-dimensional data onto a **single scalar** $y$.

**Class means in projected space:**
$$
\bar{y}_1 = w^T \mu_1, \quad \bar{y}_2 = w^T \mu_2
$$

**Difference between projected means:**
$$
\bar{y}_2 - \bar{y}_1 = w^T (\mu_2 - \mu_1) = w^T d
$$
where $d = \mu_2 - \mu_1$ is the difference vector in original space.

**Within-class variance in projected space:**
For class $k$, the variance of projected values is:
$$
\text{Var}_k(y) = \frac{1}{n_k} \sum_{i: y_i=k} (w^T x_i - w^T \mu_k)^2 = w^T \left( \frac{1}{n_k} \sum_{i: y_i=k} (x_i - \mu_k)(x_i - \mu_k)^T \right) w
$$

Define the **within-class scatter matrix** for class $k$:
$$
S_k = \sum_{i: y_i=k} (x_i - \mu_k)(x_i - \mu_k)^T
$$

Total within-class scatter (sum over both classes):
$$
S_W = S_1 + S_2 = \sum_{k=1}^2 \sum_{i: y_i=k} (x_i - \mu_k)(x_i - \mu_k)^T
$$

Pooled within-class variance of projected data:
$$
\text{Var}_{\text{within}}(y) = \frac{1}{N-2} w^T S_W w \quad \text{(up to constant factor)}
$$

### 2.2 Fisher's Criterion

We want to maximize the **signal-to-noise ratio**:
$$
J(w) = \frac{(\text{difference between means})^2}{\text{within-class variance}} = \frac{(w^T d)^2}{w^T S_W w}
$$

Since scaling $w$ doesn't change $J(w)$, we can impose a constraint $w^T S_W w = 1$ and maximize $(w^T d)^2$.

### 2.3 Optimization via Lagrange Multipliers

Maximize $J(w) = \frac{(w^T d)^2}{w^T S_W w}$.

Equivalently, maximize $w^T d d^T w$ subject to $w^T S_W w = 1$.

**Lagrangian:**
$$
\mathcal{L}(w, \lambda) = w^T d d^T w - \lambda (w^T S_W w - 1)
$$

**Derivative w.r.t $w$:**
$$
\frac{\partial \mathcal{L}}{\partial w} = 2 d d^T w - 2 \lambda S_W w = 0
$$

$$
d d^T w = \lambda S_W w
$$

Since $d d^T w = d (d^T w) = (d^T w) d$ (a scalar times vector $d$), we have:
$$
(d^T w) d = \lambda S_W w
$$

This implies $w$ is proportional to $S_W^{-1} d$:
$$
w \propto S_W^{-1} d
$$

**Therefore, the optimal direction satisfies:**
$$
S_W w = c d \quad \text{for some constant } c
$$

We can absorb $c$ into the scaling of $w$, giving the **fundamental equation**:
$$
\boxed{S_W a = d}
$$

where $a$ is the coefficient vector (Fisher's notation).

### 2.4 Fisher's Solution for Iris (Setosa vs Versicolor)

From Fisher's paper, Table III gives $S_W$ (sums of squares and products, 98 d.f.):

| | Sepal L | Sepal W | Petal L | Petal W |
|---|---|---|---|---|
| **Sepal L** | 9.7634 | 3.2394 | 9.0356 | 3.2394 |
| **Sepal W** | 3.2394 | 11.8658 | 4.6232 | 2.4746 |
| **Petal L** | 9.0356 | 4.6232 | 12.2978 | 3.8794 |
| **Petal W** | 3.2394 | 2.4746 | 3.8794 | 2.4604 |

Difference vector $d = \mu_{\text{versicolor}} - \mu_{\text{setosa}}$:
| Feature | Setosa | Versicolor | Difference |
|---|---|---|---|
| Sepal L | 5.006 | 5.936 | 0.930 |
| Sepal W | 3.428 | 2.770 | -0.658 |
| Petal L | 1.462 | 4.260 | 2.798 |
| Petal W | 0.246 | 1.326 | 1.080 |

**Solve $S_W a = d$:**

Fisher computes the inverse matrix (Table IV), then $a = S_W^{-1} d$.

Result (before rescaling):
$$
a = [-0.0311511,\; -0.1839075,\; 0.2221044,\; 0.3147370]^T
$$

**Rescale so sepal length coefficient = 1:**
Divide by $a_1 = -0.0311511$:
$$
a_{\text{scaled}} = [1,\; 5.9037,\; -7.1299,\; -10.1036]^T
$$

**Linear discriminant function:**
$$
X = x_1 + 5.9037 x_2 - 7.1299 x_3 - 10.1036 x_4
$$

**Mean values:**
- Setosa: $5.006 + 5.9037(3.428) - 7.1299(1.462) - 10.1036(0.246) = 12.3345$
- Versicolor: $5.936 + 5.9037(2.770) - 7.1299(4.260) - 10.1036(1.326) = -21.4815$
- Difference: $33.816$

**Variance calculation:**
$$
\text{Var}(X) = \frac{a^T S_W a}{95} = \frac{1085.5522}{95} = 11.4269
$$

---

## 3. Multi-Class Scatter Matrices

### 3.1 Within-Class Scatter Matrix ($S_W$)

Generalization to $C$ classes:
$$
S_W = \sum_{k=1}^C \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^T
$$

**Properties:**
- $S_W \in \mathbb{R}^{p \times p}$
- Symmetric: $S_W^T = S_W$
- Positive semi-definite: $v^T S_W v \geq 0$ for all $v$
- Degrees of freedom: $N - C = 147$

**Why this formula?** It sums the scatter (covariance × n) within each class, pooling all classes together.

### 3.2 Between-Class Scatter Matrix ($S_B$)

Measures how far class means are from the overall mean:
$$
S_B = \sum_{k=1}^C n_k (\mu_k - \mu)(\mu_k - \mu)^T
$$

**Derivation:** The total scatter matrix is:
$$
S_T = \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T
$$

Decompose each $x_i - \mu = (x_i - \mu_k) + (\mu_k - \mu)$:
$$
S_T = \underbrace{\sum_{k=1}^C \sum_{i: y_i=k} (x_i - \mu_k)(x_i - \mu_k)^T}_{S_W} + \underbrace{\sum_{k=1}^C n_k (\mu_k - \mu)(\mu_k - \mu)^T}_{S_B}
$$

**Cross-term vanishes:**
$$
\sum_{i: y_i=k} (x_i - \mu_k) = n_k \mu_k - n_k \mu_k = 0
$$

Thus:
$$
\boxed{S_T = S_W + S_B}
$$

**Properties of $S_B$:**
- Symmetric, positive semi-definite
- **Rank**: $\text{rank}(S_B) \leq C - 1$
- **Proof**: The vectors $(\mu_k - \mu)$ for $k=1,\dots,C$ satisfy $\sum_{k=1}^C n_k (\mu_k - \mu) = 0$. They lie in a $(C-1)$-dimensional subspace.

---

## 4. Multi-Class LDA: Generalized Eigenvalue Problem

### 4.1 Fisher's Criterion for Multi-Class

For a projection direction $w \in \mathbb{R}^p$, the between-class and within-class variances are:
$$
\text{Between: } w^T S_B w, \quad \text{Within: } w^T S_W w
$$

Fisher's criterion:
$$
J(w) = \frac{w^T S_B w}{w^T S_W w}
$$

### 4.2 Maximization

Maximize $J(w)$. Since scaling $w$ doesn't change $J(w)$, we can constrain $w^T S_W w = 1$ and maximize $w^T S_B w$.

**Lagrangian:**
$$
\mathcal{L}(w, \lambda) = w^T S_B w - \lambda (w^T S_W w - 1)
$$

**Derivative:**
$$
\frac{\partial \mathcal{L}}{\partial w} = 2 S_B w - 2 \lambda S_W w = 0
$$

$$
\boxed{S_B w = \lambda S_W w}
$$

This is a **generalized eigenvalue problem**.

### 4.3 Solving the Generalized Eigenvalue Problem

Since $S_W$ is positive definite (full rank for Iris), we can multiply by $S_W^{-1}$:
$$
S_W^{-1} S_B w = \lambda w
$$

This is a standard eigenvalue problem for matrix $M = S_W^{-1} S_B$.

**However**, numerically it's better to avoid explicit inversion:
1. Compute Cholesky: $S_W = L L^T$
2. Solve: $L^{-1} S_B L^{-T} v = \lambda v$ where $w = L^{-T} v$
3. Or use `scipy.linalg.eigh(S_B, S_W)` directly

### 4.4 Number of Discriminant Directions

$S_B$ has rank $\leq C-1$. Therefore:
- At most $C-1$ eigenvalues are non-zero
- For $C=3$: **at most 2 discriminant directions**

The eigenvectors corresponding to the largest eigenvalues give the directions of maximum separation.

### 4.5 Discriminant Subspace

Let $W = [w_1, w_2, \dots, w_{C-1}] \in \mathbb{R}^{p \times (C-1)}$ be the matrix of top $C-1$ eigenvectors.

**Projection of observation $x_i$:**
$$
z_i = W^T x_i \in \mathbb{R}^{C-1}
$$

This maps $\mathbb{R}^p \to \mathbb{R}^{C-1}$.

---

## 5. Fisher's Three-Species Construction (Section VI of Paper)

### 5.1 Chromosome-Based Contrast

Fisher notes:
- Setosa: diploid (38 chromosomes)
- Versicolor: hexaploid (70 chromosomes)
- Virginica: tetraploid (70 chromosomes)

He hypothesizes versicolor is a hybrid: $2 \times$ setosa + $4 \times$ virginica $\to$ $6 \times$ versicolor.

The expected mean relationship:
$$
\mu_{\text{versicolor}} \approx \frac{2}{6} \mu_{\text{setosa}} + \frac{4}{6} \mu_{\text{virginica}}
$$

Rearranging: $4 \mu_{\text{virginica}} + \mu_{\text{versicolor}} - 5 \mu_{\text{setosa}} \approx 0$

### 5.2 Weighted Scatter Matrix

Weights proportional to square of coefficients: $4^2=16$, $1^2=1$, $(-5)^2=25$.

Weighted within-class scatter:
$$
S_W^{(3)} = 16 S_W^{\text{virginica}} + 1 S_W^{\text{versicolor}} + 25 S_W^{\text{setosa}}
$$

Weighted mean contrast:
$$
d^{(3)} = 4 \mu_{\text{virginica}} + \mu_{\text{versicolor}} - 5 \mu_{\text{setosa}}
$$

### 5.3 Solve $S_W^{(3)} a = d^{(3)}$

This gives the specific linear compound Fisher reports:
$$
a \approx [-3.308998,\; -2.759132,\; 8.866048,\; 9.392551]^T \times 10^{-2}
$$

**Note**: This is a **specific contrast** based on biological hypothesis, NOT the standard LDA solution. Standard LDA finds directions maximizing overall separation for all classes simultaneously.

---

## 6. Relationship Between Two-Class and Multi-Class LDA

### 6.1 Two-Class $S_B$ is Rank-1

For $C=2$ with equal sample sizes $n_1 = n_2 = n$:
$$
\mu = \frac{n \mu_1 + n \mu_2}{2n} = \frac{\mu_1 + \mu_2}{2}
$$

$$
\mu_1 - \mu = \frac{\mu_1 - \mu_2}{2} = -\frac{d}{2}
$$
$$
\mu_2 - \mu = \frac{\mu_2 - \mu_1}{2} = \frac{d}{2}
$$

$$
S_B = n (\mu_1 - \mu)(\mu_1 - \mu)^T + n (\mu_2 - \mu)(\mu_2 - \mu)^T = \frac{n}{2} d d^T
$$

So $S_B = \frac{n}{2} d d^T$ (rank 1, outer product of $d$ with itself).

### 6.2 Generalized Eigenproblem Reduces to $S_W a = d$

$$
S_B w = \lambda S_W w \implies \frac{n}{2} d d^T w = \lambda S_W w
$$

Since $d^T w$ is a scalar, let $c = \frac{n}{2} (d^T w)$:
$$
c d = \lambda S_W w \implies S_W w = \frac{c}{\lambda} d
$$

This is exactly $S_W a = d$ with $a = \frac{\lambda}{c} w$.

**Conclusion**: Two-class LDA is a special case of multi-class LDA where $S_B$ has rank 1.

---

## 7. Sign and Scale Ambiguities of Eigenvectors

### 7.1 Mathematical Property

If $w$ satisfies $S_B w = \lambda S_W w$, then for any $c \neq 0$:
$$
S_B (c w) = c S_B w = c \lambda S_W w = \lambda S_W (c w)
$$

So $c w$ is also an eigenvector with the same eigenvalue.

### 7.2 Practical Implications

- **Sign flip**: $w \to -w$ reverses the axis but preserves class separation
- **Scaling**: $w \to c w$ scales projections but preserves relative distances
- **Discriminant subspace** $\text{span}\{w_1, w_2\}$ is unique; basis vectors within it are not

**For interpretation:** Typically normalize to unit length $\|w\| = 1$ and fix sign (e.g., first non-zero element positive).

---

## 8. Numerical Implementation Details

### 8.1 Computing $S_W$ and $S_B$

```python
# Within-class scatter
SW = np.zeros((p, p))
for k in range(C):
    Xk = X[y == k]
    mu_k = Xk.mean(axis=0)
    diff = Xk - mu_k  # shape (n_k, p)
    SW += diff.T @ diff  # sum of outer products

# Between-class scatter
SB = np.zeros((p, p))
for k in range(C):
    n_k = np.sum(y == k)
    diff = class_means[k] - overall_mean
    SB += n_k * np.outer(diff, diff)
```

### 8.2 Solving Generalized Eigenproblem

**Preferred (stable):**
```python
from scipy.linalg import eigh
eigvals, eigvecs = eigh(SB, SW)  # Solves SB w = lambda SW w
```

**Alternative (using solve):**
```python
# SW_inv_SB = SW^{-1} SB without explicit inversion
SW_inv_SB = np.linalg.solve(SW, SB)
eigvals, eigvecs = np.linalg.eig(SW_inv_SB)
```

### 8.3 Verification Checks

| Check | Formula | Expected |
|-------|---------|----------|
| $S_W$ symmetry | $\|S_W - S_W^T\|_F$ | $\sim 10^{-15}$ |
| $S_B$ symmetry | $\|S_B - S_B^T\|_F$ | $\sim 10^{-15}$ |
| $S_B$ rank | $\text{rank}(S_B)$ | $\leq 2$ |
| Eigenvalue residual | $\|S_B w_j - \lambda_j S_W w_j\|_2$ | $\sim 10^{-14}$ |
| Sign invariance | $\text{corr}(X w, X (-w))$ | $-1$ |
| Scale invariance | $\text{corr}(X w, X (c w))$ | $1$ |

---

## 9. Interpretation of Results

### 9.1 Eigenvalues
- $\lambda_j = \frac{w_j^T S_B w_j}{w_j^T S_W w_j}$ = ratio of between-class to within-class variance along $w_j$
- Large $\lambda_j$ = strong discrimination
- For Iris: $\lambda_1 \approx 32.2$, $\lambda_2 \approx 0.29$, $\lambda_3, \lambda_4 \approx 0$

### 9.2 Eigenvector Coefficients
For each discriminant direction $w_j$, the coefficient $w_{jk}$ indicates:
- **Magnitude**: Importance of feature $k$ for that discriminant
- **Sign**: Direction of separation (which class has larger projected values)

**Caution**: Coefficients depend on feature scaling. Standardizing features changes coefficients but not the discriminant subspace.

### 9.3 Information Loss in Projection
Reducing from $p=4$ to $C-1=2$ dimensions discards:
1. **Null space of $S_B$**: Directions with zero between-class variance (2 dimensions)
2. **Within-class structure** orthogonal to discriminant directions
3. **Non-linear boundaries** (LDA only finds linear discriminants)

---

## 10. Summary of Key Equations

| Concept | Equation |
|---|---|
| Class mean | $\mu_k = \frac{1}{n_k} \sum_{i: y_i=k} x_i$ |
| Overall mean | $\mu = \frac{1}{N} \sum_{k=1}^C n_k \mu_k$ |
| Within-class scatter | $S_W = \sum_{k=1}^C \sum_{i: y_i=k} (x_i - \mu_k)(x_i - \mu_k)^T$ |
| Between-class scatter | $S_B = \sum_{k=1}^C n_k (\mu_k - \mu)(\mu_k - \mu)^T$ |
| Total scatter | $S_T = S_W + S_B$ |
| Two-class optimal | $S_W a = d$ where $d = \mu_2 - \mu_1$ |
| Multi-class optimal | $S_B w = \lambda S_W w$ |
| Projection | $z_i = W^T x_i$ where $W = [w_1, \dots, w_{C-1}]$ |
| Fisher's criterion | $J(w) = \frac{w^T S_B w}{w^T S_W w}$ |

---

## 11. References

1. **R. A. Fisher**, "The Use of Multiple Measurements in Taxonomic Problems," *Annals of Eugenics*, 7(2), 179–188, 1936.

2. **R. A. Fisher**, "The Statistical Utilization of Multiple Measurements," *Annals of Eugenics*, 8(4), 376–386, 1938.

3. **K. Fukunaga**, *Introduction to Statistical Pattern Recognition*, 2nd ed., Academic Press, 1990. (Chapter on LDA)

4. **T. Hastie, R. Tibshirani, J. Friedman**, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009. (Section 4.3)