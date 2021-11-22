---
theme: gaia
author: Evgeny Noi
#_class: lead
paginate: true
backgroundColor: #fff
marp: true
header: ECE 598BB 
# backgroundImage: url('https://marp.app/assets/hero-background.jpg')
---
<!-- ![bg left:40% 80%](https://marp.app/assets/marp.svg) -->
<!-- _paginate: false -->
<!-- _class: lead -->
<!-- _header: '' -->

# **HOSVD Tensor Emulator for Spatiotemporal Emulators** 

### Evgeny Noi

noi@ucsb.edu
*UC Santa Barbara, Department of Geography*

![](logo1.png)

---
# Problem 
* Modeling spatiotemporal processes is **complicated** due to inherent complex ```dependencies``` across time, space and processes themselves
* These processes are **uncertain**: due to errors in measurement, observation, specification and parameter estimation 
* Find the model ```complex``` enough to capture these dependencies realistically (non-linear interactions, non-stationarity, non-separability, collectiveness) yet ```parsimonious```. 

---
# Solution
* Approaches: 
    * physical-statistical models within hierarchical statistical framework (Berliner 2003; Kuhnert 2014)
    * black-box simulator (e.g. PDE and ABM)
    * **emulators** (surrogate statistical model to emulate input-output, which is computationally less expensive to evaluate)

---

<!-- _backgroundColor: black -->
<!-- _color: #DC267F -->
<!-- _class: lead -->

# Emulators

---

# Emulators 

> Emulator is a function that mimics the output of a simulator at a fraction of simulator's cost. Most frequently specified by Gaussian Processes, polynomial basis expansions, and non-linear surrogate models. 

* Projecting the output onto a standard basis representation (principal components) and adapting emulators to lower-dimensional projection of these fields. 

---

# SVD Emulators

- Hidgon et al. (2008) 
    - SVD derived PCs of simulated runs and GP prior distribution on the weights 
- Hidgon et al. (2011) 
    - relax GP-based prior, model mean response via random-forests

```Simulator output can be approximately expressed as a linear combination of the UD columns from SVD```

---

$$ C = U D V^T $$ 
where $C$ is an $M \times N$ matrix (output dims, runs).  
$$ c = U D v(\theta) + \epsilon $$
where $\epsilon$ is a mean-zero residual with non-diagonal covariance matrix $\Sigma$,  $v(\theta)$ specifies the $N$ coefficients of the linear combination for a particular value of $\theta$. 
$$ c = U D g(\theta, \beta) + \epsilon \quad \text{(non-linear regression)} $$
where $\beta$ are tuning parameters, and $\theta$ are inferential parameters. 

---

# Computational savings: 

1. use $r$ columns capturing most of the data variation from $UD$, 
2. train $r$ machine learning models for emulator
3. reduce computational time on function evaluation 

---

<!-- _backgroundColor: black -->
<!-- _color: #DC267F -->
<!-- _class: lead -->

# Model formulation

---

* Emulate non-linear function $f(x, y, t, \theta_t, ... , \theta_p)$
* Emulations stored as multidimensional tensor $\mathcal{X}$ $M$ by $N$ by $T$ by $P_1$ ... by $P_p$
* latin hypercube is used for initial prior sampling 

Then for HOSVD we have: 

$$ f(x*, y*, t*, \theta*1,..., \theta*p) = Z \times u_1(x*) \times u_2(y*) \times u_3(t*) \times ... \times u_{p+3}(\theta*p) + \epsilon $$

where $u_1, ... , u_p$ - are non-linear vector-valued functions (behaving like $v(\theta)$ from SVD). 

--- 

# Emulator Construction

- Train supervised ML (e.g. GP regression) on $M$ values of $x$ and first column of $U_1$ to get $\hat{u}_{11}$. The choice of supervised ML method needs to be investigated. 

---
<!-- _backgroundColor: black -->
<!-- _color: #DC267F -->

## Algorithm: 

1. Unfold and scale a tensor 
2. Run regular SVD. Compare screeplot with SVD of permuted unfolded matrices. Keep $\psi$ singular vectors.
3. For each tensor mode set truncated SVD rank at $\psi$ and adding 1 to account for scaling. 
4. Calculate decomposition quality - prop of var explained by R (R - low rank approx, $X_c = X - \bar{X}$)

$$ 1 - \frac{RSS}{TSS} =  1 - \frac{||X-R||_F^2}{||X_c||_F^2} $$



---

# Conclusion 

---
<!-- _class: lead -->
<!-- _header: '' -->
<!-- _paginate: false -->
<style scoped>
section {
  /* font-family: 'Times New Roman', serif !important; */
  font-size: 150%;
}
blockquote {
    text-align:left;
    border-left:3px;
    border-right:px;
    width:auto;
    display:inline-block;
    padding:1px px;
    font-size: 80%;
}
</style>

![bg left](mos.png)

# ~~Beers~~ Questions?
# :beers: 

*Evgeny Noi*  
![w:150](me_new.png) 
> If geography is a prose, maps are iconography. (Lennar Meri)

:incoming_envelope: noi@ucsb.edu
