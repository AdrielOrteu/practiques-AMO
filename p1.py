"""
Nom 1: Adriel Orteu Portocarrero
NIU 1: 1750927
Nom 2: David Pérez Rodríguez
NIU 2: 1745480
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d


### PART 1
### TASCA 1.1
def in_region(x):
    x1, x2 = x
    # retorna True nomes si x satisfa TOTES CINC restriccions anteriors
    return bool(((x1 + x2) <= 10) and (x1 <= 6) and (x2 <= 7) and (x1 >= 0) and (x2 >= 0))


np.random.seed(0)
samples = np.random.uniform(0, 8, size=(4000, 2))
# queda't nomes amb les mostres factibles
feasible = [x for x in samples if in_region(x) is True]
print(f"Mostres factibles: {len(feasible)}")  # NOTE: 2374

### TASCA 1.2
hull = ConvexHull(feasible)
convex_hull_plot_2d(hull)
plt.show()

### TASCA 1.3
print()
print("Resposta Tasca 1.3: Obtenim un Hull Convex amb multiples vertex per que al ser aleatoris alguns punts prop" \
      "la vora sobresurten i son maxims locals. Aixi el ConvexHull els identifica com arestes" \
      "i obtenim un poligon amb mes de cinc cantonades.")

### TASCA 1.4
propostes = (
    (4, 5),
    (7, 2),
    (5, 6)
)
propostes_valides = [x for x in propostes if in_region(x)]
print()
print(f"Propostes Valides: {propostes_valides}")  # NOTE: (4,5)


### PART 2
### Tasca 2.1
def verify_convexity_numerically(f, sampler, n_trials=5000, tol=1e-9):
    # compta quants assaigs de combinacio convexa aleatoria VIOLEN
    # f(lam*x + (1-lam)*y) <= lam*f(x) + (1-lam)*f(y)
    c = 0
    for _ in range(n_trials):
        x1 = sampler()
        x2 = sampler()
        lmbda = np.random.uniform(0, 1)
        
        esquerre = f((lmbda) * x1 + (1 - lmbda) * x2)
        dret = lmbda * f(x1) + (1 - lmbda) * f(x2)
        
        if esquerre > dret + tol:
            c += 1
    
    return c


def hessian_psd(H, tol=1e-9):
    # retorna True nomes si tots els valors propis de la H simetrica son >= -tol
    EgnV = np.linalg.eigvalsh(H)  # eigenvalues
    return np.all(EgnV >= -tol)


### Tasca 2.2
def g1(x):
    return x ** 4


def g2(x):
    return x * np.sin(x)


interval = lambda: np.random.uniform(-5, 5)  # Generador de mostres per a l'interval [-5, 5]

# Executem la comprovació per a cada funció
violations_g1 = verify_convexity_numerically(g1, interval)
violations_g2 = verify_convexity_numerically(g2, interval)
print()
print(f"Violacions per a g1(x) = x^4: {violations_g1}")  # NOTE: 0
print(f"Violacions per a g2(x) = x*sin(x): {violations_g2}")  # NOTE: 3413

# CONCLUSIÓ
if violations_g2 > 0:
    print("La funció segura per posar al reentrenament nocturn és g1(x).")
else:
    print("Ambdues són segures.")

### Tasca 2.3
np.random.seed(2)
A = np.random.randn(30, 2)  # matriu 30x2
rho = 0.3  # rho (d'l2)

I = np.eye(2)  # matriy identitat 2x2 perquè x té 2D
H = np.dot(A.T, A) + rho * I  # H = A^T * A + rho * Identitat

es_convexa = hessian_psd(H)
valors_propis = np.linalg.eigvalsh(H)

print()
print(f"Convexa?: {es_convexa}")  # NOTE: True
print(f"Valors propis: {valors_propis}")  # NOTE: [25.82396343 37.17854836]

### Tasca 2.4
print()
print("Resposta Tasca 2.4: El descens de gradient només troba mínims locals i no sap si són globals o si s'encallarà" \
      "en altres llocs pitjors. En canvi, el nostre conjunt d'eines comprova matemàticament que la funció és convexa abans " \
      "de començar, assegurant així que qualsevol mínim local que trobem serà directament el global.")