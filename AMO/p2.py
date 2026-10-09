import numpy as np
import scipy
from scipy.optimize import \
    linprog  # NOTE: linprog --> Per resoldre programes lineals (només minimitza, per tant, si volem maximitzar hem de multiplicar per -1)

# Verificació llibreries
# print (" NumPy ", np . __version__ , "| SciPy ", scipy . __version__ )
# print ("OK , ready to go.")


"""
IMPORTANT durant tota la pràctica
Nomenclatura que utilitzarem
    Centres de dades:
        - A -> US-East
        - B -> EU-West
    Serveis
        - 1 -> Analytics
        - 2 -> ML-Training
        - 3 -> Back-up

Recepta de Modelització
1. Estem minimitzant el cost d'encaminar recursos des dels centres de dades fins als serveis.
2. Variables de decisió: Xa1 (de US-East a Analytics), Xa2 (de US-East a ML-Training), Xa3 (de US-East a Back-up), Xb1 (de EU-West a Analytics), Xb2 (de EU-West a ML-Training), Xb3 (de EU-West a Back-up)
3. min Z = 4*Xa1 + 6*Xa2 + 8*Xa3 + 5*Xb1 + 3*Xb2 + 7*Xb3
4. Restriccions:
    - Capacitat: Xa1 + Xa2 + Xa3 <= 100  i  Xb1 + Xb2 + Xb3 <= 150.
    - Demanda: Xa1 + Xb1 >= 80, Xa2 + Xb2 >= 120  i  Xa3 + Xb3 >= 50.
    - No negativitat: X_ij >= 0
5. Límits de les variables: 0 <= X_ij < infinit (sense cota superior explícita, ja que la capacitat de cada centre ja les acota indirectament).
   Mètode de resolució: programa lineal (objectiu i restriccions lineals)
"""

cost = np.array([
    [4, 6, 8],  # US-East
    [5, 3, 7],  # EU-West
])
capacity = np.array([100, 150])
demand = np.array([80, 120, 50])
dcs = ["US-East", "EU-West"]  # data centers
svcs = ["Analytics", "ML-Training", "Back-up"]  # services

# Tasca 1 (a)
C_total = capacity.sum()
D_total = demand.sum()
print()
print("Tasca 1 (a)")
print(f"Capacitat total = {C_total} TB/dia | Demanda total = {D_total} TB/dia")
print(
    "Conclusió: La capacitat total és >= que la demanda total, però això no vol dir que cada servei rebi la seva demanda, perquè la capacitat és la suma de tots els centres i s'ha de repartir entre els serveis.")

# Tasca 1 (b)
print()
print("Tasca 1 (b)")
print(
    "Variables de decisió: Xa1 (de US-East a Analytics), Xa2 (de US-East a ML-Training), Xa3 (de US-East a Back-up), Xb1 (de EU-West a Analytics), Xb2 (de EU-West a ML-Training), Xb3 (de EU-West a Back-up)")
print("Funció objectiu: min Z = 4*Xa1 + 6*Xa2 + 8*Xa3 + 5*Xb1 + 3*Xb2 + 7*Xb3")
print("Restriccions:"
      "   - Capacitat: Xa1 + Xa2 + Xa3 <= 100  i  Xb1 + Xb2 + Xb3 <= 150."
      "   - Demanda: Xa1 + Xb1 >= 80, Xa2 + Xb2 >= 120  i  Xa3 + Xb3 >= 50."
      "   - No negativitat: X_ij >= 0")

# Tasca 1 (c)
print()
print("Tasca 1 (c)")
print("Ho reescribim de la següent manera: -Xa1 - Xb1 <= -80 | -Xa2 - Xb2 <= -120 | -Xa3 - Xb3 <= -50")
print("Conclusió: La regla és multiplicar els dos costats per -1, i així el >= passa a ser <=.")

# Tasca 1 (d)
print()
print("Tasca 1 (d)")
millor = cost.argmin(axis=0)
cost_drecera = int((cost.min(axis=0) * demand).sum())
print(f"Centre més barat per servei: {[dcs[k] for k in millor]}")
print(f"Cost total de la drecera = {cost_drecera} EUR")
# Comprovació ràpida de capacitats per servei assignat
carrega_eu = demand[millor == 1].sum()
print(f"EU-West necessita: {carrega_eu} TB/day | Capacitat màxima: {capacity[1]} TB/day")
print(
    "Conclusió: La drecera falla perquè assigna a EU-West 170 TB/day quan només en pot assumir 150. És un plantejament atractiu però inviable a la pràctica.")


# Tasca 1 (e) i (f)

def solve_transport(cost, capacity, demand, forbidden=None):
    """ Cost mínim d'encaminar de centres (files) a serveis (columnes).
    L'òptim és global perquè la regió factible d'un programa lineal és convexa i l'objectiu és lineal. """
    
    n_data_centers, n_services = cost.shape  # n_data_centers seria n_dc i n_services seria n_svc en el codi proporcionat. Ho canviem perquè ens és més intuitiu així.
    c = cost.flatten()  # Convertim la matriu 2x3 en una llista de 6 números: [4,6,8,5,3,7] --> [Xa1, Xa2, Xa3, Xb1, Xb2, Xb3].
    
    A_ub = []
    b_ub = []
    
    # capacitat de cada centre
    for i in range(n_data_centers):
        fila = np.zeros(n_data_centers * n_services)
        fila[i * n_services:(i + 1) * n_services] = 1
        A_ub.append(fila)
        b_ub.append(capacity[i])
    
    # demanda de cada servei (canviem ja el signe)
    for j in range(n_services):
        fila = np.zeros(n_data_centers * n_services)
        for i in range(n_data_centers):
            fila[i * n_services + j] = -1
        A_ub.append(fila)
        b_ub.append(-demand[j])
    
    bounds = [(0, None)] * (
                n_data_centers * n_services)  # bounds = (0, None)  <-->  0 <= x < inf  (NO podem enviat TB negatius)
    if forbidden is not None:
        for (i, j) in forbidden:
            bounds[i * n_services + j] = (0, 0)  # Camí prohibit
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return result


# Tasca 1 (f)
print()
print("Tasca 1 (f)")
res_f = solve_transport(cost, capacity, demand)
print("status:", res_f.status)
print("message:", res_f.message)
print("fun:", res_f.fun)

# Tasca 1 (g)
print()
print("Tasca 1 (g)")
print("Assignacions òptimes (linprog):")
matrix = res_f.x.reshape(len(capacity), len(demand))
for i, data_center in enumerate(dcs):
    for j, service in enumerate(svcs):
        v = matrix[i, j]
        if v > 1e-5:  # Evitem amb això errors de precisió
            print(f"    - {data_center} --> {service}: {v} TB/day")

print(f"Comparació de costos:")
print(f"    - Cost de la drecera (teòricament més barat però inviable): {cost_drecera} EUR")
print(f"    - Cost òptim real (linprog): {res_f.fun} EUR")
print(
    "Conclusió: La drecera surt a menys cost, però quan s'incmpleix la restricció de capacitat d'EU-West, la única solució real és la que dóna l'optimitzador lineal.")

# Tasca 1 (h)
print()
print("Tasca 1 (h)")
print(
    "Conclusió: És l'òptim global perquè en un prorama lineal la regió factible és convexa i l'objectiu és lineal, així que no hi ha mínims locals.")

# Tasca 2 (a)
print()
print("Tasca 2 (a)")
capacity_2a = capacity.copy()
capacity_2a[1] = 40
res_2a = solve_transport(cost, capacity_2a, demand)
print("status:", res_2a.status)
print("message:", res_2a.message)
print("fun:", res_2a.fun)
print(
    f"Conclusió: L'exercici NO és factible perquè la capacitat total ({capacity_2a.sum()}) és menor que la demanda total ({D_total}), així que cap repartiment pot cobrir tota la demanda. No és un error del codi, el model reflecta un cas impossible. El propietari d'un servei veuria que no li arriba tota la seva demanda.")

# Tasca 2 (b)
print()
print("Tasca 2 (b)")
res_2b = linprog(c=[-1], bounds=[(0, None)], method="highs")
print("status:", res_2b.status)
print("message:", res_2b.message)
print(
    "És no acotat perwue busquem minimizar f(x)=-x sense imposar cap restricció (cota superior) a la variable x. En el model cada variable té un límit finit (la capacitat dels centres) i els costos són positius, per tant la regió factible està acotada i mai pot sortir no acotat.")

# Tasca 2 (c)
print()
print("Tasca 2 (c)")
print(
    "Conclusió: Primer miraria què ha canviat a les dades d'entrada: si la demanda ha augmentat, si la capacitat ha disminuït o si alguna ruta ha deixat de funcionar. Augmentar la capacitat podria solucionar els dos primers casos, però costa diners i pot ser que el problema sigui una dada mal posada, i en el tercer cas no tindria cap efecte perquè la ruta no es pot fer servir igualment.")

# DADES
cost2 = np.array([[4, 7, 9, 3],  # US-East
                  [6, 3, 6, 5],  # EU-West
                  [8, 5, 4, 7],  # APAC-South
                  ])
capacity2 = np.array([120, 150, 90])
demand2 = np.array([70, 130, 60, 80])
dcs2 = ["US-East", "EU-West", "APAC-South"]
svcs2 = ["Analytics", "ML-Training", "Back-up", "Real-Time"]

# Tasca 2 (d)
print()
print("Tasca 2 (d)")
prohibit = [(2, 3)]  # APAC-South no pot servir Real-Time
print(f"Parell prohibit (centre, servei): {prohibit}")
print(
    "Conclusió: solve_transport ja té l'argument forbidden, que posa bounds=(0,0) a APAC-South --> Real-Time, i així aquesta variable sempre val 0.")

# Tasca 2 (e)
print()
print("Tasca 2 (e)")
res2 = solve_transport(cost2, capacity2, demand2, forbidden=prohibit)
print("status:", res2.status, "| cost:", res2.fun)
x2 = res2.x.reshape(3, 4)
print("Taula d'assignació (TB/dia). Columnes:", svcs2)
for i in range(3):
    print(f"{dcs2[i]}: {x2[i]}")
carrega2 = x2.sum(axis=1)
for i in range(3):
    print(f"{dcs2[i]}: càrrega {carrega2[i]} de {capacity2[i]} --> {carrega2[i] / capacity2[i] * 100} %")

# Tasca 2 (f)
print()
print("Tasca 2 (f)")
baix = int(np.argmin(carrega2 / capacity2))
print(f"Centre infrautilitzat: {dcs2[baix]}")
print(
    f"Conclusió: El centre amb capacitat sobrant és {dcs2[baix]}, i la seva fila de costos és {cost2[baix].tolist()}. Només és el més barat a Back-up (4), i per això el serveix al complet. A Analytics és el més car (8), a ML-Training és el del mig (5) i a Real-Time no pot servir. Com que sobra capacitat en total, l'optimitzador deixa sense fer servir el centre que té menys avantatges.")

# Tasca 2 (g)
# NOTE: Mateix valor baix de la tasca 2 (f)
capacity_20 = capacity2.copy()
capacity_20[baix] = capacity_20[baix] - 20
res_20 = solve_transport(cost2, capacity_20, demand2, forbidden=prohibit)
print("Reduint 20 --> status:", res_20.status, "| cost:", res_20.fun)

capacity_21 = capacity2.copy()
capacity_21[baix] = capacity_21[baix] - 21
res_21 = solve_transport(cost2, capacity_21, demand2, forbidden=prohibit)
print("Reduint 21 --> status:", res_21.status, "| missatge:", res_21.message)

print(f"Capacitat total - demanda total = {capacity2.sum()} - {demand2.sum()} = {capacity2.sum() - demand2.sum()}")
min_baix = demand2.sum() - (capacity2[0] + capacity2[1])
print(
    f"Conclusió: Amb 20 el cost no canvia i amb 21 no és factible. {dcs2[baix]} ha d'aportar com a mínim {demand2.sum()} - ({capacity2[0]} + {capacity2[1]}) = {min_baix}, i en té {capacity2[baix]}, per tant el 20 és {capacity2[baix]} - {min_baix}, és a dir, la capacitat que sobra. La restricció de {dcs2[baix]} és inactiva (té marge) i les d'US-East i EU-West són actives (estan al màxim).")

# Tasca 3 (a)
# Variables: Xn1, Xn2, Xs1, Xs2 (North-DC i South-DC cap a Cold-Storage i Deep-Archive)
c_3 = [3, 5, 6, 2]
A_ub_3 = [
    [1, 1, 0, 0],  # North-DC: Xn1 + Xn2 <= 60
    [0, 0, 1, 1],  # South-DC: Xs1 + Xs2 <= 90
    [-1, 0, -1, 0],  # Cold-Storage: -Xn1 - Xs1 <= -70
    [0, -1, 0, -1],  # Deep-Archive: -Xn2 - Xs2 <= -50
]
B_ub_3 = [60, 90, -70, -50]
bounds_3 = [(0, None)] * 4

data_centers_3 = ["North-DC", "South-DC"]
services_3 = ["Cold-Storage", "Deep-Archive"]

res_3 = linprog(c_3, A_ub=A_ub_3, b_ub=B_ub_3, bounds=bounds_3, method="highs")

print()
print("Tasca 3 (a)")
# RESULTATS PRINTEJATS
print(f"Status: {res_3.status} | Missatge: {res_3.message}")
print(f"Cost total mínim = {res_3.fun} EUR")

print("Encaminament òptim:")
matrix_3 = res_3.x.reshape(2, 2)
for i, data_center_3 in enumerate(data_centers_3):
    for j, service_3 in enumerate(services_3):
        val = matrix_3[i, j]
        if val > 1e-5:
            print(f"- {data_center_3} --> {service_3}: {val} TB/day")

# Tasca 3 (b)
print()
print(
    "Tasca 3 (b) <-- Llegir la nota en el bloc de codi corresponent a la Tasca 3 (b), allà trobaràs la nostra resposta.")
"""
L'bjectiu d'aquest model es minimitzar el cost total d'encaminar dades des de dos centres d'arxivament (North-DC i South-DC)
fins als dos nivells d'emmagatzematge (Cold-Storage i Deep-Archive).
Variables de decisió: X_ij és la quantitat en TB/dia enviada des del centre de dades (i) cap al servei d'arxivament (j). iE{n, s}, jE{1, 2}
Suposarem North-DC --> n, South-DC --> s | Cold_Storage --> 1, Deep-Archieve --> 2
Restriccions:
Restriccions de capacitat:
    - Capacitat: Xn1 + Xn2 <= 60  i  Xs1 + Xs2 <= 90
    - Demanda: Xn1 + Xs1 >= 70  i  Xn2 + Xs2 >= 50
    - No negativitat: X_ij >= 0
"""

# Reflexió 1
print()
print("Reflexió 1")
print(
    "Conclusió: Ho detecta mirar l'status del solver (de la part 2) perque si surt 2 significa que la capacitat no arriba. És fiable perquè la regió factible d'un programa lineal és convexa i, per això, si surt status 0 l'òptim serà global.")

# Reflexió 2
print()
print("Reflexió 2")
print(
    "Conclusió: La Tasca 3 (b) és com la secció de Model Matemàtic de l'informe del projecte perquè s'hi explica objectiu, variables i restriccions.")