
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


#_______________________________________________________________________
#         Espacio para definir funciones
#__________________________________________________________________________

#Definimos la funcion de regresion lineal
def regresion (x,y):
  RegresorVP=linear_model.LinearRegression()
  X=np.array(x).reshape(-1,1)
  Y=np.array(y).reshape(-1,1)

  Regresor=linear_model.LinearRegression()
  Regresor.fit(X,Y)
  Predict=Regresor.predict(X)

  m=Regresor.coef_[0][0]
  b=Regresor.intercept_[0]


  return m,b

#Definimos la funcion para la obtencion de los coeficientes

def Fij (b,c,n,m_curve,phi_r,ai):

  #Obtenemos los datos de la matriz de coeficientes
  for j in range(1,n+1):
    renglon=[]
    for i in phi_r:

      aux=(((4*b)/(m_curve*c))+((j)/(math.sin(i))))*(math.sin(j*i))
      #Metemos el dato auxiliar a una lista de renglones
      renglon.append(aux)
    #Cragamos el renglo a la matriz para que sea una fila.
    ai.append(renglon)
  return




#________________________________________________________________________________
#             Seccion de codigo principal.
#_______________________________________________________________________________

#____________________LISTAS DE APOYO_____________________________________________

AoA=[]              #Lista de angulos de ataque en grados
Rad_AoA=[]          #Lista de apoyo de angulos de ataque a radianes
cl_list=[]          #Lista de los coeficientes de sustentacion a diferentes AoA
ai=[]               #Lista de los coeficientes de para apoyo de los AJ
aj=[]               #Lista de los Coeficientes finales
V_ind=[]            #Listas de terminos independientes
Aj=[]               #Coeficientes de solucion del modelo.
L_distribution=[]   #Distribucion de sustentacion

Env_dis=[]      #Valores de envergadura discretizada

phi=[]         #Angulos de la discretizacion
phi_rad=[]     #Angulos de discretizacion en radianes

#valores de apoyo para el caso de un torsimiento simple.

bj=[]          #Vector de bj del primer modelo de sustentacion
Aj_t=[]        #Vectores modificados para Aj con torsimientos
w_lineal=[]    #Valores de los angulos con las aproximaciones
L_distribution_t=[]   #Con torsimiento geometrico


#Valores de apoyo para torsimiento y despliegue de superficies de control

E_cuerdas=[]      #Lista con las razones de cuerda de cada una de las secciones del ala
Epsilon_phi=[]    #Valor epsilon en funcion de la relacion de cuerdas local
xi_phi=[]         #Arreglo de terminos independientes xi
xi_phi_transpose=[] #Arreglo de los coeficientes transpuestos
cj=[]             #Coeficientes para el despliegue de la superficie de control
Aj_t_d=[]         #Coeficientes con torsimiento y despliegue de superficies.
L_distribution_t_d=[]  #Valores de la fuerza de sustentacion con la inclusion de los alerones.

#Nombramos y asignamos un data frame con los valores del archivo del perfil omitiendo las
#Primeras 10 filas
df = pd.read_csv('/content/drive/MyDrive/MateriasFI/Aeroelasticidad /xf_n2415.csv', skiprows=10)
#print(df)

#Asignamos los headers de cada una de las columnas de informacion.
df.columns = ["AoA", "Cl", "cd", "cdp", "cm", "Top_xtr", "Bot_xtr"]
#print(df["AoA"])

#Ploteamos la curva solo para verificar que todo este de manera correcta.
plt.plot(df["AoA"], df["Cl"])
plt.title("Curva de sustentacion vs AoA")
plt.xlabel('AoA')
plt.grid()
plt.ylabel('cl')
plt.show()


#Obtener el modelo lineal de la curva de -5 a 5 grados

df["Rad_AoA"]=np.radians(df["AoA"])

#Guardamos la informacion en una lista de AoA y los cl
AoA=df["AoA"].tolist()

AoA_inf=AoA.index(-5.0)
AoA_sup=AoA.index(5.0)

#Solo agregar los datos de -5 a 5 de los data frame a las listas

Rad_AoA = df['Rad_AoA'].iloc[AoA_inf: AoA_sup].values.tolist()
cl_list=df['Cl'].iloc[AoA_inf: AoA_sup].values.tolist()

#Mandamos los arreglos a la regresion lineal

lineal=regresion(Rad_AoA,cl_list)

#Obtenemos el valor de cl0 y la pendiente de la aproximacion.
cl0=lineal[1]
m_Polar=lineal[0]


#Calculo de valor de nula sustentacion

alpha_L0=-1*(cl0)/m_Polar


#__________________________________________________________
#             Solucion de ala recta
#__________________________________________________________

#Definicon de los datos de la simulacion

###################    Datos de entrada ###################################
#____________________________________________________________________________
v_inf= 600*(1000)*(1/3600)    #m/s Velocidad relativa al viento
c=1                           #m   Cuerda constante del ala
b=6                           #m   Evergadura completa avion
rho0=1.2                      #Kg/m3 Densidad del aire
alpha_ini=10                  #grados    Angulo de ataque de vuelo

#Numero de nodos
Nx=151

#Calculos de la discretizacion

dx=(180/Nx)

#_____________________________________________________________________________

#Guardamos la discretizacion de los angulos en una lista
for i in range(1,Nx):
  phi.append(dx*i)


#Angulos de la discretizacion en radianes
phi_rad=[math.radians(angulo) for angulo in phi]

#Obtenemos la matriz
n=len(phi)

Fij(b,c,n,m_Polar,phi_rad,ai)

#Trnsponemos la matriz de coeficientes obtenida.

ai_transpose = [[ai[j][i] for j in range(n)] for i in range(n)]

#asignamos a un arrray los datos de la lista
MatrizCoef=np.array(ai_transpose)

#Invertimos la transpuesta de la matriz
Inv_ai_transpose=np.linalg.inv(MatrizCoef)

#Comprobamos el resultado

resultado = np.matmul(Inv_ai_transpose, MatrizCoef)

#Vector de terminos independientes

for i in range(1,n+1):
  V_ind.append(1)

aj = np.matmul(Inv_ai_transpose, V_ind)

#Damos el umbral para que los datos se vueelvan cero
umbral = 1e-15

# Reemplazar valores pequeños por cero
aj[abs(aj) < umbral] = 0.0


#Pasamos dato de entrada a radianes
alpha_ini_rad=math.radians(alpha_ini)

#Valores de los coeficientes Aj
for i in aj:
  Aj.append(i*(alpha_ini_rad-alpha_L0))



#Distribucion de sustentacion
for i in range(n):
  aux2=0.0
  for j in range(n):
    aux2=2*rho0*((v_inf)**2)*b*Aj[j]*math.sin((j+1)*phi_rad[i])+aux2

  L_distribution.append(aux2)

#Lista para guardar los valores eespecificos del calculo

for i in phi_rad:

  val=(b/2)*math.cos(i)
  Env_dis.append(val)


#Ploteamos la curva solo para verificar que todo este de manera correcta.
plt.plot(Env_dis, L_distribution)
plt.title("Distribucion de sustentacion")
plt.xlabel('Envergadura[m]')
plt.grid('minor')
plt.ylabel('Lift[N]')
plt.show()


print(max(L_distribution))


#_________________________________________________________________________
#                 Ala recta con torsion geometrica
#________________________________________________________________________

#Dato de entrada

Washout=-10    #grados

#Consideracion de washout lineal
#Obteniendo esos valores
for i in phi_rad:
  aux=abs(math.cos(i))
  w_lineal.append(aux)

#Agregamos los angulos a un array
W_lineal_A=np.array(w_lineal)
#transponemos la matriz para que sea de 150x1
W_lineal_transpose=np.transpose(W_lineal_A)

#Calculando los coeficientes de apoyo
bj = np.matmul(Inv_ai_transpose, W_lineal_transpose)

#Damos el umbral para que los datos se vueelvan cero
umbral = 1e-15

# Reemplazar valores pequeños por cero
bj[abs(bj) < umbral] = 0.0

#Valores de los coeficientes Aj_t
#Este comando corre dos listas en simultaneo con dos indices diferentes
for i, j in zip(aj,bj):
  Aj_t.append(i*(alpha_ini_rad-alpha_L0)-1*(j*math.radians(Washout)))

#Distribucion de sustentacion
for i in range(n):
  aux2=0.0
  for j in range(n):
    aux2=2*rho0*((v_inf)**2)*b*Aj_t[j]*math.sin((j+1)*phi_rad[i])+aux2

  L_distribution_t.append(aux2)


#Ploteamos la curva solo para verificar que todo este de manera correcta.
plt.plot(Env_dis, L_distribution_t)
plt.title(f"Distribucion de sustentacion con torsion geometrica de {Washout} [°]")
plt.xlabel('Envergadura[m]')
plt.grid('minor')
plt.ylabel('Lift[N]')
plt.show()

#_________________________________________________________________________
#       Ala recta con torsion geometrica y despliegue de alerones
#
# Asumimos un despliegue simetrico de las superficies de control +-20 como maximo
# Definir cual baja y cual sube si baja -1 y si sube 1

W_left=-1
W_rigth=-1*(W_left)

#_________________________________________________________________________

#Angulo de despliegue de las superficies de control
delta= 20    #Grados
#Cuerda del aleron
c_aileron=0.5  #metros
#Longitud horizontal del aleron.
L_aileron=0.6  #metros

#Calculamos el inicio y final de cada uno de los angulos.
#En el sentido de cuerda a raiz en donde inician

i_aileron=0.10 #Inicio del aleron respecto ala cuerda de punta

#Calculo del final del aleron respecto a la medida que se dio
f_aileron=i_aileron+L_aileron

#Convertir estos valores a angulos para la discretizacion.

phii_aileron=  math.acos(((b/2)-f_aileron)/(b/2))       #Angulo de inicio en radianes por defecto
phif_aileron= math.acos(((b/2)-i_aileron)/(b/2))         #Angulo final en radianes por defecto


#Consideramos que posiblemente la razon de cuerda tendra valor diferente cuando el ala tiene un taper
#En ese caso se deberian calcular todas las razones de cuerda y esas ser funcion de

for i in range(n):
  #Calculo de la razon de superficie
  E_superficies=c_aileron/c
  E_cuerdas.append(E_superficies)


#Local airfol efectiveness

for i in E_cuerdas:
  aux=(math.acos(2*i-1)-math.sin(math.acos(2*i-1)))/math.pi
  Epsilon_phi.append(1-aux)

#Calculo del vector de terminos independientes x(phi)
#Si estan en el intervalo en el que definimos los alerones se asignamos lo valores de epsilon
#Si no se agregan ceros
for i in range(n):
  if(phi_rad[i]>=phif_aileron):
    if (phi_rad[i]<=phii_aileron):
      xi_phi.append(W_left*Epsilon_phi[i])

    elif(phi_rad[i]>=(math.pi-phii_aileron)):
      if(phi_rad[i]<=(math.pi-phif_aileron)):
        xi_phi.append(W_rigth*Epsilon_phi[i])
      else:
         xi_phi.append(0)
    else:
      xi_phi.append(0)
  else:
    xi_phi.append(0)

#Obtencion de los coeficientes cj

xi_phi_transpose=np.transpose(xi_phi)

#Calculando los coeficientes de apoyo
cj = np.matmul(Inv_ai_transpose, xi_phi_transpose)

#Damos el umbral para que los datos se vueelvan cero
umbral = 1e-15

# Reemplazar valores pequeños por cero
cj[abs(cj) < umbral] = 0.0

for i,j,k in zip(aj,bj,cj):
  Aj_t_d.append(i*(alpha_ini_rad-alpha_L0)-1*(j*math.radians(Washout))+k*math.radians(delta))

#Distribucion de sustentacion
for i in range(n):
  aux2=0.0
  for j in range(n):
    aux2=2*rho0*((v_inf)**2)*b*Aj_t_d[j]*math.sin((j+1)*phi_rad[i])+aux2

  L_distribution_t_d.append(aux2)

  #Ploteamos la curva solo para verificar que todo este de manera correcta.
plt.plot(Env_dis, L_distribution_t_d)
plt.title(f"Distribucion de sustentacion con torsion geometrica de {Washout} [°] y despliegue de superficies de {delta} [°]")
plt.xlabel('Envergadura[m]')
plt.grid('minor')
plt.ylabel('Lift[N]')
plt.show()

print(xi_phi)
