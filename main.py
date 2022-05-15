import random
import Prediccion as pda
from tqdm import tqdm
from time import sleep
#Rango normal de ptencial de accion es de -70mv a +40mv
#Rango del Oximetro es de normales es de 95 a 100 
#Rangos Bajos 90 a 95(Este es rango el cual marca una oxigenacion baja en el cuerpo)
Datos_demo_ner=[-70,30,10,-60,5,20,1,-10]
Datos_oximetro=[91,100,90,95,98,93,92,]
img=['Prueba.jpeg','Pie_bueno.jpeg','Pie_malo.jpg']
img2=random.choice(img)
print("******Demo CareDiabetics********")
r=pda.predict(img2)
tareas=["R","F","N"]
for i in tqdm(tareas):
    f=random.choice(Datos_oximetro)#ESto se cambiaria por una funcion que obtenga los datos del microcontrlador el cual captara esta infromacion
        #la teoria es que solo lleguen datos de este estilo
        
    n=random.choice(Datos_demo_ner)##ESto se cambiaria por una funcion que obtenga los datos del microcontrlador el cual captara esta infromacion
        #la teoria es que solo lleguen datos de este estilo
    sleep(0.6)
print(r,n,f)
print("Diagnostico")
for i in range(11):
    print("..")
if(r==1)and(f<=100)and(f>=95)and(n>=10):
    print("Riesgo Bajo")
if(r==1)and(f<=100)and(f>=95)and(n<10):
    print("Riego Medio")
if(r==0)and(f<=100)and(f>=95)and(n>=10):
    print("Riesgo Medio")
if(r==1)and(f<95)and(n>=10):
    print("Riesgo Medio Alto")
if(r==1)and(f<95)and(n<10):
    print("Riesgo Alto")
if(r==0)and(f<95)and(n<10):
    print("Riesgo Urgencia")

