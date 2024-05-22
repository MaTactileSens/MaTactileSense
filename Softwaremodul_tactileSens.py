import rtde_control
import rtde_receive
import rtde_io
import time
from datetime import datetime
import numpy as np
from tempfile import TemporaryFile
import math
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

##-1-##
#initialize RTDE commands
rtde_c = rtde_control.RTDEControlInterface("192.168.56.101", 250)
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101", 250)
rtde_i = rtde_io.RTDEIOInterface("192.168.56.101")

# Commands
pose = rtde_r.getActualTCPPose()
force = rtde_r.getActualTCPForce()

def get_robot_pose():
    pose = rtde_r.getActualTCPPose()
    pose[2] = pose[2] #- z_offset
    return pose

def get_robot_force():
    return rtde_r.getActualTCPForce()

def move_to_pose(pose, spd=0.3, acc=0.1):
    print("Robot moves to ", pose)
    new_pose = pose.copy()
    new_pose[2] = new_pose[2] #+ z_offset
    rtde_c.moveL(new_pose, spd, acc)


def move_until_force(rtde_r, rtde_c, pose, spd, acc, force_limit, axes="all"):

    collision = True
    compare_val = np.array(rtde_r.getActualTCPForce()) ##earlier force? if yes why not out of function?##

    rtde_c.moveL(pose, spd, acc, True) # truebedeutet "asinc"= progr wartet nicht bis robi an pos
    stime = time.time()
    if axes=="all":
        while(np.linalg.norm(np.array(rtde_r.getActualTCPForce())-compare_val) < force_limit): # Checks if current force - compare_force id greater than force_limit
            if np.allclose(np.array(rtde_r.getActualTCPPose())[:3], np.array(pose)[:3], atol=0.001): # True if current pose within 1mm of desired pose - WHY TRUE WHEN CLOSE?
                collision = False
                break

    elif axes=="z":
        while(np.abs(rtde_r.getActualTCPForce()[2]-compare_val[2]) < force_limit):
            if np.allclose(np.array(rtde_r.getActualTCPPose())[:3], np.array(pose)[:3], atol=0.001):
                collision = False
                break
    rtde_c.stopL(2.0, False) # 2.0 is acc ##ACC=False, why not SPD=false??##
    
    return collision

## Erstellung der Übertragungsfunktion mit Sollwerten. Falls Abhängigkeit gegeben, wäre so das optimale ergbnis zzu erzielen, welches sich dann mit dem über die calc-values vergleichen lässt
#keine daten im modell für die gilt theta_soll<10°

def reg_static_theta(X):
    X = np.array(X)

    #Koeffizienten für p66 Funktion
    coefficients = np.array([ 2.14731038e-02,  1.68337725e-01,  1.68873240e-01, -1.04930142e-02,
 -7.23830453e-03, -1.03248563e-02,  1.04317087e-03,  1.25254890e-04,
  9.23123887e-05,  3.12804878e-04, -4.09064348e-05, -2.02205612e-06,
 -9.15804905e-07, -4.95261889e-07, -4.56158261e-06,  7.03787853e-07,
  2.05046875e-08, 4.96297396e-09,  2.74104542e-09,  1.19409147e-09,
  2.57260046e-08, -4.77520614e-09,  1.00129592e-10, -6.25917200e-11,
  1.06233772e-12, -3.16879856e-12, -1.06753495e-12,])

    # Achsenabschnitt (Intercept)
    intercept = 4.001763318521029


    # Erstellen der erweiterten Merkmalsmatrix
    X1, X2 = X[:, 0], X[:, 1]
    features = np.column_stack([
        X1, X2, X1**2, X1*X2, X2**2, X1**3, X1**2*X2, X1*X2**2, X2**3,
        X1**4, X1**3*X2, X1**2*X2**2, X1*X2**3, X2**4,
        X1**5, X1**4*X2, X1**3*X2**2, X1**2*X2**3, X1*X2**4, X2**5,
        X1**6, X1**5*X2, X1**4*X2**2, X1**3*X2**3, X1**2*X2**4, X1*X2**5, X2**6
    ])

    # Berechnung der Regressionsgleichung inklusive des Achsenabschnitts
    Z = intercept + np.dot(features, coefficients)

    return Z

Phi_static_offset = -0.03641293069918843 # offset in Grad:-2.086307248765847

def reg_dynamic_t(X):
    X = np.array(X)

    #Koeffizienten für p333 Funktion
    coefficients =np.array([ 1.03310228e+01,  1.14371897e+02,  2.09512634e-02, -9.55088755e+00,
 -3.13290245e+02, -2.11417258e-02, -4.19718254e+02, -8.79374184e-02,
 -5.68923044e-06,  2.47128372e+00,  1.37905317e+02, -4.05867143e-04,
  5.90635402e+02,  3.69076037e-02,  5.66980316e-05,  4.46700093e+02,
  4.54529913e-02,  1.82374188e-04, -1.37308959e-07])

    intercept =-1.0462880236907206
    
    X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
    features = np.column_stack([
        X1, X2, X3,
        X1**2, X1*X2, X1*X3, X2**2, X2*X3, X3**2,
        X1**3, X1**2*X2, X1**2*X3, X1*X2**2, X1*X2*X3, X1*X3**2, X2**3, X2**2*X3, X2*X3**2, X3**3
    ])

    Z = intercept + np.dot(features, coefficients)

    return Z

Phi_dynamic_offset= 0.16226695980534345 #entspricht degree 9.297211951265151


def relative_rotation(current_pose, relative_rpy, order="xyz", degrees=False):    
    if degrees:
        relative_rpy = list(np.radians(relative_rpy))
 
    current_rvec = current_pose[3:]
    current_rmat = Rotation.from_rotvec(current_rvec, degrees=False).as_matrix()
    relative_rmat = Rotation.from_euler(order, relative_rpy, degrees=False).as_matrix()
    result_rmat = np.matmul(relative_rmat, current_rmat)
    rot = Rotation.from_matrix(result_rmat).as_rotvec(degrees=False)
    return list(rot)

def relative_rotation_reverse(relative_rotation, order="xyz", degrees=False):
    #INPUT      : Gegebene Ausrichtung (Vektor)-> Gegebene Ausrichtung Matrix [C]
    #Prozess    : wird ebenso wie der Vetor des senktrecht nach unten stehenden Greifers [B] zur Rotationsmatrix formatiert.
    #             Diese wird dann Invertiert[B^-1] und mit der Input-Matrix Multipliziert A*B=C <=> A=C*B^-1
    #OUTPUT     : [A] beschreibt die Euler-Rotation von B nach C
    if degrees:
        relative_rpy = list(np.radians(relative_rpy))
 
    rot=relative_rotation
    current_rvec=[0, 3.142, 0]
    result_rmat= Rotation.from_rotvec(rot, degrees=False).as_matrix()
    current_rmat = Rotation.from_rotvec(current_rvec, degrees=False).as_matrix() 
    current_rmat_inv=np.linalg.inv(current_rmat)
    relative_rmat=np.matmul(result_rmat, current_rmat_inv)
    relative_rpy=Rotation.from_matrix(relative_rmat).as_euler("xyz",degrees=False)
    return relative_rpy    #Rx,Ry,Rz zurück

def BackToPi(Rx=np.sqrt(0.1),Ry=np.sqrt(0.1)):
    #   liefert bei Umlaufender angestellter Rotation um eine Achse den Rotationswinkel und den Anstellwinkel
    Anstellwinkel=np.sqrt(Rx**2+Ry**2)
    
    
    Rxx=Rx/Anstellwinkel 
    Rxx=Rxx.round(8)
    Ryy=Ry/Anstellwinkel
    Ryy=Ryy.round(8)

    #Rotationswinkel über Rx und Ry und Fallunterscheidung
    PiRx=math.asin(Rxx)
    PiRy=math.acos(Ryy)

    if PiRx >=0:
        return PiRy, Anstellwinkel
    elif PiRx<0:
        return 2*np.pi-PiRy, Anstellwinkel

def winkel_norm(angles): # begrenzt den winkel auf 0 bis 2Pi
    return angles % (2 * np.pi)   

dt= datetime.now().strftime("%Y%m%d_%H%M%S")
##-2-##
### PROGRAM
## Struktur der Record-datei:
## sind die eindeutigen zeiträume vorh., ist es sehr einfach mit ihnen zu arbeiten, ohne sie ist es hingegen sehr mühsam. Daher: mehr ist mehr.
#0  1   2  3    4   5   6   7    8  9   10  11  12      13      14      15      16      17      18      19      20                               
#X  Y   Z  t0   t1  t2  t3  t4   t5 t6  t7  t8  i(t0)   i(t1)   i(t2)   i(t3)   i(t4)   i(t5)   i(t6)   i(t7)   i(t8)
recordheader=["X","Y","Z","t0","t1","t2","t3","t4","t5","t6","t7","t8","i(t0)","i(t1)","i(t2)","i(t3)","i(t4)","i(t5)","i(t6)","i(t7)","i(t8)"]
record=np.zeros((1,21))


### PARAMETER ###
##  Start-Pose ##
pose1 =[-0.0175, -0.8331, 0.112, 0, 3.142, 0]   # Start-Pose 
pose2=pose1.copy()
pose2[2]=pose2[2]-0.15                          # Ziel-Pose

## Parameter für die Kreisbahnberechnung am Kontaktpunkt ##
Anstellwinkel=10*np.pi/180
kreis=np.linspace(0,2*np.pi,73)                 #Intervall [0,2Pi]            
pos_blend=np.zeros((73,9))                      #Array für die Kreisbahnkoordianten 
Rx=Anstellwinkel*np.sin(kreis).round(16)        #X-Anteil
Ry=Anstellwinkel*np.cos(kreis).round(16)        #Y-Anteil
Rz=np.zeros(73)                                 #Y-Anteil=0
velocity = 0.5                                  
acceleration = 0.5
blend_1 = 0.0038                                #Glättungsfaktor für die Kreisbahn
Rueck=0.04*np.sin(Anstellwinkel)*0.5            #Rückstellung

##-3-##
### PROGRAMMSTART ###

rtde_r.startFileRecording(f"objectLoc_dataStream_{dt}.csv")
offset_time=time.time()                                 # Referenzzeitpunkt

print(f'Verfahren zu Ausgangskoordinaten: {pose1}') # 1. Fahren zur Startpose
move_to_pose(pose1)
record[:,:3] = pose2[:3]
record[:,3] = time.time() - offset_time                 # Zeitpunkt t0 in Ausgangsposition

print(f'Verfahren entlang Z- zu Zielkoordinaten: {pose2}') # 2. Fahren zur Zielpose

move_until_force(rtde_r, rtde_c, pose=pose2, spd=0.05, acc=0.05, force_limit=10, axes="all")
record[:,4] = time.time() - offset_time                 # Zeitpunkt t1

move_until_force(rtde_r, rtde_c, pose=pose2, spd=0.005, acc=0.005, force_limit=20, axes="all")
record[:,5] = time.time() - offset_time                 # Zeitpunkt t2

time.sleep(0.4)
record[:,6] = time.time() - offset_time             # Zeitpunkt t3    -   Ende Kraftplateau / Start Rückbewegung

move_until_force(rtde_r, rtde_c, pose=pose1, spd=0.005, acc=0.005, force_limit=10, axes="all")
record[:,7] = time.time() - offset_time             # Zeitpunkt t4    -   Erste Rückbewegung abgesl. /Kippen um 10 Grad

pose3=get_robot_pose()
liste_kreis=[]
for i in range(73):                                 # Erstellen der oordinatenmatrix der Kreisbahn auf Grundlage der aktuellen Position                    
    pos_blend[i,:6]=pose3.copy()
    pos_blend[i,3:6]=relative_rotation(current_pose=pose3,relative_rpy=[Rx[i],Ry[i],0])
    pos_blend[i,6]=velocity
    pos_blend[i,7]=acceleration
    pos_blend[i,8]=blend_1
    liste_kreis.append(pos_blend[i].tolist())
liste_kreis[-1][8]=0                                # der letzte Blend-Faktor muss Null gesetzt werden, damit die Bahn sauber Ausgeführt wird.
move_to_pose(liste_kreis[0][:6])
record[:,8] = time.time() - offset_time             # Zeitpunkt t5    -   kippung theta_GS=10° erreicht / Start Kreisbewegung Theta 0°->360


rtde_c.moveL(liste_kreis)
record[:,9] = time.time() - offset_time             # Zeitpunkt t6    -   Kreisbewegung beendet / kippung zurück zu theta_GS=0°
move_to_pose(pose3)

record[:,10] = time.time() - offset_time            # Zeitpunkt t7    -   Greifer senkrecht/ zurück zu pose 1
print(f'Verfahren entlang Z+ zu Ausgangskoordinaten: {pose1}')
move_to_pose(pose1)

record[:,11] = time.time() - offset_time            # Zeitpunkt t8    -   in Ausgangpose angekommen
rtde_r.stopFileRecording()
### ENDE BEWEGUNG ROBOTER ###



### AUSWERTUNG ###
##-4-##

#Laden des Datenstreams#
data=pd.read_csv(f"objectLoc_dataStream_{dt}.csv", sep=',', header=0, dtype=np.float64)
dataheader=data.columns.values.tolist()
data=data.to_numpy()

# Erstellung einer Datei für die Analyseergebnisse
## Extrahierte daten zum Zeitpunkt (t2+t3)/2:
#0  1   2   3   4   5   6  7   8   9   10  11 
#X  Y   Z   Rx  Ry  Rz  Fx Fy  Fz  Mx  My  Mz   
header1=["X","Y","Z","Rx","Ry","Rz","Fx","Fy","Fz","Mx","My","Mz"]
analyse=np.zeros((1,31))

data[:,0]=data[:,0]-data[0,0]               # Offset Zeit Falls meherer Anfahrmanöver
data[:,67:73]=data[:,67:73]-data[0,67:73]   # Offset Lasten Falls meherer Anfahrmanöver

#Ermittlung der Indizes der Zeitpunkte t0-t8
n=0                                                                         # Bei Erweiterung mit n>0 
record[:,12]=t0=(np.argmin(abs(data[:,0]-record[n,3]))).astype(np.int64)    
record[:,13]=t1=(np.argmin(abs(data[:,0]-record[n,4]))).astype(np.int64)   
record[:,14]=t2=(np.argmin(abs(data[:,0]-record[n,5]))).astype(np.int64)   
record[:,15]=t3=(np.argmin(abs(data[:,0]-record[n,6]))).astype(np.int64)   
record[:,16]=t4=(np.argmin(abs(data[:,0]-record[n,7]))).astype(np.int64)   
record[:,17]=t5=(np.argmin(abs(data[:,0]-record[n,8]))).astype(np.int64)
record[:,18]=t6=(np.argmin(abs(data[:,0]-record[n,9]))).astype(np.int64)   
record[:,19]=t7=(np.argmin(abs(data[:,0]-record[n,10]))).astype(np.int64)   
record[:,20]=t8=(np.argmin(abs(data[:,0]-record[n,11]))).astype(np.int64)

# offset zum zeitpunkt t0 von Prozesskräften Abziehen im Zeitraum t0-t8:
data[t0:t8+1,67]=data[t0:t8+1,67]-data[t0,67]
data[t0:t8+1,68]=data[t0:t8+1,68]-data[t0,68]
data[t0:t8+1,69]=data[t0:t8+1,69]-data[t0,69]
data[t0:t8+1,70]=data[t0:t8+1,70]-data[t0,70]
data[t0:t8+1,71]=data[t0:t8+1,71]-data[t0,71]
data[t0:t8+1,72]=data[t0:t8+1,72]-data[t0,72]

# Speichern t0-t8 in DataStream
data[t0:t1+1,-1]=0
data[t1:t2+1,-1]=1
data[t2:t3+1,-1]=2
data[t3:t4+1,-1]=3
data[t4:t5+1,-1]=4
data[t5:t6+1,-1]=5
data[t6:t7+1,-1]=6
data[t7:t8+1,-1]=6
data[t8:,-1]=8

#Lasten zum Zeitpunkt des Kraftplateaus (t2+t3)/2:
analyse[:,0]= data[round((t2+t3)/2),67-12] #x
analyse[:,1]= data[round((t2+t3)/2),68-12] #y
analyse[:,2]= data[round((t2+t3)/2),69-12] #z
analyse[:,3]= data[round((t2+t3)/2),70-12] #Rx
analyse[:,4]= data[round((t2+t3)/2),71-12] #Ry
analyse[:,5]= data[round((t2+t3)/2),72-12] #Rz
analyse[:,6]= data[round((t2+t3)/2),67] #Fx
analyse[:,7]= data[round((t2+t3)/2),68] #Fy
analyse[:,8]= data[round((t2+t3)/2),69] #Fz
analyse[:,9]= data[round((t2+t3)/2),70] #Mx
analyse[:,10]= data[round((t2+t3)/2),71] #My
analyse[:,11]= data[round((t2+t3)/2),72] #Mz

## Errechnen von Winkellagen, statischer Ansatz
#14     15               16                 17      18              19             
#Fres   Phi_calc(Fres)   theta_calc(Fres)   Mres    Phi_calc(Mres)  leer  
header2=["Fres", "Phi_calc(Fres)", "theta_calc(Fres)", "Mres", "Phi_calc(Mres)","leer1"]
analyse[:,12]=np.sqrt(analyse[:,6]**2+analyse[:,7]**2)            #Fres~Fx,Fy  
analyse[:,13]=np.arctan2(analyse[:,7],analyse[:,6])+np.pi         #Phi_calc(Fres)
analyse[:,14]=np.arctan2(analyse[:,12],analyse[:,8])              #theta_calc~Fres,Fz
analyse[:,15]=np.sqrt(analyse[:,9]**2+analyse[:,10]**2)           #Mres~Mx,My
analyse[:,16]=np.arctan2(analyse[:,10],analyse[:,9])+3*np.pi/2    #Phi_calc(Mres) ohne Formhand #+np.pi/2, mit +3*np.pi/2  # 

## Regression STATIC
#20             21                  22                      23      24      25                                
#ThetaRegression  PhiRegression     thetaRegression>15deg   leer    leer    leer            
header3=["theta_static_mod(deg)","Phi_static_mod(deg)","theta_static_mod>18.279deg"]

analyse[:,18]=reg_static_theta(X=[[analyse[:,14]*180/np.pi,analyse[:,13]*180/np.pi]]) # input kalkulierte Werte: 1. Theta_calc, 2. Phi_calc(Fres)
analyse[:,19]=(winkel_norm(analyse[:,13]-Phi_static_offset))*180/np.pi
analyse[:,20]=np.where(analyse[:,18]>18.279,True,False)
analyse[:,30]=np.where(analyse[:,28]<5.04,True,False)



## ANALYSE DYNAMISCHER ANSATZ ##
##-5-##
#Erweiterung des Datastreams#
dataheader=[*dataheader,"Rotationswinkel","Anstellwinkel","Resultierende Fxy", "Resultierende Mxy","Zeitraum"] #Speichern und erweitern des headers 
data=np.c_[data,np.zeros(len(data)),np.zeros(len(data)),np.zeros(len(data)),np.zeros(len(data)),np.zeros(len(data))] # Umwandeln Dataframe in Numpy und Erweiterung

for i in data:                                                      # Rotations- und Anstellwinkel
    Rx,Ry,Rz=relative_rotation_reverse(relative_rotation=i[58:61])
    i[-5:-3]=BackToPi(Rx=Rx,Ry=Ry)

data[:,-5]= 2*np.pi-data[:,-5]                                      # Anpassung d.Rotationswinkels auf Roboterkoordinatensystem
data[:,-3]= np.sqrt(data[:,67]**2+data[:,68]**2)                    # Resultierende Fxy
data[:,-2]= np.sqrt(data[:,70]**2+data[:,71]**2)                    # Resultierende Mxy


header4=["Fz_max", "Fz_min", "spanne", "P50", "cv", "RatioF100", "kontrolle"]

Pi=np.array(data[t5:t6,-5])          
Fx=np.array(data[t5:t6,67])
Fz=np.array(data[t5:t6,69]) 
Mx=np.array(data[t5:t6,70])         
My=np.array(data[t5:t6,71])          
Mz=np.array(data[t5:t6,72])

analyse[:,21]= FzMax=max(Fz)  
analyse[:,22]= FzMin=min(Fz) 
analyse[:,23]= spreadTotal=FzMax-FzMin


#P50
percent=0.5
sortFz=sorted(Fz,reverse=True)                          #   Fz der Größe nach Sortiert
RoundSortFz=round(len(Fz)*percent)                      #   Anzahl der Werte der Obersten 50-Prozent von Fz in ganzen Zahlen
PercentFz=sortFz[:RoundSortFz]                          #   Die Obersten 50-Prozent von Fz 
PercentFz_index=[]
PercentPi=[]


#herausfinden: 1. Indexe der obersten 50%, dann 2. winkelkleagen 
for b in range(len(PercentFz)):
    PercentFz_index=(round(np.mean(np.where(abs(Fz-PercentFz[b])==0)[0])))  # Da tw. die exakt selben Fz's mit verschiedenen Pi's existieren[wert bleibt konsantüber meherere i], diese sich dann also doppeln wird der Schnitt des indexed genommen und dieser dann gerundet
    PercentPi=np.append(PercentPi,Pi[PercentFz_index])                      # Die Winkel der x-Prozent höchsten Fz werden in persentPi gespeichert

    #Berechnung des Mittelpunktes der x-Prozent höchsten Fz in Kartesischen koordinaten da keine Fallunterscheidung zwischen 0pi/2pi/4p usw notwendig
r=PercentFz
rho=PercentPi 
XX=np.cos(rho)*r    #Fz-Anteil der auf die x-Richtung zurückzufürhen ist
YY=np.sin(rho)*r    #Fz-Anteil der auf die y-Richtung zurückzufürhen ist

MeanXX=np.mean(XX)  #Durchschnittlicher x-Anteil der obertsen x-prozent
MeanYY=np.mean(YY)  #Durchschnittlicher y-Anteil der obertsen x-prozent

MeanPi=np.arctan2(MeanYY,MeanXX)    #zurückrechnen in polkoordinaten # GESUCHTER WERT

if MeanPi<0:
    MeanPi=MeanPi+2*np.pi           #da arctan2[-pi;pi] verschiebung nach [0;2Pi]
analyse[:,24]=MeanPi

#### Errechnung der statistsichen Metriken des dynamischen Ansatzes
num_segments = 100
segment_length = len(Fz) // num_segments
segment_swings = []
segment_variance = []

for i in range(num_segments):
    segment = Fz[i*segment_length : (i+1)*segment_length]
    swing = segment.max() - segment.min()
    segment_swings.append(swing)
    variance = segment.var()
    segment_variance.append(variance)

Ratio100_Fz= np.mean(segment_swings)/(Fz.max() - Fz.min())
cv = Fz.std() / Fz.mean()

analyse[:,25]=Ratio100_Fz
analyse[:,26]=cv
analyse[:,27]=123456789 # kontrollwert
      
### Model dynamischer Ansatz ###
header5=["t_dynamic_mod(cm)","phi_dynamic_mod(deg)","t_static_mod<5,04cm" ]

analyse[:,28]=reg_dynamic_t(X=[[analyse[:,26],analyse[:,25],analyse[:,24]]]) # input kalkulierte Werte: 1. cv, 2. Ratio100_Fz, 3.P50
analyse[:,29]=(winkel_norm(analyse[:,24]-Phi_dynamic_offset))*180/np.pi
analyse[:,30]=np.where(analyse[:,28]<5.04,True,False)


###AUSGABE##
##-6-##
#staticher Ansatz#
print("static theta in deg: ", analyse[:,18])
if analyse[:,20]:
    print("horizontale Lage der Fläche static phi in deg: ", analyse[:,19])
else:
    print("horizontale Lage der Fläche static phi nicht zuverlässig bestimmbar, da Neigungswinkel theta_mod zu gering")

#dynamsicher Ansatz
print("dynamic t in cm: ", analyse[:,28])

if analyse[:,30]:
    print("horizontale Lage der Flächenkante dynamic phi in deg: ",analyse[:,29])
else:
    print("horizontale Lage der Flächenkante dynamic phi nicht zuverlässig bestimmbar, da Überlappung t_mod mit Fläche zu groß")



###SPEICHERN##
##-7-##

# Speichern recorddatei mit den Markern für die relevanten Zeitpunkten zwischen t0 und t8 
dfrecord=pd.DataFrame(record, columns=recordheader) # hier werden die zeiten gespeichert
dfrecord.to_csv(f"objectLoc_TIME{dt}.csv")

# Speichern Analysedatei zur statischen Winkelberechnung für t=(t2+t3)/2
header=[*header1,*header2, *header3,*header4,*header5]
print("analyse: ",analyse.shape)
print("header: ",len(header))
dfanalyse=pd.DataFrame(analyse, columns=header)
dfanalyse.to_csv(f"objectLoc_RESULTS{dt}.csv")

#Speichern Erweiterten DataStream
dfdata=pd.DataFrame(data, columns=dataheader)
dfdata.to_csv(f"objectLoc_DATA{dt}.csv")


rtde_c.stopScript()