
import numpy as np
from sensor_msgs.msg import JointState
import numpy as np

from numpy.linalg import pinv
from copy import copy
# from pyquaternion import Quaternion

from markers import *
from functions import*

cos=np.cos; sin=np.sin; pi=np.pi

def dh(d, theta, a, alpha):
 """
 Calcular la matriz de transformacion homogenea asociada con los parametros
 de Denavit-Hartenberg.
 Los valores d, theta, a, alpha son escalares.
 """
 # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
 T = np.array([
    [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
    [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
    [0, np.sin(alpha), np.cos(alpha), d],
    [0, 0, 0, 1]
 ])
 return T
    
    

def fkine(q):
 """
 Calcular la cinematica directa del brazo robotico dados sus valores articulares. 
 q es un vector numpy de la forma [q1, q2, q3, ..., qn]
 """
 # Longitudes (en metros)
 # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
 T1 = dh(0.100, q[0] + np.pi, 0.0, np.pi/2)
 T2 = dh(0.300, q[1], 0.0, np.pi/2) 
 T3 = dh(0.250, q[2], 0.0, -np.pi/2) 
 T4 = dh(0.300, q[3] + np.pi, 0.0, np.pi/2)
 T5 = dh(0.300, q[4], 0.0, -np.pi/2) 
 T6 = dh(0.140, q[5], 0.0, np.pi/2)
 # Efector final con respecto a la base
 T = T1 @ T2 @ T3 @ T4 @ T5 @ T6
 return T


def jacobian_position(q, delta=0.0001):
 """
 Jacobiano analitico para la posicion de un brazo robotico de n grados de libertad. 
 Retorna una matriz de 3xn y toma como entrada el vector de configuracion articular 
 q=[q1, q2, q3, ..., qn]
 """
 # Determinar la cantidad de articulaciones
 n = len(q)
 # Crear una matriz 3xn 
 J = np.zeros((3, n)) 
 # Calcular la transformacion homogenea inicial (usando q)
 T0 = fkine(q)
 p0 = T0[0:3, 3]
    
 # Iteracion para la derivada de cada articulacion (columna)
 for i in range(n):
  # Copiar la configuracion articular inicial
  dq = copy(q)
  # Calcular nuevamenta la transformacion homogenea e
  # incrementar la articulacion i-esima usando un delta, 
  # usar la copia de configuración inicial
  dq[i] += delta
  # Transformacion homogenea luego del incremento (q+delta)
  Ti = fkine(dq)
  # Aproximacion del Jacobiano de posicion usando diferencias finitas
  pi = Ti[0:3, 3]
  J[:, i] = (pi - p0) / delta
 return J


def jacobian_pose(q, delta=0.0001):
 """
 Jacobiano analitico para la posicion y orientacion (usando un
 cuaternion). Retorna una matriz de 7xn y toma como entrada el vector de
 configuracion articular q=[q1, q2, q3, ..., qn]
 """
 n = q.size
 J = np.zeros((7,n))
 # Implementar este Jacobiano aqui
 
 return J


def ikine(xdes, q0):
 """
 Calcular la cinematica inversa de un brazo robotico numericamente a partir 
 de la configuracion articular inicial de q0. Emplear el metodo de newton.
 """
 epsilon  = 0.001
 max_iter = 1000
 delta    = 0.00001

 q  = copy(q0)

 for i in range(max_iter):
  # Main loop
  T = fkine(q)
  x = T[0:3, 3]
  e = xdes - x

  if np.linalg.norm(e) < epsilon:
    break

  J = jacobian_position(q, delta)
  try:
    dq = np.linalg.pinv(J) @ e
  except np.linalg.LinAlgError:
    print("El Jacobiano es singular.")  
    break
  #pass
  
  q = q + dq

 return q


def ik_gradient(xdes, q0):
 """
 Calcular la cinematica inversa de un brazo robotico numericamente a partir 
 de la configuracion articular inicial de q0. Emplear el metodo gradiente.
 """
 epsilon  = 0.001
 max_iter = 1000
 delta    = 0.00001

 q  = copy(q0)
 for i in range(max_iter):
  # Main loop
  pass
    
 return q

    
def rot2quat(R):
 """
 Convertir una matriz de rotacion en un cuaternion

 Entrada:
  R -- Matriz de rotacion
 Salida:
  Q -- Cuaternion [ew, ex, ey, ez]

 """
 dEpsilon = 1e-6
 quat = 4*[0.,]

 quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
 if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
  quat[1] = 0.0
 else:
  quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
 if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
  quat[2] = 0.0
 else:
  quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
 if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
  quat[3] = 0.0
 else:
  quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

 return np.array(quat)


def TF2xyzquat(T):
 """
 Convert a homogeneous transformation matrix into the a vector containing the
 pose of the robot.

 Input:
  T -- A homogeneous transformation
 Output:
  X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
       is Cartesian coordinates and the last part is a quaternion
 """
 quat = rot2quat(T[0:3,0:3])
 res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
 return np.array(res)

#def PoseError(x,xd):
# """
# Determine the pose error of the end effector.

# Input:
# x -- Actual position of the end effector, in the format [x y z ew ex ey ez]
# xd -- Desire position of the end effector, in the format [x y z ew ex ey ez]
# Output:
# err_pose -- Error position of the end effector, in the format [x y z ew ex ey ez]
# """
# pos_err = x[0:3]-xd[0:3]
# qact = Quaternion(x[3:7])
# qdes = Quaternion(xd[3:7])
# qdif =  qdes*qact.inverse
# qua_err = np.array([qdif.w,qdif.x,qdif.y,qdif.z])
# err_pose = np.hstack((pos_err,qua_err))
# return err_pose