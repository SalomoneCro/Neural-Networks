import numpy as np


class solver:

    # def euler(fun, interval, x_0, h, p):
    #     x_i = x_0
    #     t_i = interval[0]
    #     its =  int(np.abs(interval[0] - interval[1]) / np.abs(h))
    #     sol = [x_0]

    #     for _ in range(its - 1):
    #         x_i += h*fun(p,x_i,t_i)
    #         t_i += h
    #         sol.append(x_i)
    #     return sol

    def Runge_Kutta(fun, interval, x_0, h, p):
        x_i = x_0
        t_i = interval[0]
        its =  int(np.abs(interval[0] - interval[1]) / np.abs(h))
        sol = [x_0]

        for _ in range(its - 1):

            F1 = h*fun(p, t_i, x_i)
            F2 = h*fun(p, t_i + h/2, x_i + F1/2)
            F3 = h*fun(p, t_i + h/2, x_i + F2/2)
            F4 = h*fun(p, t_i + h, x_i + F3)
            x_i += (F1 + 2*F2 + 2*F3 + F4)/6
            t_i += h
            sol.append(x_i)
        return sol
    

    def euler_step(f,x,t,h,p):
        """
        Calcula un paso de integración del método de Euler.

        Argumentos de entrada:

            f : R^n,R -> R^n
            x = x(t) : R^n
            t = tiempo : R
            h = paso de tiempo : R
            p = parametros : R^q

        Retorna aproximacion numérica de

            x(t+h) : R^n

        según el método de Euler.

        # Ejemplos:
        """
        return x+h*f(x,t,p)

    def euler(f,xa,a,b,k,p,c=lambda x,t,p:x):
        """
        Integra numéricamente la ODE

            dx/dt = f(x,t)

        sobre el intervalo t:[a,b] usando k pasos de integración y el método m, bajo condicion inicial x(a)=x0.
        No es necesario que a<b.

        Argumentos de entrada:

            m = metodo de integracion (ej. euler, rk2, etc.)
            f : R^n -> R^n
            xa = condicion inicial : R
            a = tiempo inicial : R
            b = tiempo final : R
            k = num. pasos de integracion : N
            p = parametros : R^q
            c = función condicionante : R^n,R,p -> R^n

        Retorna:

            t : R^{k+1} , t_j = a+j*h para j=0,1,...,k
            w : R^{n,k+1} , w_ij = x_i(t_j) para i=0,1,...,n-1 y j=0,1,...,k

        donde a+k*dt = b.
        """
        assert k>0
        n = len(xa)
        h = (b-a)/k
        w = np.zeros((n,k+1)) # Produce un array con forma y tipo especificada con los parametros,
                            # lleno de ceros. la forma puede ser espcificada con un entero o tupla (n,k+1)
        t = np.zeros(k+1)
        w[:,0] = xa           # actualiza la posicion inicial (columna de indice 0) de las variables con los valores
                            # de las condiciones iniciales
        t[0] = a              # actualiza la posicion cero con el valor del tiempo inicial

        for j in range(k):    #Aca se produce la iteración en j

            t[j+1] = t[j] + h                # iteracion tiempo 
            w[:,j+1] = w[:,j]+h*f(w[:,j],t[j],p)  # iteracion de w
            w[:,j+1] = c(w[:,j+1],t[j+1],p)  # condicion sobre w

        return t,w
