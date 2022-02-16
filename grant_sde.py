# 
# Module written by Kesiyan G.A., grant.kesiyan@mail.ru 
# Krasnodar, october, 2019.
#

import numpy as np
import statistics as stat
from scipy import optimize
# dx = sigma*dW и small_param*dx = sigma*dW
# xt = x0 + sigma * Wt
def MatrixAdditiveVineraMotion(x0, t0, t1, h, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    while k < len_:
        x[k,:] = x0 + sigma*np.sqrt(tt[k])*nrndVector[k,:]
        k = k + 1;
    VMean = x0*np.ones(len_)
    VVol = sigma*np.sqrt(tt - t0)
    return x #, VMean, VVol

def MatrixAdditiveVineraMotionSmallParam(x0, t0, t1, h, sigma, small_param, count):
    return MatrixAdditiveVineraMotion(x0, t0, t1, h, sigma/small_param, count)
    
# Матрица count - Винеровских блужданий с ЛИНЕЙНОЙ волатильностью
# dx = sigma*х*dW
# Input:
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   sigma <- волатильность (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
#           <- VMean(N x 1) вектор среднего процесса x
#           <- VVol(N x 1) вектор волатильности процесса x
def MatrixVineraMotionLinearVolatility(x0, t0, t1, h, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    while k < len_:
        x[k,:] = x0 * np.exp(-tt[k]*sigma*sigma/2 + sigma*np.sqrt(tt[k])*nrndVector[k,:])
        k = k + 1;
    VMean = x0
    VVol = x0*np.sqrt(np.exp(sigma*sigma*tt) - 1)
    return x #, VMean, VVol
               
def MatrixVineraMotionLinearVolatilitySmallParam(x0, t0, t1, h, sigma, small_param, count):
    return MatrixVineraMotionLinearVolatility(x0, t0, t1, h, sigma/small_param, count)
            
#Матрица count - Винеровских блужданий
# Wt = e * sqrt(t), e~N(0,1), W~N(0,t)
# Input: 
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
def MatrixVineraMotion(t0, t1, h, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_, count), dtype=float)
    k = 0
    nrndVector = np.random.normal(0, 1, (len_, count))
    while k < len_:
        x[k,:] = np.sqrt(tt[k])*nrndVector[k,:]
        k = k + 1;
    return x

# Две матрицы count-версий Аддитивного независимого дискретного случайного блуждания:
# --- 1. без малого и 2. с малым параметром
# Но с одинаковыми версиями случаного процесса в соответствующих столбцах
#                        и
# +Два вектора средневыборочных процессов
# В целях сравнительного анализа влияния малого параметра на процесс
# dx = sigma*dW и small_param*dx = sigma*dW
# xt = x0 + sigma * Wt
# Output:
#   return  <- M(N x count) матрица процесса,
#            <- VMean(N x 1) вектор среднего процесса для M
#            <- Msp(N x count) матрица процесса с малым параметром
#            <- VspMean(N x 1) вектор среднего процесса для Msp
def DMatrixAdditiveMotionSmallParam(x0, t0, t1, h, sigma, small_param, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    M = np.zeros((len_, count), dtype=float)
    M[0,:] = x0
    Msp = np.zeros((len_, count), dtype=float) #с малым параметром
    Msp[0,:] = x0
    sigma_sp = sigma/small_param
    k = 1
    nrndVector = np.random.normal(0, 1, (len_, count))
    while k < len_:
        M[k,:] = x0 + sigma*np.sqrt(tt[k])*nrndVector[k,:]
        Msp[k,:] = x0 + sigma_sp*np.sqrt(tt[k])*nrndVector[k,:]
        k = k + 1;
    return M, np.mean(M, axis=1), Msp, np.mean(Msp, axis=1)

# Вектор средневыборочных процессов Аддитивного независимого дискретного случайного блуждания с малым параметром
# small_param*dx = sigma*dW
# xt = x0 + sigma * Wt
# Output:
#   return <- VMean(N x 1) вектор среднего процесса для M
def VectorMeanAdditiveMotionSmallParam(x0, t0, t1, h, sigma, small_param, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    Msp = np.zeros((len_, count), dtype=float) #с малым параметром
    Msp[0,:] = x0
    sigma_sp = sigma/small_param
    k = 1
    nrndVector = np.random.normal(0, 1, (len_, count))
    while k < len_:
        Msp[k,:] = x0 + sigma_sp*np.sqrt(tt[k])*nrndVector[k,:]
        k = k + 1;
    return np.mean(Msp, axis=1)

#Матрица count-версий Аддитивного независимого дискретного случайного блуждания
# dx = sigma*dW
# xt = x0 + sigma * Wt
def MatrixAdditiveMotion(x0, t0, t1, h, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_, count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_, count))
    while k < len_:
        x[k,:] = x0 + sigma*np.sqrt(tt[k])*nrndVector[k,:] #Может так: np.sqrt(tt[k]-t0)?
        k = k + 1;
    return x

#Матрица count-версий Аддитивного независимого дискретного случайного блуждания с малым параметром
# small_param*dx = sigma*dW
def MatrixAdditiveMotionSmallParam(x0, t0, t1, h, sigma, small_param, count):
    return MatrixAdditiveMotion(x0, t0, t1, h, sigma/small_param, count)

# Матрица count - Винеровских блужданий со сносом (непрерывный Винеровский процесс)
# ---- ЭТО БАЗОВЫЙ ПРОЦЕСС ДЛЯ ИТО-ПРОЦЕССОВ ----
# dx = mu*dt + sigma*dW
# xt = x0 + mu*(t-t0) + sigma * W(t-t0)
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   mu  <- снос (float)
#   sigma <- волатильность (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
#            <- VMean(N x 1) вектор среднего процесса x
#            <- VVol(N x 1) вектор волатильности процесса x
def MatrixVineraMotionDrifted(x0, t0, t1, h, mu, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    while k < len_:
        x[k,:] = x0 + mu*(tt[k]-t0) + sigma*np.sqrt(tt[k]-t0)*nrndVector[k,:]
        k = k + 1;
    VMean = x0 + mu*(tt-t0) #TO Test
    VVol = sigma*np.sqrt(tt-t0) #TO Test
    return x #, VMean, VVol

def MatrixVineraMotionDriftedSmallParam(x0, t0, t1, h, mu, sigma, small_param, count):
    return MatrixVineraMotionDrifted(x0, t0, t1, h, mu/small_param, sigma/small_param, count)

# Матрица count - непрерывный Винеровский процесс с линейной волатильностью
# dx = mu*dt + sigma*x*dW
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   mu  <- снос (float)
#   sigma <- волатильность (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
#            <- VMean(N x 1) вектор среднего процесса x
#            <- VVol(N x 1) вектор волатильности процесса x
def MatrixVineraMotionDriftedLineVolatility(x0, t0, t1, h, mu, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0 #X ->>>>>>>>>> TODO <<<<<<<<<<<<<<<<
    #k = 1
    #nrndVector = np.random.normal(0, 1, (len_,count))
    #while k < len_:
    #    x[k,:] = x0 + mu*(tt[k]-t0) + sigma*np.sqrt(tt[k]-t0)*nrndVector[k,:]
    #    k = k + 1;
    VMean = x0 + mu*tt
    lamda = mu / (sigma*sigma)
    tt2 = [i*i for i in tt]
    VVol = np.sqrt(((x0+lamda)**2 + lamda**2)*(np.exp(tt*(sigma**2))-1) - 2*mu*(x0+lamda)*tt - mu*mu*tt2)
    return x #, VMean, VVol

def MatrixVineraMotionDriftedLineVolatilitySmallParam(x0, t0, t1, h, mu, sigma, small_param, count):
    MatrixVineraMotionDriftedLineVolatility(x0, t0, t1, h, mu/small_param, sigma/small_param, count)

def MatrixRungeKutta4order_VineraMotionLineVolatility(x0, t0, t1, h, mu, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    for i in range(count):
        x[:,i] = RungeKutta4order_VineraMotionLineVolatility(x0, t0, t1, h, mu, sigma)
    VMean = x.mean(axis=1)
    Vval = 0#stat.stdev(x)# - выборочная, несмещенная
    return x #,VMean, Vval

def MatrixRungeKutta4order_VineraMotionLineVolatilitySmallParam(x0, t0, t1, h, mu, sigma, small_param, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    for i in range(count):
        x[:,i] = RungeKutta4order_VineraMotionLineVolatilitySmallParam(x0, t0, t1, h, mu, sigma, small_param)
    VMean = x.mean(axis=1)
    Vval = 0#stat.stdev(x)# - выборочная, несмещенная
    return x #,VMean, Vval


# Вектор-Численное решение методом Рунге-Кутта 4 порядка Процесса Винера с Линейной волатильностью
# dx = mu*dt + sigma*x*dW
def RungeKutta4order_VineraMotionLineVolatility(x0, t0, t1, h, mu, sigma):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    dW = np.sqrt(h) * np.random.normal(0, 1, len_)
    k = 0
    while k < len_ - 1:
        k1 = mu*h + sigma*x[k]*dW[k]
        k2 = mu*h + sigma*(x[k]+k1/2)*dW[k]
        k3 = mu*h + sigma*(x[k]+k2/2)*dW[k]
        k4 = mu*h + sigma*(x[k]+k3)*dW[k]        
        x[k+1] = x[k] + (k1 + 2*k2 + 2*k3 + k4)/6 - sigma*sigma*x[k]*h
        k = k + 1
    return x

# Вектор-Численное решение методом Рунге-Кутта 4 порядка Процесса Винера с Линейной волатильностью с малым параметром
# small_param * dx = mu*dt + sigma*x*dW
def RungeKutta4order_VineraMotionLineVolatilitySmallParam(x0, t0, t1, h, mu, sigma, small_param):
    return RungeKutta4order_VineraMotionLineVolatility(x0, t0, t1, h, mu/small_param, sigma/small_param)
    
#Матрица count - точных решений логарифмического блуждания СДУ
# dx = mu * x * dt + sigma * x * dW
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   mu  <- снос (float)
#   sigma <- волатильность (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
#            <- VMean(N x 1) вектор среднего процесса x
#            <- VVol(N x 1) вектор волатильности процесса x
def MatrixExactLogMotion(x0, t0, t1, h, mu, sigma, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    sigma2 = sigma * sigma
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    while k < len_:
        x[k,:] = x0 * np.exp((mu - sigma2/2)*tt[k] + sigma*np.sqrt(tt[k])*nrndVector[k,:])
        k = k + 1;
    VMean = x0 * np.exp(mu*tt)
    VVol = VMean * np.sqrt(np.exp(tt*(sigma**2))-1)
    return x #, VMean, VVol

#Матрица count - точных решений логарифмического блуждания СДУ с малым параметром
# small_param * dx = mu*x*dt + sigma*x*dW
# dx = mu*x*dt/small_param + sigma*x*dW/small_param
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   mu  <- снос
#   sigma  <- волатильность (float)
#   small_param  <- малый параметр (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
def MatrixExactLogMotionSmallParam(x0, t0, t1, h, mu, sigma, small_param, count):
    return MatrixExactLogMotion(x0, t0, t1, h, mu / small_param, sigma / small_param, count)
    
#Матрица count - точных решений Процесса Орнштейна-Уленбека
# dx = -b*(x-a)*dt + sigma*dW
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   a  <- уровень притяжения (float)
#   b  <- сила притяжения (b > 0) (float)
#   sigma <- волатильность (float)
#   count <- количество версий (int)
# Output:
#   return  <- (count x N) матрица
def MatrixExactOrnUlen(x0, t0, t1, h, sigma, a, b, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    x0_a = x0-a
    sqrt2b = np.sqrt(2*b)
    while k < len_:
        x[k,:] = a + x0_a * np.exp(-b*tt[k]) + sigma*np.sqrt(1-np.exp(-2*b*tt[k]))*nrndVector[k,:] / sqrt2b
        k = k + 1;
    VMean = a + (x0 - a)*np.exp(-b*tt)
    VVol = sigma*np.sqrt(1 - np.exp(-2*b*tt))/np.sqrt(2*b)
    return x #, VMean, VVol

#Матрица count - точных решений Процесса Орнштейна-Уленбека с малым параметром
def MatrixExactOrnUlenSmallParam(x0, t0, t1, h, sigma, a, b, small_param, count):
    return MatrixExactOrnUlen(x0, t0, t1, h, sigma / small_param, a, b / small_param, count)
    
#Матрица count - точных решений Процесса Броуновский мост
# dx = -(x-a)/(T-t)*dt + sigma*dW
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   a  <- уровень притяжения (float)
#   T  <- x(T)=a (T>t) (float)
#   sigma <- волатильность (float)
#   count <- количество версий (int)
# Output:
#   return  <- (N x count) матрица
def MatrixExactBrownianBridge(x0, t0, t1, h, sigma, a, T, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    x0_a = x0-a
    T_t0 = T - t0
    while k < len_:
        x[k,:] = a + x0_a*(T-tt[k])/T_t0 + sigma*np.sqrt((tt[k]-t0)*(T-tt[k])/T_t0)*nrndVector[k,:]
        k = k + 1;
    VMean = a + x0_a*(T-tt)/T_t0
    VVol = sigma * np.sqrt((tt-t0)*(T-tt)/T_t0)
    return np.transpose(x) #, VMean, VVol

#Матрица count - точных решений Процесса Броуновский мост с малым параметром
def MatrixExactBrownianBridgeSmallParam(x0, t0, t1, h, sigma, a, T, small_param, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    x0_a = x0-a
    T_t0 = T - t0
    sigma_sp = sigma/small_param
    while k < len_:
        x[k,:] = a + x0_a*(T-tt[k])*small_param/T_t0 + sigma_sp*np.sqrt((tt[k]-t0)*(T-tt[k])/T_t0)*nrndVector[k,:]
        k = k + 1;
    VMean = a + x0_a*(T-tt)*small_param/T_t0
    VVol = sigma_sp * np.sqrt((tt-t0)*(T-tt)/T_t0)
    return x #, VMean, VVol
    
#Вектор - реализация схемы Элера численного решения СДУ
# dx = a(x,t)*dt + b(x,t)*dW
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   Axt <- функция сноса a(x,t) (<class 'function'>)
#   Bxt <- функция волатильности b(x,t)(<class 'function'>)
# Output:
#   return  <- (1 x N) Вектор
def VectorEilerScheme(x0, t0, t1, h, Axt, Bxt):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    sqrt_h = np.sqrt(h)
    k = 1
    nrndVector = np.random.normal(0, 1, len_)
    while k < len_:
        x[k] = x[k-1] + Axt(x[k-1],tt[k-1])*h + Bxt(x[k-1],tt[k-1])*sqrt_h*nrndVector[k]
        k = k + 1;
    return x

#Вектор - реализация схемы Элера численного решения СДУ с малым параметром
def VectorEilerSchemeSmallParam(x0, t0, t1, h, Axt, Bxt, small_param):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    h_sp = h / small_param
    sqrt_h_sp = np.sqrt(h) / small_param
    k = 1
    nrndVector = np.random.normal(0, 1, len_)
    while k < len_:
        x[k] = x[k-1] + Axt(x[k-1],tt[k-1])*h_sp + Bxt(x[k-1],tt[k-1])*sqrt_h_sp*nrndVector[k]
        k = k + 1;
    return x
    
#Матрица - реализация схемы Элера численного решения СДУ
def MatrixEilerScheme(x0, t0, t1, h, Axt, Bxt, count):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    sqrt_h = np.sqrt(h)
    for r in range(len_):
        for k in range(count):
            x[r,k] = x[r-1,k] + Axt(x[r-1,k],tt[r-1])*h + Bxt(x[r-1,k],tt[r-1])*sqrt_h*nrndVector[r,k]      
    return x
    
#Матрица - реализация схемы Элера численного решения СДУ с малым параметром
def MatrixEilerSchemeSmallParam(x0, t0, t1, h, Axt, Bxt, count, small_param):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros((len_,count), dtype=float)
    x[0,:] = x0
    k = 1
    nrndVector = np.random.normal(0, 1, (len_,count))
    h_sp = h / small_param
    sqrt_h_sp = np.sqrt(h) / small_param
    for r in range(len_):
        for k in range(count):
            x[r,k] = x[r-1,k] + Axt(x[r-1,k],tt[r-1])*h_sp + Bxt(x[r-1,k],tt[r-1])*sqrt_h_sp*nrndVector[r,k]
    return x
    #Вектор - реализация схемы Рунге-Кутта 4 порядка численного решения СДУ
    # dx = a(x,t)*dt + b(x,t)*dW
    # Input:
    # x0 <- Начальное значение (float)
    # t0 <- начальное значение t
    # t1 <- конечное значение t
    # h <- шаг (float)
    # Axt <- функция сноса a(x,t) (<class 'function'>)
    # Bxt <- функция волатильности b(x,t)(<class 'function'>)
    # DiffBxt <- b'(x,t), производная функции волатильности b(x,t)(<class 'function'>)
    # Output:
    # return <- (1 x N) Вектор
def VectorRungeKutta4order_(x0, t0, t1, h, Axt, Bxt, DiffBxt):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    dW = np.sqrt(h) * np.random.normal(0, 1, len_)
    k = 0
    while k < len_ - 1:
      k1 = Axt(x[k],tt[k])*h + Bxt(x[k],tt[k])*dW[k+1]
      k2 = Axt(x[k]+k1/2,tt[k]+h/2)*h + Bxt(x[k]+k1/2,tt[k]+h/2)*dW[k+1]
      k3 = Axt(x[k]+k2/2,tt[k]+h/2)*h + Bxt(x[k]+k2/2,tt[k]+h/2)*dW[k+1]
      k4 = Axt(x[k]+k3,tt[k+1])*h + Bxt(x[k]+k3,tt[k+1])*dW[k+1]
      x[k+1] = x[k] + (k1 + 2*k2 + 2*k3 + k4)/6 - DiffBxt(x[k],tt[k])*Bxt(x[k],tt[k])*h/2
      k = k + 1
    return x


#Вектор - реализация схемы Рунге-Кутта 4 порядка численного решения СДУ
# dx = a(x,t)*dt + b(x,t)*dW
# Input: 
#   x0 <- Начальное значение (float)
#   t0 <- начальное значение t
#   t1 <- конечное значение t
#   h  <- шаг (float)
#   Axt <- функция сноса a(x,t) (<class 'function'>)
#   Bxt <- функция волатильности b(x,t)(<class 'function'>)
#   DiffBxt <- b'(x,t), производная функции волатильности b(x,t)(<class 'function'>)
# Output:
#   return  <- (1 x N) Вектор
def VectorRungeKutta4order(x0, t0, t1, h, Axt, Bxt, DiffBxt):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    dW = np.sqrt(h) * np.random.normal(0, 1, len_)
    k = 0
    while k < len_ - 1:
        k1 = Axt(x[k],tt[k])*h + Bxt(x[k],tt[k])*dW[k]
        k2 = Axt(x[k]+k1/2,tt[k]+h/2)*h + Bxt(x[k]+k1/2,tt[k]+h/2)*dW[k]
        k3 = Axt(x[k]+k2/2,tt[k]+h/2)*h + Bxt(x[k]+k2/2,tt[k]+h/2)*dW[k]
        k4 = Axt(x[k]+k3,tt[k+1])*h + Bxt(x[k]+k3,tt[k+1])*dW[k]    
        x[k+1] = x[k] + (k1 + 2*k2 + 2*k3 + k4)/6 - DiffBxt(x[k],tt[k])*Bxt(x[k],tt[k])*h/2
        k = k + 1
    return x
    
#Вектор - реализация схемы Рунге-Кутта 4 порядка численного решения СДУ с малым параметром
def VectorRungeKutta4orderSmallParam(x0, t0, t1, h, Axt, Bxt, DiffBxt, small_param):
    tt = np.arange(t0,t1+h,h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    dW = np.sqrt(h) * np.random.normal(0, 1, len_)
    k = 0
    while k < len_ - 1:
        k1 = Axt(x[k],tt[k])*h/small_param + Bxt(x[k],tt[k])*dW[k]/small_param
        k2 = Axt(x[k]+k1/2,tt[k]+h/2)*h/small_param + Bxt(x[k]+k1/2,tt[k]+h/2)*dW[k]/small_param
        k3 = Axt(x[k]+k2/2,tt[k]+h/2)*h/small_param + Bxt(x[k]+k2/2,tt[k]+h/2)*dW[k]/small_param
        k4 = Axt(x[k]+k3,tt[k+1])*h/small_param + Bxt(x[k]+k3,tt[k+1])*dW[k]/small_param    
        x[k+1] = x[k] + (k1 + 2*k2 + 2*k3 + k4)/6 - DiffBxt(x[k],tt[k])*Bxt(x[k],tt[k])*h/(2*small_param*small_param)
        k = k + 1
    return x

# Вектор - реализация нявной схемы SRK (SADISRK2) 1.0 порядок сходимости.
# Stability analysis of explicit and implicit stochasticRunge-Kutta methods for stochastic differential equations
# Journal of Physics: Conference Series
# To cite this article: A Samsudin et al 2017 J. Phys.: Conf. Ser. 890 012084
def VectorSADISRK2(x0, t0, t1, h, Axt, Bxt):  # , DiffBxt):#, small_param):
    # DiffBxt, small_param не используются

    tt = np.arange(t0, t1 + h, h)
    len_ = len(tt)
    x = np.zeros(len_, dtype=float)
    x[0] = x0
    J1 = np.sqrt(h) * np.random.normal(0, 1, len_)
    c1 = 1 - np.sqrt(2) / 2
    c2 = np.sqrt(2) / 2

    def Phi_x1(x1, t, j1, xn):
        return x1 - (xn + c1 * h * Axt(x1, t) + c1 * j1 * Bxt(x1, t))

    def Phi_x2(x2, t, j1, xn):
        return x2 - (xn + h * (c2 * Axt(x1, t) + c1 * Axt(x2, t)) + j1 * (c2 * Bxt(x1, t) + c1 * Bxt(x2, t)))

    x1 = 0
    x2 = 0
    k = 0
    while k < len_ - 1:
        x1 = optimize.fsolve(Phi_x1, x0=x1, args=(tt[0], J1[k], x[k]))
        x2 = optimize.fsolve(Phi_x2, x0=x2, args=(tt[1], J1[k], x[k]))
        x[k + 1] = x[k] + h * (c2 * Axt(x1, tt[k]) + c1 * Axt(x2, tt[k + 1])) + J1[k] * (
                    c2 * Bxt(x1, tt[k]) + c1 * Bxt(x2, tt[k + 1]))
        k = k + 1
    return x
