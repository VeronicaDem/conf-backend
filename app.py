from flask import Flask,request
from sympy.utilities.autowrap import ufuncify
import numpy as np
import json
import statistics as stat
from sympy import *
import concurrent.futures
import grant_sde as gs
import flask_cors
from multiprocessing.pool import ThreadPool
app = Flask(__name__)
flask_cors.CORS(app)
sigma = 0.1
small_param = 0.1
x0 = -0.5
t0 = -0.5
t1 = 0.5
h = 0.01
YlimFrom = -5
YlimTo = 5
XlimFrom = t0
XlimTo = t1
x0_range = np.arange(-2,4,1)

# def Axt(x,t):
#     return -(x*x) - t
#
# def Bxt(x,t):
#     return sigma
#
# def DiffBxt(x,t):
#     return 0

def SDEPlot(SDEFunc, x0, t0, t1, h, small_param, YlimFrom, YlimTo, XlimFrom, XlimTo, x0_range, Axt, Bxt, DiffBxt, signature, mu, sigma, count, a, T, b):
    print("x(0) = {0}, t=[{1} : {2}], h={3}, small_param={4}".format(x0_range, t0, t1, h, small_param))
    x_s = []
   # plt.figure()
    print(type(Axt))
    # версии СДУ Риккати
    tt = np.arange(t0, t1 + h, h)
    for i in range(5):
        if signature == 1:
             x_s.append(SDEFunc(x0, t0, t1, h, Axt, Bxt, DiffBxt, small_param).tolist())
        elif signature == 2:
            x_s.append(SDEFunc(x0, t0, t1, h, Axt, Bxt, DiffBxt).tolist())
        elif signature == 3:
            x_s.append(SDEFunc(x0, t0, t1, h, mu, sigma, small_param).tolist())
        elif signature == 4:
            x_s.append(SDEFunc(x0, t0, t1, h, mu, sigma).tolist())
        elif signature == 5:
            x_s.append(SDEFunc(x0, t0, t1, h, mu, sigma, small_param, count).tolist())
        elif signature == 6:
            x_s.append(SDEFunc(x0, t0, t1, h, mu, sigma,count).tolist())
        elif signature == 7:
            x_s.append(SDEFunc(x0, t0, t1, h, Axt, Bxt, count, small_param).tolist())
        elif signature == 8:
            x_s.append(SDEFunc(x0, t0, t1, h, Axt, Bxt, count).tolist())
        elif signature == 9:
            x_s.append(SDEFunc(x0, t0, t1, h, Axt, Bxt, small_param).tolist())
        elif signature == 10:
            x_s.append(SDEFunc(x0, t0, t1, h ,Axt, Bxt).tolist())
        elif signature == 11:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma, a, T, small_param, count).tolist())
        elif signature == 12:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma, a, T, count).tolist())
        elif signature == 13:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma, a, b, small_param, count).tolist())
        elif signature == 14:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma, a, b, count).tolist())
        elif signature == 15:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma, a, b, count).tolist())
        elif signature == 16:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma, small_param, count).tolist())
        elif signature == 17:
            x_s.append(SDEFunc(x0, t0, t1, h, sigma,count).tolist())
        elif signature == 18:
            x_s.append(SDEFunc(x0, t0, t1, h, Axt, Bxt).tolist())
        else:
            print("unknown method")
            return
        #plot(tt, x, linestyle='solid',
         #    linewidth=1, markersize=2, label='x', scalex=True, scaley=True)

    # unstable manifold - критическое многообразие
    tt_uns_man = np.arange(t0, 0, h)
    tt_uns_man = np.append(tt_uns_man, 0)
    lower = -np.sqrt(-tt_uns_man)
    upper = np.sqrt(-tt_uns_man)

    #plot(tt_uns_man, lower, color='red', linestyle='dashed',
     #    linewidth=3, markersize=1, label='lower', scalex=True, scaley=True)
   # plot(tt_uns_man, upper, color='blue', linestyle='dashed',
    #     linewidth=3, markersize=1, label='upper', scalex=True, scaley=True)

    #plt.title('SDE Риккати')
    #plt.grid(True)
    #plt.xlim((XlimFrom, XlimTo))
    #plt.ylim((YlimFrom, YlimTo))

    #N = 2
    #params = plt.gcf()
    #plSize = params.get_size_inches()
    #params.set_size_inches((plSize[0] * N, plSize[1] * N))
    #plt.show()

    plot = [{"y":x_s, "x": tt.tolist()}]#,{"x": tt_uns_man.tolist(), "y":lower.tolist()}, {"x": tt_uns_man.tolist(), "y": upper.tolist()}]
    plot = json.dumps(plot).replace("-Infinity", "\"negative-infinity\"").replace("Infinity","\'positive-infinity\'")
    print(plot)
    return plot


methods_dict = {
    "VectorRungeKutta4orderSmallParam":  gs.VectorRungeKutta4orderSmallParam,
    "VectorRungeKutta4order_": gs.VectorRungeKutta4order_,
    "MatrixEilerSchemeSmallParam" : gs.MatrixEilerSchemeSmallParam,
    "MatrixEilerScheme" : gs.MatrixEilerScheme,
    "VectorEilerSchemeSmallParam" : gs.VectorEilerSchemeSmallParam,
    "VectorEilerScheme" : gs.VectorEilerScheme,
    "MatrixExactBrownianBridgeSmallParam" : gs.MatrixExactBrownianBridgeSmallParam,
    "MatrixExactBrownianBridge" : gs.MatrixExactBrownianBridge,
    "MatrixExactOrnUlenSmallParam" : gs.MatrixExactOrnUlenSmallParam,
    "MatrixExactOrnUlen" : gs.MatrixExactOrnUlen,
    "MatrixExactLogMotionSmallParam" : gs.MatrixExactLogMotionSmallParam,
    "MatrixExactLogMotion" : gs.MatrixExactLogMotion,
    "RungeKutta4order_VineraMotionLineVolatilitySmallParam" : gs.RungeKutta4order_VineraMotionLineVolatilitySmallParam,
    "RungeKutta4order_VineraMotionLineVolatility" : gs.RungeKutta4order_VineraMotionLineVolatility,
    "MatrixRungeKutta4order_VineraMotionLineVolatilitySmallParam" : gs.MatrixRungeKutta4order_VineraMotionLineVolatilitySmallParam,
    "MatrixRungeKutta4order_VineraMotionLineVolatility" : gs.MatrixRungeKutta4order_VineraMotionLineVolatility,
    "MatrixVineraMotionDriftedLineVolatilitySmallParam" : gs.MatrixVineraMotionDriftedLineVolatilitySmallParam,
    "MatrixVineraMotionDriftedLineVolatility" : gs.MatrixVineraMotionDriftedLineVolatility,
    "MatrixVineraMotionDrifted" : gs.MatrixVineraMotionDrifted,
    "MatrixVineraMotionDriftedSmallParam" : gs.MatrixVineraMotionDriftedSmallParam,
    "MatrixAdditiveMotionSmallParam" : gs.MatrixAdditiveMotionSmallParam,
    "MatrixAdditiveMotion" : gs.MatrixAdditiveMotion,
    "VectorMeanAdditiveMotionSmallParam" : gs.VectorMeanAdditiveMotionSmallParam,
    "DMatrixAdditiveMotionSmallParam" : gs.DMatrixAdditiveMotionSmallParam,
    "MatrixVineraMotion" : gs.MatrixVineraMotion,
    "MatrixVineraMotionLinearVolatility" : gs.MatrixVineraMotionLinearVolatility,
    "MatrixVineraMotionLinearVolatilitySmallParam" : gs.MatrixVineraMotionLinearVolatilitySmallParam,
     'VectorSADISRK2' : gs.VectorSADISRK2
}
@app.route('/calc', methods=['GET'])
def prepare_data():
    Axt = request.args.get("Axt")
    Bxt = request.args.get("Bxt")
    small_param = float(request.args.get("small_param"))
    h = float(request.args.get("h"))
    x0 = float(request.args.get("x0"))
    t0 = float(request.args.get("t0"))
    t1 = float(request.args.get("t1"))
    mu = float(request.args.get("mu"))
    sigma = float(request.args.get("sigma"))
    count = int(request.args.get("count"))
    a = float(request.args.get("a"))
    T = float(request.args.get('T'))
    b = float(request.args.get("b"))
    signature = int(request.args.get('signature'))
    x = Symbol("x")
    t = Symbol("t")

    DiffBxt = diff(Bxt, x)
    fAxt = ufuncify([x,t], Axt)
    fBxt = ufuncify([x,t], Bxt)
    fDiffBxt = ufuncify([x,t], DiffBxt)

    method = request.args.get("method")
    matrix_flag = method.startswith("Matrix");
    return SDEPlot(methods_dict[method], x0, t0, t1, h, small_param, YlimFrom, YlimTo, XlimFrom, XlimTo,
            x0_range, fAxt, fBxt, fDiffBxt, signature, mu, sigma, count, a, T, b)


if __name__ == '__main__':
    app.run()
