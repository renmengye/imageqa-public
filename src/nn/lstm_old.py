from stage import *
from func import *

def sliceWeights(inputDim,outputDim,W):
    s1 = inputDim + outputDim * 2 + 1
    s2 = s1 * 2
    s3 = s2 + inputDim + outputDim + 1
    s4 = s3 + s1
    Wi = W[:, 0 : s1]
    Wf = W[:, s1 : s2]
    Wc = W[:, s2 : s3]
    Wo = W[:, s3 : s4]

    return Wi, Wf, Wc, Wo

def sliceWeightsSmall(inputDim,outputDim,W):
    Wi, Wf, Wc, Wo = sliceWeights(inputDim, outputDim, W)

    Wxi = Wi[:, 0 : inputDim]
    Wyi = Wi[:, inputDim : inputDim + outputDim]
    Wci = Wi[:, inputDim + outputDim : inputDim + outputDim + outputDim]
    Wxf = Wf[:, 0 : inputDim]
    Wyf = Wf[:, inputDim : inputDim + outputDim]
    Wcf = Wf[:, inputDim + outputDim : inputDim + outputDim + outputDim]
    Wxc = Wc[:, 0 : inputDim]
    Wyc = Wc[:, inputDim : inputDim + outputDim]
    Wxo = Wo[:, 0 : inputDim]
    Wyo = Wo[:, inputDim : inputDim + outputDim]
    Wco = Wo[:, inputDim + outputDim : inputDim + outputDim + outputDim]

    return Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco

def forwardPassN(X,cutOffZeroEnd,W):
    numEx = X.shape[0]
    timespan = X.shape[1]
    inputDim = X.shape[2]
    outputDim = W.shape[0]
    Wi, Wf, Wc, Wo = sliceWeights(inputDim, outputDim, W)
    Xend = np.zeros(numEx)
    Gi = np.zeros((numEx,timespan,outputDim))
    Gf = np.zeros((numEx,timespan,outputDim))
    Go = np.zeros((numEx,timespan,outputDim))
    Z = np.zeros((numEx,timespan,outputDim))
    C = np.zeros((numEx,timespan,outputDim))
    myShape = (numEx,timespan,outputDim)
    if cutOffZeroEnd:
        Y = np.zeros((numEx, timespan + 1, outputDim),)
        reachedEnd = np.sum(X, axis=-1) == 0.0
    else:
        Y = np.zeros(myShape,)
        reachedEnd = np.zeros((numEx, timespan))

    for n in range(0, numEx):
        Y[n], C[n], Z[n], \
        Gi[n], Gf[n], Go[n], \
        Xend[n] = \
            forwardPassOne(
                X[n], reachedEnd[n], cutOffZeroEnd, Wi, Wf, Wc, Wo)

    return Y, C, Z, Gi, Gf, Go, Xend

def forwardPassOne(X,reachedEnd,cutOffZeroEnd,
                   Wi,Wf,Wc,Wo):
    timespan = X.shape[0]
    outputDim = Wi.shape[0]
    # Last time step is reserved for final output of the entire input.
    if cutOffZeroEnd:
        Y = np.zeros((timespan + 1, outputDim))
    else:
        Y = np.zeros((timespan, outputDim))
    C = np.zeros((timespan, outputDim))
    Z = np.zeros((timespan, outputDim))
    Gi = np.zeros((timespan, outputDim))
    Gf = np.zeros((timespan, outputDim))
    Go = np.zeros((timespan, outputDim))
    Xend = timespan
    for t in range(0, timespan):
        if cutOffZeroEnd and reachedEnd[t]:
            Xend = t
            Y[-1, :] = Y[t - 1, :]
            break

        states1 = np.concatenate((X[t, :], \
                                  Y[t-1, :], \
                                  C[t-1, :], \
                                  np.ones(1)))
        states2 = np.concatenate((X[t, :], \
                                  Y[t-1, :], \
                                  np.ones(1)))
        Gi[t, :] = sigmoidFn(np.dot(Wi, states1))
        Gf[t, :] = sigmoidFn(np.dot(Wf, states1))
        Z[t, :] = np.tanh(np.dot(Wc, states2))
        C[t, :] = Gf[t, :] * C[t-1, :] + Gi[t, :] * Z[t, :]
        states3 = np.concatenate((X[t, :], \
                                  Y[t-1, :], \
                                  C[t, :], \
                                  np.ones(1)))
        Go[t, :] = sigmoidFn(np.dot(Wo, states3))
        Y[t, :] = Go[t, :] * np.tanh(C[t, :])
    if cutOffZeroEnd and not reachedEnd[-1]:
        Y[-1] = Y[-2]
        Xend = timespan
    return Y, C, Z, Gi, Gf, Go, Xend

def backPropagateN(dEdY,X,Y,C,Z,Gi,Gf,Go,Xend,
                   cutOffZeroEnd,multiErr,
                   outputdEdX,
                   W):
    numEx = X.shape[0]
    inputDim = X.shape[2]
    outputDim = Y.shape[2]
    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,Wyc,Wxo,Wyo,Wco = \
        sliceWeightsSmall(inputDim, outputDim, W)
    dEdW = np.zeros((W.shape[0], W.shape[1]))
    dEdX = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for n in range(0, numEx):
        dEdWtmp, dEdX[n] = \
            backPropagateOne(dEdY[n],X[n],Y[n],
                        C[n],Z[n],Gi[n],
                        Gf[n],Go[n],
                        Xend[n],cutOffZeroEnd,
                        multiErr,outputdEdX,
                        Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
                        Wyc,Wxo,Wyo,Wco,
                        (W.shape[0], W.shape[1]))
        dEdW += dEdWtmp
    return dEdW, dEdX

def backPropagateOne(dEdY,X,Y,C,Z,
                    Gi,Gf,Go,Xend,
                    cutOffZeroEnd,multiErr,
                    outputdEdX,
                    Wxi,Wyi,Wci,
                    Wxf,Wyf,Wcf,
                    Wxc,Wyc,Wxo,Wyo,
                    Wco,Wshape):
    Xend = int(Xend)
    if cutOffZeroEnd and multiErr:
        dEdY[Xend - 1] += dEdY[-1]
    inputDim = X.shape[1]
    outputDim = Y.shape[1]
    dEdW = np.zeros(Wshape)
    if Xend == 0: return dEdW, np.zeros(X.shape)
    dEdWi,dEdWf,dEdWc,dEdWo = sliceWeights(inputDim, outputDim, dEdW)
    ddim = (outputDim, Xend)

    # (j, t)
    dEdGi = np.zeros(ddim)
    dEdGf = np.zeros(ddim)
    dEdZ = np.zeros(ddim)
    dEdGo = np.zeros(ddim)

    dEdX = np.zeros(X.shape)

    # (k -> t)
    one = np.ones((Xend, 1))
    Yt1 = np.concatenate((np.zeros((1, outputDim)), Y[:Xend-1]))
    Ct1 = np.concatenate((np.zeros((1, outputDim)), C[:Xend-1]))
    states1T = np.concatenate((X[:Xend], Yt1, Ct1, one), axis=-1)
    states2T = np.concatenate((X[:Xend], Yt1, one), axis=-1)
    states3T = np.concatenate((X[:Xend], Yt1, C[:Xend], one), axis=-1)
    U = np.tanh(C[:Xend])
    dU = 1 - U[:Xend] * U[:Xend]
    dZ = 1 - Z[:Xend] * Z[:Xend]
    dGi = Gi[:Xend] * (1 - Gi[:Xend])
    dGf = Gf[:Xend] * (1 - Gf[:Xend])
    dGo = Go[:Xend] * (1 - Go[:Xend])

    # (j, t)
    dCdGi = (Z[:Xend] * dGi).transpose()
    dCdGf = (Ct1 * dGf).transpose()
    dCdZ = (Gi[:Xend] * dZ).transpose()
    dYdGo = (U[:Xend] * dGo).transpose()

    for t in reversed(range(0, Xend)):
        dEdYnow = dEdY[t] if multiErr else 0
        dYdC = np.diag(Go[t] * dU[t]) + \
               dYdGo[:,t:t+1] * Wco
        if t < Xend - 1:
            dEdYt = np.dot(dEdYt, dYdY) + np.dot(dEdCt, dCdY) + dEdYnow
            dEdCt = np.dot(dEdCt, dCdC) + np.dot(dEdYt, dYdC)
        else:
            dEdYt = dEdYnow if multiErr else dEdY
            dEdCt = np.dot(dEdYt, dYdC)
        dEdGi[:, t] = dEdCt * dCdGi[:,t]
        dEdGf[:, t] = dEdCt * dCdGf[:,t]
        dEdZ[:, t] = dEdCt * dCdZ[:,t]
        dEdGo[:, t] = dEdYt * dYdGo[:,t]
        dCdC = dCdGf[:,t:t+1] * Wcf + \
               np.diag(Gf[t]) + \
               dCdGi[:,t:t+1] * Wci
        dCdY = dCdGf[:,t:t+1] * Wyf + \
               dCdZ[:,t:t+1] * Wyc + \
               dCdGi[:,t:t+1] * Wyi
        dYdY = dYdGo[:,t:t+1] * Wyo

    dEdWi += np.dot(dEdGi, states1T)
    dEdWf += np.dot(dEdGf, states1T)
    dEdWc += np.dot(dEdZ, states2T)
    dEdWo += np.dot(dEdGo, states3T)

    if outputdEdX:
        dEdX[0:Xend] = np.dot(dEdGi.transpose(), Wxi) + \
                       np.dot(dEdGf.transpose(), Wxf) + \
                       np.dot(dEdZ.transpose(), Wxc) + \
                       np.dot(dEdGo.transpose(), Wxo)

    return dEdW, dEdX

class LSTM_Old(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 inputNames=None,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 cutOffZeroEnd=False,
                 multiErr=False,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.cutOffZeroEnd = cutOffZeroEnd
        self.multiErr = multiErr
        self.random = np.random.RandomState(initSeed)

        if needInit:
            start = -initRange / 2.0
            end = initRange / 2.0
            Wxi = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxf = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxc = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxo = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wyi = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyf = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyc = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyo = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wci = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wcf = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wco = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wbi = np.ones((self.outputDim, 1))
            Wbf = np.ones((self.outputDim, 1))
            Wbc = np.zeros((self.outputDim, 1))
            Wbo = np.ones((self.outputDim, 1))

            Wi = np.concatenate((Wxi, Wyi, Wci, Wbi), axis=1)
            Wf = np.concatenate((Wxf, Wyf, Wcf, Wbf), axis=1)
            Wc = np.concatenate((Wxc, Wyc, Wbc), axis=1)
            Wo = np.concatenate((Wxo, Wyo, Wco, Wbo), axis=1)
            self.W = np.concatenate((Wi, Wf, Wc, Wo), axis = 1)
        else:
            self.W = initWeights
        self.X = 0
        self.Xend = 0
        self.Y = 0
        self.C = 0
        self.Z = 0
        self.Gi = 0
        self.Gf = 0
        self.Go = 0
        pass

    def forward(self, X):
        Y, C, Z, Gi, Gf, Go, Xend = \
            forwardPassN(
            X, self.cutOffZeroEnd, self.W)

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        self.Xend = Xend

        return Y if self.multiErr else Y[:,-1]

    def backward(self, dEdY):
        self.dEdW, dEdX = backPropagateN(dEdY,self.X,self.Y,
                                self.C,self.Z,self.Gi,
                                self.Gf,self.Go,
                                self.Xend,self.cutOffZeroEnd,
                                self.multiErr,self.outputdEdX,
                                self.W)
        self.dEdX = dEdX
        return dEdX if self.outputdEdX else None
