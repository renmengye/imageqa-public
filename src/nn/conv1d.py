import os
use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
if use_gpu:
    import gnumpy as gpu
    import gnumpy as gnp
from stage import *

class Conv1D(Stage):
    """
    1D temporal convolution.
    No padding, stride=1.
    """
    def __init__(self,
                 numChannels,
                 windowSize,
                 numFilters,
                 inputNames=None,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=None,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 defaultValue=0.0,
                 outputdEdX=True,
                 gpu=use_gpu,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=numFilters,
                 defaultValue=defaultValue,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 gpu=gpu,
                 outputdEdX=outputdEdX)
        self.numFilters = numFilters
        self.numChannels = numChannels
        self.windowSize = windowSize
        self.random = np.random.RandomState(initSeed)
        if needInit:
            self.W = self.random.uniform(-initRange/2.0, initRange/2.0,
                        (self.windowSize * self.numChannels, self.numFilters))
        else:
            self.W = initWeights
        if self.gpu:
            self.W = gnp.as_garray(self.W.astype('float32'))
        self.X = 0
        self.Y = 0

    def forward(self, X):
        self.X = X
        # Num of examples
        N = X.shape[0]
        # Timespan
        T = X.shape[1]
        # Windows size
        S = self.windowSize
        # Channels
        D = self.numChannels
        # Num filters
        F = self.numFilters
        Z = np.zeros((N, T - S + 1, S, D), X.dtype)
        for i in range(T - S + 1):
            Z[:, i, :, :] = X[:, i : i + S, :]
        Z = Z.reshape(N * (T - S + 1), S * D)
        if self.gpu:
            Z = gpu.as_garray(Z.astype('float32'))
            Y = gpu.dot(Z, self.W)
            Y = gpu.as_numpy_array(Y)
        else:
            Y = np.dot(Z, self.W)

        Y = Y.reshape(N, T - S + 1, F)
        self.Z = Z
        return Y

    def backward(self, dEdY):
        N = dEdY.shape[0]
        S = self.windowSize
        T = dEdY.shape[1] + S - 1
        F = dEdY.shape[2]
        D = self.X.shape[2]
        dEdY = dEdY.reshape(N * (T - S + 1), F)
        dEdX = np.zeros(self.X.shape, self.X.dtype)
        
        if self.gpu:
            gdEdY = gpu.as_garray(dEdY.astype('float32'))
            self.dEdW = gpu.dot(self.Z.transpose(), gdEdY)
        else:
            self.dEdW = np.dot(self.Z.transpose(), dEdY)

        if self.outputdEdX:
            if self.gpu:
                gdEdZ = gpu.dot(gdEdY, self.W.transpose())
                dEdZ = gpu.as_numpy_array(gdEdZ)
            else:
                dEdZ = np.dot(dEdY, self.W.transpose())

            dEdZ = dEdZ.reshape(N, T - S + 1, S, D)
            for t in range(0, T):
                if t <= S - 1:
                    dEdX[:, t, :] = np.sum(dEdZ[:, range(0, t + 1), range(t, -1, -1), :], axis=1)
                elif t >= T - S + 1:
                    dEdX[:, t, :] = np.sum(dEdZ[:, range(t - S + 1, T - S + 1), range(S - 1, S - (T - t) - 1, -1), :], axis=1)
                else:
                    dEdX[:, t, :] = np.sum(dEdZ[:, range(t - S + 1, t + 1), range(S - 1, -1, -1), :], axis=1)
        return dEdX
