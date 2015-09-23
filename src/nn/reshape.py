from stage import *

class Reshape(Stage):
    def __init__(self, reshapeFn, inputNames=None, outputDim=0, name=None, outputdEdX=True):
        Stage.__init__(self, name=name, inputNames=inputNames, outputDim=outputDim, outputdEdX=outputdEdX)
        self.reshapeFn = eval('lambda x: ' + reshapeFn)
        self.Xshape = 0

    def forward(self, X):
        self.Xshape = X.shape
        return np.reshape(X, self.reshapeFn(X.shape))

    def backward(self, dEdY):
        if self.outputdEdX:
            return np.reshape(dEdY, self.Xshape)

class TimeUnfold(Reshape):
    def __init__(self, inputNames=None, name=None, outputdEdX=True):
        Reshape.__init__(self,
                         name=name,
                         inputNames=inputNames,
                         reshapeFn='(x[0] * x[1], x[2])',
                         outputdEdX=outputdEdX)

class TimeFold(Reshape):
    def __init__(self, timespan, inputNames=None, name=None, outputdEdX=True):
        self.timespan = timespan
        t = str(self.timespan)
        Reshape.__init__(self,
                         name=name,
                         inputNames=inputNames,
                         reshapeFn='(x[0] / '+t+','+t+', x[1])',
                         outputdEdX=outputdEdX)

class TimeReverse(Stage):
    def __init__(self, inputNames, outputDim=0, name=None, outputdEdX=True):
        Stage.__init__(self, 
                       name=name,
                       inputNames=inputNames,
                       outputDim=outputDim,
                       outputdEdX=outputdEdX)

    def forward(self, X):
        #print self.name, X.shape
        N = X.shape[0]
        self.Xend = np.zeros(N, dtype=int) + X.shape[1]
        reachedEnd = np.sum(X, axis=-1) == 0.0
        Y = np.zeros(X.shape)
        # Scan for the end of the sequence.
        for n in range(N):
            found = False
            for t in range(X.shape[1]):
                if reachedEnd[n, t]:
                    self.Xend[n] = t
                    if t > 0:
                        found = True
                        Y[n, 0:t, :] = X[n, t-1::-1, :]
                    break
            if found == False:
                self.Xend[n] = X.shape[1]
                Y[n, :, :] = X[n, ::-1, :]
        return Y

    def backward(self, dEdY):
        if self.outputdEdX:
            dEdX = np.zeros(dEdY.shape)
            for n in range(dEdY.shape[0]):
                t = self.Xend[n]
                if t > 0:
                    dEdX[n, 0:t, :] = dEdY[n, t-1::-1, :]
            return dEdX
        else:
            return None

class TimeRepeat(Stage):
    def __init__(self, numRepeats, inputNames=None, outputDim=0, name=None, outputdEdX=True):
        Stage.__init__(self, name=name, inputNames=inputNames, outputDim=outputDim, outputdEdX=outputdEdX)
        self.numRepeats = numRepeats

    def forward(self, X):
        self.Xshape = X.shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return np.tile(X, (1, self.numRepeats, 1))

    def backward(self, dEdY):
        if self.outputdEdX:
            dEdY = dEdY.reshape(
                dEdY.shape[0], self.numRepeats, dEdY.shape[1] / self.numRepeats, dEdY.shape[2])
            dEdX = np.sum(dEdY, axis=1)
            if len(self.Xshape) == 2:
                dEdX = dEdX.reshape(dEdX.shape[0], dEdX.shape[-1])
            return dEdX

class TimeFinal(Stage):
    """
    Scans and selects the last timestep.
    """
    def __init__(self, inputNames, outputDim=0, name=None, outputdEdX=True):
        Stage.__init__(self, 
                       name=name, 
                       inputNames=inputNames, 
                       outputDim=outputDim, 
                       outputdEdX=outputdEdX)
        self.Xend = 0.0

    def forward(self, X):
        N = X.shape[0]
        self.X = X
        self.Xend = np.zeros(N, dtype=int) + X.shape[1]
        reachedEnd = np.sum(X, axis=-1) == 0.0
        Y = np.zeros((N, X.shape[-1]))
        # Scan for the end of the sequence.
        for n in range(N):
            for t in range(X.shape[1]):
                if reachedEnd[n, t]:
                    self.Xend[n] = t
                    break
        for n in range(N):
            if self.Xend[n] > 0:
                Y[n] = X[n, self.Xend[n] - 1]
        return Y

    def backward(self, dEdY):
        if self.outputdEdX:
            dEdX = np.zeros(self.X.shape)
            for n in range(dEdY.shape[0]):
                if self.Xend[n] > 0:
                    dEdX[n, self.Xend[n] - 1, :] = dEdY[n]
            return dEdX
        else:
            return None

class Concat(Stage):
    def __init__(self, inputNames, axis, name=None):
        Stage.__init__(self, name=name, inputNames=inputNames, outputDim=0)
        self.axis = axis
    def getInput(self):
        if len(self.inputs) > 1:
            self.splX = []
            for stage in self.inputs:
                X = stage.Y
                self.splX.append(X)
            return np.concatenate(self.splX, axis=self.axis)
        else:
            return self.inputs[0].Y
    def sendError(self, dEdX):
        """
        Iterates over input list and sends dEdX.
        """
        if len(self.inputs) > 1:
            s = 0
            for stage in self.inputs:
                s2 = s + stage.Y.shape[self.axis]
                if self.axis == 0:
                    stage.dEdY += dEdX[s : s2]
                elif self.axis == 1:
                    stage.dEdY += dEdX[:, s : s2]
                elif self.axis == 2:
                    stage.dEdY += dEdX[:, :, s : s2]
                s = s2
                stage.receivedError = True
        else:
            self.inputs[0].dEdY += dEdX
            self.inputs[0].receivedError = True

    def forward(self, X):
        return X
    def backward(self, dEdY):
        return dEdY
