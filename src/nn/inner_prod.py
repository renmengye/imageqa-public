from stage import *

class InnerProduct(Stage):
    """
    Inner product calculates the inner product of two input vectors.
    Two vectors aligns on the second axis (time-axis).
    """
    def __init__(self,
                name,
                inputNames,
                outputDim,
                learningRate=0.0,
                learningRateAnnealConst=0.0,
                momentum=0.0,
                deltaMomentum=0.0,
                weightClip=0.0,
                gradientClip=0.0,
                weightRegConst=0.0,
                outputdEdX=True):
        Stage.__init__(self,
                 name=name,
                 outputDim=outputDim,
                 inputNames=inputNames,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.W = 1
    def forward(self, X):
        Y = np.sum(X[:, 0, :] * X[:, 1, :], axis=-1) + self.W
        self.Y = Y
        self.X = X
        return Y

    def backward(self, dEdY):
        self.dEdW = np.sum(dEdY,axis=0)
        #print dEdY
        dEdX = np.zeros(self.X.shape)
        dEdX[:, 1, :] = dEdY.reshape(dEdY.size, 1) * self.X[:, 0, :]
        dEdX[:, 0, :] = dEdY.reshape(dEdY.size, 1) * self.X[:, 1, :]
        return dEdX