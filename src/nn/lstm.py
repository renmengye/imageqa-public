from recurrent import *
from elem_prod import *
from sum import *
from active import *

class LSTM(RecurrentContainer):
    def __init__(self,
                 inputDim,
                 outputDim,
                 timespan,
                 inputNames,
                 defaultValue=0.0,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 multiInput=True,
                 multiOutput=False,
                 cutOffZeroEnd=True,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 name=None):
        D2 = outputDim
        multiOutput = multiOutput
        if name is None: print 'Warning: name is None.'
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.I = RecurrentAdapter(Map(
                 name=name + '.I',
                 inputNames=['input(0)', name + '.H(-1)', name + '.C(-1)'],
                 outputDim=D2,
                 activeFn=SigmoidActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))

        self.F = RecurrentAdapter(Map(
                 name=name + '.F',
                 inputNames=['input(0)', name + '.H(-1)', name + '.C(-1)'],
                 outputDim=D2,
                 activeFn=SigmoidActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed+1,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))

        self.Z = RecurrentAdapter(Map(
                 name=name + '.Z',
                 inputNames=['input(0)', name + '.H(-1)'],
                 outputDim=D2,
                 activeFn=TanhActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed+2,
                 biasInitConst=0.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))

        self.O = RecurrentAdapter(Map(
                 name=name + '.O',
                 inputNames=['input(0)', name + '.H(-1)', name + '.C(0)'],
                 outputDim=D2,
                 activeFn=SigmoidActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed+3,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))
        
        if not needInit:
            self.I.W, self.F.W, self.Z.W, self.O.W = self.splitWeights(initWeights)

        self.FC = RecurrentAdapter(ElementProduct(
                  name=name + '.F*C',
                  inputNames=[name + '.F', name + '.C(-1)'],
                  outputDim=D2))

        self.IZ = RecurrentAdapter(ElementProduct(
                  name=name + '.I*Z',
                  inputNames=[name + '.I', name + '.Z'],
                  outputDim=D2))

        self.C = RecurrentAdapter(Sum(
                 name=name + '.C',
                 inputNames=[name + '.F*C', name + '.I*Z'],
                 numComponents=2,
                 outputDim=D2))

        self.U = RecurrentAdapter(Active(
                 name=name + '.U',
                 inputNames=[name + '.C'],
                 outputDim=D2,
                 activeFn=TanhActiveFn()))

        self.H = RecurrentAdapter(ElementProduct(
                 name=name + '.H',
                 inputNames=[name + '.O', name + '.U'],
                 outputDim=D2,
                 defaultValue=defaultValue))

        stages = [self.I, self.F, self.Z, self.FC, self.IZ, self.C, self.O, self.U, self.H]
        RecurrentContainer.__init__(self,
                           stages=stages,
                           timespan=timespan,
                           inputNames=inputNames,
                           outputStageNames=[name + '.H'],
                           inputDim=inputDim,
                           outputDim=outputDim,
                           multiInput=multiInput,
                           multiOutput=multiOutput,
                           cutOffZeroEnd=cutOffZeroEnd,
                           name=name,
                           outputdEdX=outputdEdX)

    def getWeights(self):
        if self.I.stages[0].gpu:
            return np.concatenate((
                    gpu.as_numpy_array(self.I.getWeights()),
                    gpu.as_numpy_array(self.F.getWeights()),
                    gpu.as_numpy_array(self.Z.getWeights()),
                    gpu.as_numpy_array(self.O.getWeights())), axis=0)
        else:
            return np.concatenate((self.I.getWeights(),
                               self.F.getWeights(),
                               self.Z.getWeights(),
                               self.O.getWeights()), axis=0)
            
    def getGradient(self):
        if self.I.stages[0].gpu:
            return np.concatenate((
                gpu.as_numpy_array(self.I.getGradient()),
                gpu.as_numpy_array(self.F.getGradient()),
                gpu.as_numpy_array(self.Z.getGradient()),
                gpu.as_numpy_array(self.O.getGradient())), axis=0)
        else:
            return np.concatenate((self.I.getGradient(),
                                   self.F.getGradient(),
                                   self.Z.getGradient(),
                                   self.O.getGradient()), axis=0)

    def splitWeights(self, W):
        D = self.inputDim
        D2 = self.outputDim
        s = D + D2 + D2 + 1
        s2 = D + D2 + 1
        IW = W[:s, :]
        FW = W[s:s + s, :]
        ZW = W[s + s:s + s + s2, :]
        OW = W[s + s +s2:s + s + s2 + s, :]
        return IW, FW, ZW, OW

    def loadWeights(self, W):
        IW, FW, ZW, OW = self.splitWeights(W)
        if self.I.stages[0].gpu:
            self.I.loadWeights(gpu.as_garray(IW))
            self.F.loadWeights(gpu.as_garray(FW))
            self.Z.loadWeights(gpu.as_garray(ZW))
            self.O.loadWeights(gpu.as_garray(OW))
        else:
            self.I.loadWeights(IW)
            self.F.loadWeights(FW)
            self.Z.loadWeights(ZW)
            self.O.loadWeights(OW)
