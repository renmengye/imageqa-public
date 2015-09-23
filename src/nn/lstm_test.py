from sequential import *
from lstm_old import *
from dropout import *
from reshape import *
from lut import *
from lstm import *
import unittest

class LSTM_Recurrent_Real_Tests(unittest.TestCase):
    def test_all(self):
        trainInput = np.loadtxt('lstm_test_input.csv', delimiter=',')
        trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
        trainTarget = np.loadtxt('lstm_test_target.csv', delimiter=',')
        trainTarget = trainTarget.reshape(trainTarget.shape[0], 1)
        wordEmbed = np.loadtxt('lstm_test_word.csv', delimiter=',')
        D = 300
        D2 = 10
        N = trainInput.shape[0]
        Time = trainInput.shape[1]
        multiOutput = False
        time_unfold = TimeUnfold()
        lut = LUT(
            inputDim=np.max(trainInput)+1,
            outputDim=D,
            inputNames=None,
            needInit=False,
            initWeights=wordEmbed
        )

        time_fold = TimeFold(
            timespan=Time,
            inputNames=None
        )

        dropout = Dropout(
            name='d1',
            dropoutRate=0.2,
            initSeed=2,
            inputNames=None,
            outputDim=D2
        )
        dropout2 = Dropout(
            name='d2',
            dropoutRate=0.2,
            initSeed=2,
            inputNames=None,
            outputDim=D2
        )
        lstm = LSTM(
                name='lstm',
                timespan=Time,
                inputDim=D,
                outputDim=D2,
                inputNames=None,
                multiOutput=multiOutput,
                cutOffZeroEnd=True,
                learningRate=0.8,
                momentum=0.9,
                outputdEdX=True)

        lstm2 = LSTM_Old(
            name='lstm',
            inputDim=D,
            outputDim=D2,
            needInit=False,
            initRange=0.1,
            initSeed=0,
            cutOffZeroEnd=True,
            multiErr=multiOutput,
            learningRate=0.8,
            momentum=0.9,
            outputdEdX=True)

        sig = Map(
            name='sig',
            outputDim=1,
            activeFn=SigmoidActiveFn(),
            initRange=0.1,
            initSeed=5,
            learningRate=0.01,
            momentum=0.9,
            weightClip=10.0,
            gradientClip=0.1,
            weightRegConst=0.00005
        )
        sig2 = Map(
            name='sig',
            outputDim=1,
            activeFn=SigmoidActiveFn(),
            initRange=0.1,
            initSeed=5,
            learningRate=0.01,
            momentum=0.9,
            weightClip=10.0,
            gradientClip=0.1,
            weightRegConst=0.00005
        )

        costFn = crossEntOne
        model1 = Sequential(
            stages=[
                time_unfold,
                lut,
                time_fold,
                dropout,
                lstm,
                sig
            ]
        )

        model2 = Sequential(
            stages=[
                time_unfold,
                lut,
                time_fold,
                dropout2,
                lstm2,
                sig2
            ]
        )

        input_ = trainInput[0:N, 0:Time]
        target_ = trainTarget[0:N]
        Y1 = model1.forward(input_)

        W = lstm.getWeights()
        lstm2.W = W.transpose()
        Y2 = model2.forward(input_)
        self.chkEqual(Y1, Y2)

        E, dEdY1 = costFn(Y1, target_)
        E, dEdY2 = costFn(Y2, target_)
        model1.backward(dEdY1)
        model2.backward(dEdY2)

        dEdW = lstm.getGradient()
        self.chkEqual(dEdW.transpose(), lstm2.dEdW)
        lstm.updateWeights()
        lstm2.updateWeights()
        W = lstm.getWeights()
        self.chkEqual(W.transpose(), lstm2.W)

    def chkEqual(self, a, b):
        tolerance = 1e-1
        a = a.reshape(a.size)
        b = b.reshape(b.size)
        for i in range(a.size):
           if not ((a[i] == 0 and b[i] == 0) or
                (np.abs(a[i]) < 1e-7 and np.abs(b[i]) < 1e-7) or
                (np.abs(a[i] / b[i] - 1) < tolerance)):
                    print a[i], b[i], a[i]/b[i]
           self.assertTrue(
                (a[i] == 0 and b[i] == 0) or
                (np.abs(a[i]) < 1e-7 and np.abs(b[i]) < 1e-7) or
                (np.abs(a[i] / b[i] - 1) < tolerance))

if __name__ == '__main__':
    unittest.main()