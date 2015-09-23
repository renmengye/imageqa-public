from sequential import *
from lstm_old import *
from map import *
from dropout import *
from reshape import *
from lut import *
from active_func import *
import unittest

class Sequential_Tests(unittest.TestCase):
    """Sequential stacks of stages tests"""
    def setUp(self):
        random = np.random.RandomState(2)
        self.trainInput = random.uniform(0, 10, (5, 5, 1)).astype(int)
        self.trainTarget = random.uniform(0, 1, (5, 1)).astype(int)

    def test_grad(self):
        wordEmbed = np.random.rand(np.max(self.trainInput), 5)
        timespan = self.trainInput.shape[1]
        time_unfold = TimeUnfold()

        lut = LUT(
            inputDim=np.max(self.trainInput)+1,
            outputDim=5,
            inputNames=None,
            needInit=False,
            initWeights=wordEmbed
        )

        m = Map(
            outputDim=5,
            activeFn=IdentityActiveFn(),
            inputNames=None,
            initRange=0.1,
            initSeed=1,
        )

        time_fold = TimeFold(
            timespan=timespan
        )

        lstm = LSTM_Old(
            inputDim=5,
            outputDim=5,
            initRange=.1,
            initSeed=3,
            cutOffZeroEnd=True,
            multiErr=True
        )

        dropout = Dropout(
            name='d1',
            dropoutRate=0.5,
            inputNames=None,
            outputDim=5,
            initSeed=2,
            debug=True
        )

        lstm_second = LSTM_Old(
            inputDim=5,
            outputDim=5,
            initRange=.1,
            initSeed=3,
            cutOffZeroEnd=True,
            multiErr=False
        )

        soft = Map(
            outputDim=2,
            activeFn=SoftmaxActiveFn,
            initRange=0.1,
            initSeed=5
        )

        self.model = Sequential(
            stages=[
                time_unfold,
                lut,
                m,
                time_fold,
                lstm,
                dropout,
                lstm_second,
                soft
            ])
        self.hasDropout = True
        costFn = crossEntIdx
        output = self.model.forward(self.trainInput, dropout=self.hasDropout)
        E, dEdY = costFn(output, self.trainTarget)
        dEdX = self.model.backward(dEdY)
        self.chkgrd(soft.dEdW, self.evaluateGrad(soft.getWeights(), costFn))
        #self.chkgrd(lstm_second.dEdW, self.evaluateGrad(lstm_second.getWeights(), costFn))
        #self.chkgrd(lstm.dEdW, self.evaluateGrad(lstm.getWeights(), costFn))
        self.chkgrd(m.dEdW, self.evaluateGrad(m.getWeights(), costFn))

    def chkgrd(self, dE, dETmp):
        #print dE/dETmp
        dE = dE.reshape(dE.size)
        dETmp = dETmp.reshape(dE.size)
        tolerance = 5e-1
        for i in range(dE.size):
            self.assertTrue(
                (dE[i] == 0 and dETmp[i] == 0) or
                (np.abs(dE[i] / dETmp[i] - 1) < tolerance))

    def evaluateGrad(self, W, costFn):
        eps = 1
        dEdW = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i,j] += eps
                output = self.model.forward(self.trainInput, dropout=self.hasDropout)
                Etmp1, d1 = costFn(output, self.trainTarget)

                W[i,j] -= 2 * eps
                output = self.model.forward(self.trainInput, dropout=self.hasDropout)
                Etmp2, d2 = costFn(output, self.trainTarget)

                dEdW[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                W[i,j] += eps
        return dEdW

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Sequential_Tests))
    unittest.TextTestRunner(verbosity=2).run(suite)
