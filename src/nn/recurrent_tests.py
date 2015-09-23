from recurrent import *
from lstm_old import *
import stage_tests
import unittest
from map import *
from active import *
from elem_prod import *
from sum import *

class Recurrent_Tests(stage_tests.StageTests):
    def setUp(self):
        self.N = 5
        self.T = 5
        self.D = 10
        self.D2 = 5
        self.sigm_ = Map(
                        name='sigm',
                        inputNames=['input(0)', 'sigm(-1)', 'sigm(-2)'],
                        outputDim=self.D2,
                        activeFn=SigmoidActiveFn(),
                        initRange=1,
                        initSeed=5,
                        learningRate=0.9
                    )
        self.sigm = RecurrentAdapter(
            stage=self.sigm_)
        self.stage = self.sigm.getStage(time=0)
        self.model = RecurrentContainer(
            stages=[self.sigm],
            timespan=self.T,
            inputDim=self.D,
            outputDim=self.D2,
            outputStageNames=['sigm'],
            multiOutput=True,
            name='container',
            outputdEdX=True)

        self.testInputErr = True
        self.costFn = meanSqErr

    def test_grad(self):
        random = np.random.RandomState(1)
        X = random.rand(self.N, self.T, self.D)
        T = random.rand(self.N, self.T, self.D2)
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

    def test_forward(self):
        random = np.random.RandomState(1)
        X = random.rand(self.N, self.T, self.D)
        tolerance = 1e-4
        Y2 = self.realForward(X)
        Y = self.model.forward(X)
        Y2 = Y2.reshape(Y2.size)
        Y = Y.reshape(Y.size)
        for i in range(Y.size):
            self.assertTrue((Y[i] == 0 and Y2[i] == 0) or (np.abs(Y[i] / Y2[i] - 1) < tolerance))

    def realForward(self, X):
        Y2 = np.zeros((self.N, self.T, self.D2))
        for t in range(self.T):
            Y2[:, t, :] = self.sigm_.forward(
                np.concatenate((X[:, t, :], Y2[:, t-1, :], Y2[:, t-2, :]), axis=-1))
        return Y2

class LSTM_Recurrent_Random_Tests(unittest.TestCase):
    def test_singleErr(self):
        self.func(False)

    def test_multiErr(self):
        self.func(True)

    def func(self, multiOutput):
        N = 5
        D = 10
        D2 = 5
        Time = 5
        I = RecurrentAdapter(Map(
                name='I',
                inputNames=['input(0)', 'H(-1)', 'C(-1)'],
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=0.1,
                initSeed=5,
                biasInitConst=1.0,
                learningRate=0.8,
                momentum=0.9
            ))

        F = RecurrentAdapter(Map(
                name='F',
                inputNames=['input(0)', 'H(-1)', 'C(-1)'],
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=0.1,
                initSeed=6,
                biasInitConst=1.0,
                learningRate=0.8,
                momentum=0.9
            ))

        Z = RecurrentAdapter(Map(
                name='Z',
                inputNames=['input(0)', 'H(-1)'],
                outputDim=D2,
                activeFn=TanhActiveFn(),
                initRange=0.1,
                initSeed=7,
                biasInitConst=0.0,
                learningRate=0.8,
                momentum=0.9
            ))

        FC = RecurrentAdapter(ElementProduct(
                name='F.C',
                inputNames=['F(0)', 'C(-1)'],
                outputDim=D2
            ))

        IZ = RecurrentAdapter(ElementProduct(
                name='I.Z',
                inputNames=['I(0)', 'Z(0)'],
                outputDim=D2
            ))

        C = RecurrentAdapter(Sum(
                name='C',
                inputNames=['F.C(0)', 'I.Z(0)'],
                numComponents=2,
                outputDim=D2
            ))

        O = RecurrentAdapter(Map(
                name='O',
                inputNames=['input(0)', 'H(-1)', 'C(0)'],
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=0.1,
                initSeed=8,
                biasInitConst=1.0,
                learningRate=0.8,
                momentum=0.9
            ))

        U = RecurrentAdapter(Active(
                name='U',
                inputNames=['C(0)'],
                outputDim=D2,
                activeFn=TanhActiveFn()
            ))

        H = RecurrentAdapter(ElementProduct(
                name='H',
                inputNames=['O(0)', 'U(0)'],
                outputDim=D2
            ))

        lstm = RecurrentContainer(
                name='lstm',
                stages=[I, F, Z, FC, IZ, C, O, U, H],
                timespan=Time,
                inputDim=D,
                outputDim=D2,
                outputStageNames=['H'],
                multiOutput=multiOutput,
                outputdEdX=True)

        lstm2 = LSTM_Old(
            name='lstm2',
            inputDim=D,
            outputDim=D2,
            needInit=False,
            initRange=0.1,
            initSeed=0,
            cutOffZeroEnd=True,
            multiErr=multiOutput,
            learningRate=0.8,
            momentum=0.9
        )

        random = np.random.RandomState(1)
        costFn = crossEntOne
        for i in range(3):
            X = random.rand(N, Time, D) * 0.1
            if multiOutput:
                T = random.rand(N, Time, D2) * 0.1
            else:
                T = random.rand(N, D2) * 0.1

            Y = lstm.forward(X)
            E, dEdY = costFn(Y, T)
            dEdX = lstm.backward(dEdY)
            if i == 0:
                if I.stages[0].gpu:            
                    W = np.concatenate((
                        gpu.as_numpy_array(I.getWeights()),
                        gpu.as_numpy_array(F.getWeights()),
                        gpu.as_numpy_array(Z.getWeights()),
                        gpu.as_numpy_array(O.getWeights())), axis=0)
                else:
                    W = np.concatenate((I.getWeights(), 
                        F.getWeights(), Z.getWeights(), 
                        O.getWeights()), axis=0)
                lstm2.W = W.transpose()
            if multiOutput:
                Y2 = lstm2.forward(X)[:,:-1]
            else:
                Y2 = lstm2.forward(X)

            E, dEdY2 = costFn(Y2, T)
            if multiOutput:
                dEdX2 = lstm2.backward(np.concatenate((dEdY2, np.zeros((N, 1, D2))), axis=1))
            else:
                dEdX2 = lstm2.backward(dEdY2)

            if I.stages[0].gpu:
                dEdW = np.concatenate((
                    gpu.as_numpy_array(I.getGradient()),
                    gpu.as_numpy_array(F.getGradient()),
                    gpu.as_numpy_array(Z.getGradient()),
                    gpu.as_numpy_array(O.getGradient())), axis=0)
            else:
                dEdW = np.concatenate((I.getGradient(),
                                       F.getGradient(),
                                       Z.getGradient(),
                                       O.getGradient()), axis=0)
            dEdW2 = lstm2.dEdW
            lstm.updateWeights()
            lstm2.updateWeights()
            #self.chkEqual(Y, Y2)

            #self.chkEqual(dEdX, dEdX2)
            self.chkEqual(dEdW.transpose(), dEdW2)
            if I.stages[0].gpu:            
                W = np.concatenate((
                    gpu.as_numpy_array(I.getWeights()),
                    gpu.as_numpy_array(F.getWeights()),
                    gpu.as_numpy_array(Z.getWeights()),
                    gpu.as_numpy_array(O.getWeights())), axis=0)
            else:
                W = np.concatenate((I.getWeights(), 
                    F.getWeights(), Z.getWeights(), 
                    O.getWeights()), axis=0)
            W2 = lstm2.W
            #self.chkEqual(W.transpose(), W2)

    def chkEqual(self, a, b):
        tolerance = 1e-1
        a = a.reshape(a.size)
        b = b.reshape(b.size)
        for i in range(a.size):
            if not ((a[i] == 0 and b[i] == 0) or
                (np.abs(a[i]) < 1e-8 and np.abs(b[i]) < 1e-8) or
                (np.abs(a[i] / b[i] - 1) < tolerance)):
                    print a[i], b[i], a[i]/b[i]
            self.assertTrue(
                (a[i] == 0 and b[i] == 0) or
                (np.abs(a[i]) < 1e-8 and np.abs(b[i]) < 1e-8) or
                (np.abs(a[i] / b[i] - 1) < tolerance))
if __name__ == '__main__':
    unittest.main()
