import sys
import time
import numpy as np
import lstm_old as l

start = time.time()
timespan = 100
multiErr = len(sys.argv) > 1 and sys.argv[1] == 'm'
for i in range(0, 10):
    lstm = l.LSTM_Old(
        inputDim=100,
        outputDim=100,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=multiErr,
        outputdEdX=True)
    X = np.random.rand(10, timespan, 100)
    Y = lstm.forward(X)
    dEdY = np.random.rand(10, timespan, 100) if multiErr else np.random.rand(10, 100)
    dEdY = lstm.backward(dEdY)
print '%.4f s' % (time.time() - start)