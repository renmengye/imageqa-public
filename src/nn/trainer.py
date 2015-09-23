import time
import sys
import os
import shutil
import matplotlib
import valid_tool as vt
import tester
from func import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class ProgressWriter:
    def __init__(self, total, width=80):
        self.total = total
        self.counter = 0
        self.progress = 0
        self.width = width

    def increment(self, amount=1):
        self.counter += amount
        while self.counter / float(self.total) > \
              self.progress / float(self.width):
            sys.stdout.write('.')
            sys.stdout.flush()
            self.progress += 1
        if self.counter == self.total and \
           self.progress < self.width:
            print

class Logger:
    def __init__(self, trainer, csv=True, filename=None):
        self.trainer = trainer
        self.startTime = time.time()
        self.saveCsv = csv
        if filename is None:
            self.outFilename = os.path.join(
                trainer.outputFolder, trainer.name + '.csv')
        else:
            self.outFilename = filename

    def logMsg(self, msg):
        print msg

    def logTrainStats(self):
        timeElapsed = time.time() - self.startTime
        stats = 'N: %3d T: %5d  TE: %8.4f  TR: %8.4f  VE: %8.4f  VR: %8.4f' % \
                (self.trainer.epoch,
                 timeElapsed,
                 self.trainer.loss[self.trainer.totalStep],
                 self.trainer.rate[self.trainer.totalStep],
                 self.trainer.validLoss[self.trainer.totalStep],
                 self.trainer.validRate[self.trainer.totalStep])
        print stats

        if self.saveCsv:
            statsCsv = '%d,%.4f,%.4f,%.4f,%.4f' % \
                    (self.trainer.epoch,
                     self.trainer.loss[self.trainer.totalStep],
                     self.trainer.rate[self.trainer.totalStep],
                     self.trainer.validLoss[self.trainer.totalStep],
                     self.trainer.validRate[self.trainer.totalStep])
            with open(self.outFilename, 'a+') as f:
                f.write('%s\n' % statsCsv)
    pass

class Plotter:
    def __init__(self, trainer):
        self.trainer = trainer
        self.startTime = time.time()
        self.trainer.epoch = 0
        self.lossFigFilename = \
            os.path.join(trainer.outputFolder, trainer.name + '_loss.png')
        self.errFigFilename = \
            os.path.join(trainer.outputFolder, trainer.name + '_err.png')
        self.trainer.epoch = 0

    def plot(self):
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(self.trainer.totalStep + 1),
                 self.trainer.loss[0 : self.trainer.totalStep + 1], 
                 'b-x')
        if self.trainer.trainOpt['needValid']:
            plt.plot(np.arange(self.trainer.totalStep + 1),
                     self.trainer.validLoss[0 : self.trainer.totalStep + 1], 
                     'g-o')
            plt.legend(['Train', 'Valid'])
            plt.title('Train/Valid Loss Curve')
        else:
            plt.legend(['Train'])
            plt.title('Train Loss Curve')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.draw()
        plt.savefig(self.lossFigFilename)

        if self.trainer.trainOpt['calcError']:
            plt.figure(2)
            plt.clf()
            plt.plot(np.arange(self.trainer.totalStep + 1),
                     1 - self.trainer.rate[0 : self.trainer.totalStep + 1], 
                     'b-x')
            if self.trainer.trainOpt['needValid']:
                plt.plot(np.arange(self.trainer.totalStep + 1),
                         1 - self.trainer.validRate[\
                         0 : self.trainer.totalStep + 1],
                         'g-o')
                plt.legend(['Train', 'Valid'])
                plt.title('Train/Valid Error Curve')
            else:
                plt.legend(['Train'])
                plt.title('Train Error Curve')

            plt.xlabel('Epoch')
            plt.ylabel('Prediction Error')
            plt.draw()
            plt.savefig(self.errFigFilename)

class Trainer:
    def __init__(self,
                 name,
                 model,
                 trainOpt,
                 outputFolder='',
                 seed=1000):
        self.model = model
        self.name = name + time.strftime("-%Y%m%d-%H%M%S")
        self.resultsFolder = outputFolder
        self.outputFolder = os.path.join(outputFolder, self.name)
        self.modelFilename = \
            os.path.join(self.outputFolder, self.name + '.w.npy')
        self.trainOpt = trainOpt
        self.startTime = time.time()
        self.random = np.random.RandomState(seed)
        numEpoch = trainOpt['numEpoch']
        self.loss = 0
        self.validLoss = 0
        self.rate = 0
        self.validRate = 0
        self.epoch =  0

    def initFolder(self):
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        if self.model.specFilename is not None:
            shutil.copyfile(
                self.model.specFilename,
                os.path.join(self.outputFolder, self.name + '.model.yml'))

    def initData(self, X, T, split=True):
        VX = None
        VT = None
        X, T = vt.shuffleData(X, T, self.random)
        if split:
            X, T, VX, VT = \
                vt.splitData(X, T,
                            self.trainOpt['heldOutRatio'],
                            self.trainOpt['xvalidNo'])
        return X, T, VX, VT

    def train(
                self, 
                trainInput, 
                trainTarget, 
                trainInputWeights=None,
                validInput=None, 
                validTarget=None,
                validInputWeights=None):
        self.initFolder()
        trainOpt = self.trainOpt
        if validInput is None and validTarget is None:
            X, T, VX, VT = self.initData(\
                trainInput, trainTarget, \
                split=self.trainOpt['needValid'])
        else:
            X = trainInput
            T = trainTarget
            VX = validInput
            VT = validTarget
        N = X.shape[0]
        print 'Epoch size:', N
        numEpoch = trainOpt['numEpoch']
        calcError = trainOpt['calcError']
        numExPerBat = trainOpt['batchSize']
        print 'Batch size:', numExPerBat
        numBatPerStep = trainOpt['stepSize'] \
            if trainOpt.has_key('stepSize') \
            else int(np.ceil(N / float(numExPerBat)))
        print 'Step size:', numBatPerStep
        numExPerStep = numExPerBat * numBatPerStep \
            if trainOpt.has_key('stepSize') \
            else N
        print 'Examples per step:', numExPerStep
        numStepPerEpoch = int(np.ceil(
            N / float(numExPerStep))) \
            if trainOpt.has_key('stepSize') \
            else 1
        print 'Steps per epoch:', numStepPerEpoch
        progressWriter = ProgressWriter(numExPerStep, width=80)
        logger = Logger(self, csv=trainOpt['writeRecord'])
        logger.logMsg('Trainer ' + self.name)
        plotter = Plotter(self)
        bestVscore = None
        bestTscore = None
        bestStep = 0
        totalBat = 0
        step = 0
        totalStep = 0
        nAfterBest = 0
        stop = False
        self.loss = np.zeros((numStepPerEpoch * numEpoch))
        self.validLoss = np.zeros((numStepPerEpoch * numEpoch))
        self.rate = np.zeros((numStepPerEpoch * numEpoch))
        self.validRate = np.zeros((numStepPerEpoch * numEpoch))
        
        # Train loop through epochs
        for epoch in range(0, numEpoch):
            self.epoch = epoch
            epochE = 0
            epochCorrect = 0
            epochTotal = 0
            
            # Shuffle data
            if trainOpt['shuffle']:
                X, T = vt.shuffleData(X, T, self.random)
            
            # Every step, validate
            for step in range(0, numStepPerEpoch):
                stepStart = step * numExPerStep
                stepEnd = min((step + 1) * numExPerStep, N)
                numExThisStep = stepEnd - stepStart
                E = 0
                correct = 0
                total = 0
                self.totalStep = totalStep
                
                # Every batch forward-backward
                for batch in range(0, numBatPerStep):
                    batchStart = stepStart + batch * numExPerBat
                    if batchStart > N:
                        break
                    batchEnd = min(
                        stepStart + (batch + 1) * numExPerBat, stepEnd)
                    numExThisBat = batchEnd - batchStart
                    self.totalBatch = totalBat
                    
                    if trainOpt['progress']:
                        progressWriter.increment(amount=numExThisBat)
                    
                    # Forward
                    Y_bat = self.model.forward(
                        X[batchStart:batchEnd], dropout=True)
                    T_bat = T[batchStart:batchEnd]
                    
                    # Loss
                    Etmp, dEdY = self.model.getCost(
                        Y_bat, T_bat, weights=trainInputWeights)
                    E += Etmp * numExThisBat / float(numExThisStep)
                    epochE += Etmp * numExThisBat / float(N)
                    
                    # Backward
                    self.model.backward(dEdY)
                    
                    # Update
                    self.model.updateWeights()
                    
                    # Prediction error
                    if calcError:
                        rate_, correct_, total_ = \
                            tester.calcRate(self.model, Y_bat, T_bat)
                        correct += correct_
                        total += total_
                        epochCorrect += correct_
                        epochTotal += total_
                    
                    totalBat += 1
                
                # Store train statistics
                if calcError:
                    rate = correct / float(total)
                    self.rate[totalStep] = rate
                self.loss[totalStep] = E
                
                # Early stop
                if not trainOpt.has_key('criterion'):
                    Tscore = E
                else:
                    if trainOpt['criterion'] == 'loss':
                        Tscore = E
                    elif trainOpt['criterion'] == 'rate':
                        Tscore = 1 - rate
                    else:
                        raise Exception('Unknown stopping criterion "%s"' % \
                            trainOpt['criterion'])
                
                # Run validation
                if trainOpt['needValid']:
                    VY = tester.test(self.model, VX)
                    VE, dVE = self.model.getCost(
                        VY, VT, weights=validInputWeights)
                    self.validLoss[totalStep] = VE
                    if calcError:
                        Vrate, correct, total = tester.calcRate(
                            self.model, VY, VT)
                        self.validRate[totalStep] = Vrate
                    
                    # Check stopping criterion
                    if not trainOpt.has_key('criterion'):
                        Vscore = VE
                    else:
                        if trainOpt['criterion'] == 'loss':
                            Vscore = VE
                        elif trainOpt['criterion'] == 'rate':
                            Vscore = 1 - Vrate
                        else:
                            raise Exception(
                                'Unknown stopping criterion "%s"' % \
                                trainOpt['criterion'])
                    if (bestVscore is None) or (Vscore < bestVscore):
                        bestVscore = Vscore
                        bestTscore = Tscore
                        nAfterBest = 0
                        bestStep = totalStep

                        # Save trainer if VE is best
                        if trainOpt['saveModel']:
                            self.save()
                    else:
                        nAfterBest += 1
                        # Stop training if above patience level
                        if nAfterBest > trainOpt['patience']:
                            print 'Patience level reached, early stop.'
                            print 'Will stop at score ', bestTscore
                            stop = True
                else:
                    if trainOpt['saveModel']:
                        self.save()
                    if trainOpt.has_key('stopScore') and \
                        Tscore < trainOpt['stopScore']:
                        print \
                            'Training score is lower than %.4f , ealy stop.' % \
                            trainOpt['stopScore'] 
                        stop = True
                
                logger.logTrainStats()
                if trainOpt['needValid']:
                    print 'P: %d' % nAfterBest,
                print self.name
                
                if stop:
                    break
            
            # Store train statistics
            if calcError:
                epochRate = epochCorrect / float(epochTotal)
            print 'Epoch Final: %d TE: %.4f TR:%.4f' % \
                (epoch, epochE, epochRate)
            
            # Anneal learning rate
            self.model.updateLearningParams(epoch)
            
            # Plot train curves
            if trainOpt['plotFigs']:
                plotter.plot()
                
            # Terminate
            if stop:       
                break
                
        # Report best train score
        self.stoppedTrainScore = bestTscore
        
    def save(self, filename=None):
        if filename is None:
            filename = self.modelFilename
        try:
            np.save(filename, self.model.getWeights())
        except Exception:
            print 'Exception occurred. Cannot save weights'

