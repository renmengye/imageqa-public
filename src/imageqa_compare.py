import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_render as ir
import imageqa_ensemble as ie

nameList = ['object', 'number', 'color', 'location']

def getCatName(i):
    return nameList[i]

def getBinName(n):
    bin = []
    for k in range(numModels):
        bin.append(str(n >> (numModels - k - 1)))
        n = n & (~(1 << (numModels - k - 1)))
    return ''.join(bin)

def getName(catName, binName):
    return catName + '-' + binName

def renderIndex(modelNames, numCategories, bins):
    htmlList = []
    htmlList.append('<html><head><style>%s</style><body>' % \
        'span.good {color:green;} span.bad {color:red;} \
        table{border-spacing:10px;}')
    numModels = len(modelNames)
    numCorrect = 1 << numModels
    htmlList.append('<h1>Models comparisons</h1>')
    htmlList.append('Notes:<br/><span class="good">\
        Green means the model gets correct</span><br/>')
    htmlList.append('<span class="bad">\
        Red means the model gets wrong</span>')
    for i in range(numCategories):
        htmlList.append('<h2>%s</h2>' % getCatName(i))
        htmlList.append('<table>')
        for j in range(numCorrect):
            htmlList.append('<tr>')
            binId = numCorrect * i + j
            for k, c in enumerate(getBinName(j)):
                htmlList.append('<td>')
                if c == '1':
                    htmlList.append(
                        '<span class="good">%s</span>' % modelNames[k])
                elif c == '0':
                    htmlList.append(
                        '<span class="bad">%s</span>' % modelNames[k])
                htmlList.append('</td>')
            htmlList.append('<td>%d items</td>' % len(bins[binId]))
            htmlList.append('<td><a href="%s/0.html">link</a></td>' % \
                getName(getCatName(i), getBinName(j)))
            htmlList.append('</tr>')

        htmlList.append('</table>')
    htmlList.append('</head></body></html>')
    return ''.join(htmlList)

if __name__ == '__main__':
    """
    Usage: python imageqa_compare.py 
                    -m[odel] {name1:modelId1}
                    -m[odel] {name2:modelId2}
                    -em[odel] {name3:ensembleModelId3,ensembleModelId4,...}
                    -pem[odel] {name3:ensembleModelId5,ensembleModelId6,...}
                    -d[ata] {dataFolder}
                    -o[utput] {outputFolder}
                    [-k {top K answers}]
                    [-r[esults] {resultsFolder}]
                    [-dataset {daquar/cocoqa}]
    """
    params = ir.parseComparativeParams(sys.argv)
    
    urlDict = ir.loadImgUrl(params['dataset'], params['dataFolder'])
    data = it.loadDataset(params['dataFolder'])

    print('Running models...')
    inputTest = data['testData'][0]
    targetTest = data['testData'][1]
    questionTypeArray = data['questionTypeArray']
    modelOutputs = ie.runAllModels(
                        inputTest, 
                        questionTypeArray, 
                        params['models'], 
                        params['resultsFolder'],
                        params['dataset'],
                        params['dataFolder'])

    # Sort questions by question types.
    # Sort questions by correctness differences.
    print('Sorting questions...')
    numCategories = np.max(questionTypeArray) + 1
    numModels = len(params['models'])
    numCorrect = 1 << numModels
    numBins = numCategories * numCorrect
    modelAnswers = np.zeros((numModels, inputTest.shape[0]), dtype='int')
    bins = [None] * numBins
    names = []
    for i in range(numCategories):
        catName = getCatName(i)
        for j in range(numCorrect):
            binName = getBinName(j)
            names.append(getName(catName, binName))
    for i in range(numModels):
        modelAnswers[i] = np.argmax(modelOutputs[i], axis=-1)
    for n in range(inputTest.shape[0]):
        correct = targetTest[n, 0]
        bintmp = 0
        for i in range(numModels):
            if modelAnswers[i, n] == correct:
                bintmp += 1 << (numModels - i - 1)
        category = questionTypeArray[n]
        binNum = category * numCorrect + bintmp
        if bins[binNum] == None:
            bins[binNum] = [n]
        else:
            bins[binNum].append(n)

    for i, bin in enumerate(bins):
        if bin is None:
            bins[i] = []

    # Render
    print('Rendering webpages...')
    print('Rendering index...')
    outputFolder = params['outputFolder']

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    with open(os.path.join(outputFolder, 'index.html'), 'w') as f:
        f.write(renderIndex(
            ir.getModelNames(params['models']), numCategories, bins))

    for i in range(numBins):
        if bins[i] is not None:
            print 'Rendering %s...' % names[i]
            outputSubFolder = os.path.join(outputFolder, names[i])
            idx = np.array(bins[i], dtype='int')
            inputTestSubset = inputTest[idx]
            targetTestSubset = targetTest[idx]
            modelOutputsSubset = []
            for j in range(numModels):
                modelOutputsSubset.append(modelOutputs[j][idx])
            if not os.path.exists(outputSubFolder):
                os.makedirs(outputSubFolder)
            htmlHyperLink = '%d.html'
            pages = ir.renderHtml(
                        inputTestSubset, 
                        targetTestSubset, 
                        data['questionIdict'], 
                        data['ansIdict'], 
                        urlDict, 
                        topK=params['topK'],
                        modelOutputs=modelOutputsSubset,
                        modelNames=ir.getModelNames(params['models']),
                        questionIds=idx)
            for j, page in enumerate(pages):
                with open(os.path.join(outputSubFolder, 
                        htmlHyperLink % j), 'w') as f:
                    f.write(page)