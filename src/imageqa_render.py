import sys
import os
import json
import re
import cPickle as pkl

from nn.func import *
import imageqa_test as it
import imageqa_ensemble as ie
import requests

jsonTrainFilename = '/ais/gobi3/datasets/mscoco/annotations/captions_train2014.json'
jsonValidFilename = '/ais/gobi3/datasets/mscoco/annotations/captions_val2014.json'
htmlHyperLink = '%d.html'
cssHyperLink = 'style.css'
daquarImageFolder = 'http://www.cs.toronto.edu/~mren/imageqa/data/nyu-depth-v2/jpg/'

def renderLatexAnswerList(
                            correctAnswer, 
                            topAnswers, 
                            topAnswerScores):
    result = []
    for i, answer in enumerate(topAnswers):
        if answer == correctAnswer:
            colorStr = '\\textcolor{green}{%s}'
        elif i == 0:
            colorStr = '\\textcolor{red}{%s}'
        else:
            colorStr = '%s'
        result.append(colorStr % ('%s (%.4f) ' % \
                    (answer, topAnswerScores[i])))
    return ''.join(result)

def renderLatexSingleItem(
                    questionIndex,
                    question,
                    correctAnswer,
                    pictureFolder='img',
                    comment=None,
                    topAnswers=None,
                    topAnswerScores=None,
                    modelNames=None):
    result = []
    imgPath = os.path.join(pictureFolder, '%d.jpg' % questionIndex)
    result.append('    \\scalebox{0.3}{\n')
    result.append('        \\includegraphics[width=\\textwidth, \
                    height=.7\\textwidth]{%s}}\n' % imgPath)
    result.append('    \\parbox{5cm}{\n')
    result.append('        \\vskip 0.05in\n')
    result.append('        Q%d: %s\\\\\n' % (questionIndex, question))
    result.append('        Ground truth: %s\\\\\n' % correctAnswer)
    i = 0
    if modelNames is not None and len(modelNames) > 1:
        for modelAnswer, modelAnswerScore, modelName in \
            zip(topAnswers, topAnswerScores, modelNames):
            result.append('%s: ' % modelName)
            result.append(
                renderLatexAnswerList(
                                 correctAnswer,
                                 modelAnswer,
                                 modelAnswerScore))
            if i != len(modelNames) - 1:
                result.append('\\\\')
            i += 1
            result.append('\n')
    elif topAnswers is not None:
        result.append(
            renderLatexAnswerList(
                             correctAnswer,
                             topAnswers, 
                             topAnswerScores))
        result.append('\n')
    if comment is not None:
        result.append('\\\\\n' + comment)
    result.append('}\n')
    return ''.join(result)

def renderLatexSinglePage(
                            inputData,
                            targetData,
                            questionIdict,
                            ansIdict,
                            urlDict,
                            outputFolder,
                            pictureFolder='img',
                            topK=10,
                            comments=None,
                            caption=None,
                            modelOutputs=None,
                            modelNames=None,
                            questionIds=None):
    result = []
    result.append('\\begin{figure*}[ht!]\n')
    result.append('\\centering\\small\n')
    result.append('$\\begin{array}{p{5cm} p{5cm} p{5cm}}\n')
    imgPerRow = 3
    imgFolder = os.path.join(outputFolder, pictureFolder)
    for n in range(inputData.shape[0]):
        # Download the images
        imageId = inputData[n, 0, 0]
        imageFilename = urlDict[imageId - 1]
        r = requests.get(imageFilename)
        qid = questionIds[n] if questionIds is not None else n
        if not os.path.exists(imgFolder):
            os.makedirs(imgFolder)
        with open(os.path.join(imgFolder, '%d.jpg' % qid), 'wb') as f:
            f.write(r.content)
        question = it.decodeQuestion(inputData[n], questionIdict)
        answer = ansIdict[targetData[n, 0]]
        topAnswers, topAnswerScores = \
                    pickTopAnswers(
                            ansIdict,
                            n,
                            topK=topK,
                            modelOutputs=modelOutputs, 
                            modelNames=modelNames)
        comment = comments[n] \
                if comments is not None else None
        result.append(
            renderLatexSingleItem(
                            qid,
                            question,
                            answer,
                            pictureFolder=pictureFolder,
                            comment=comment,
                            topAnswers=topAnswers,
                            topAnswerScores=topAnswerScores,
                            modelNames=modelNames))
        if np.mod(n, imgPerRow) == imgPerRow - 1:
            result.append('\\\\\n')
            if n != inputData.shape[0] - 1:
                result.append('\\noalign{\\smallskip}\\\
                    noalign{\\smallskip}\\noalign{\\smallskip}\n')
        else:
            result.append('&\n')
    result.append('\end{array}$\n')
    result.append('\caption{%s}\n' % caption if caption is not None else '')
    result.append('\end{figure*}\n')
    return ''.join(result)
    

def renderLatex(
                inputData,
                targetData,
                questionIdict,
                ansIdict,
                urlDict,
                outputFolder,
                pictureFolder='img',
                topK=10,
                comments=None,
                caption=None,
                modelOutputs=None,
                modelNames=None,
                questionIds=None,
                filename='result.tex'):
    imgPerRow = 3
    rowsPerPage = 3
    imgPerPage = imgPerRow * rowsPerPage
    imgFolder = os.path.join(outputFolder, 'img')
    if inputData.shape[0] < imgPerPage:
        latexStr = renderLatexSinglePage(
                        inputData,
                        targetData,
                        questionIdict,
                        ansIdict,
                        urlDict,
                        outputFolder,
                        pictureFolder=pictureFolder,
                        topK=topK,
                        comments=comments,
                        caption=caption,
                        modelOutputs=modelOutputs,
                        modelNames=modelNames,
                        questionIds=questionIds)
    else:
        result = []
        numPages = int(np.ceil(inputData.shape[0] / float(imgPerPage)))
        for i in range(numPages):
            start = imgPerPage * i
            end = min(inputData.shape[0], imgPerPage * (i + 1))
            if modelOutputs is not None:
                modelOutputSlice = []
                for j in range(len(modelNames)):
                    modelOutputSlice.append(modelOutputs[j][start:end])
            else:
                modelOutputSlice = None
            page = renderLatexSinglePage(
                        inputData[start:end],
                        targetData[start:end],
                        questionIdict,
                        ansIdict,
                        urlDict,
                        outputFolder,
                        pictureFolder=pictureFolder,
                        topK=topK,
                        comments=comments[start:end] \
                        if comments is not None else None,
                        caption=caption,
                        modelOutputs=modelOutputSlice,
                        modelNames=modelNames, 
                        questionIds=questionIds[start:end])
            result.append(page)
        latexStr = ''.join(result)
    with open(os.path.join(outputFolder, filename), 'w') as f:
        f.write(latexStr)

def renderHtml(
                inputData,
                targetData,
                questionIdict,
                ansIdict,
                urlDict,
                topK=10,
                modelOutputs=None,
                modelNames=None,
                questionIds=None):
    imgPerPage = 1000
    if inputData.shape[0] < imgPerPage:
        return [renderSinglePage(
                    inputData,
                    targetData,
                    questionIdict,
                    ansIdict,
                    urlDict,
                    iPage=0,
                    numPages=1,
                    topK=topK,
                    modelOutputs=modelOutputs,
                    modelNames=modelNames, 
                    questionIds=questionIds)]
    else:
        result = []
        numPages = int(np.ceil(inputData.shape[0] / float(imgPerPage)))
        for i in range(numPages):
            start = imgPerPage * i
            end = min(inputData.shape[0], imgPerPage * (i + 1))
            if modelOutputs is not None:
                modelOutputSlice = []
                for j in range(len(modelNames)):
                    modelOutputSlice.append(modelOutputs[j][start:end])
            else:
                modelOutputSlice = None
            page = renderSinglePage(
                        inputData[start:end],
                        targetData[start:end],
                        questionIdict,
                        ansIdict,
                        urlDict, 
                        iPage=i,
                        numPages=numPages,
                        topK=topK,
                        modelOutputs=modelOutputSlice,
                        modelNames=modelNames,
                        questionIds=questionIds[start:end])
            result.append(page)
        return result

def renderMenu(iPage, numPages):
    htmlList = []
    htmlList.append('<div>Navigation: ')
    for n in range(numPages):
        if n != iPage:
            htmlList.append('<a href=%s> %d </a>' % \
                        ((htmlHyperLink % n), n))
        else:
            htmlList.append('<span> %d </span>' % n)

    htmlList.append('</div>')
    return ''.join(htmlList)

def renderCss():
    cssList = []
    cssList.append('table {\
                            width:1200px;\
                            border-spacing:10px;\
                          }\n')
    cssList.append('td.item {\
                             padding:5px;\
                             border:1px solid gray;\
                             vertical-align:top;\
                            }\n')
    cssList.append('div.ans {\
                             margin-top:10px;\
                             width:300px;\
                            }')
    cssList.append('img {width:300px; height:200px;}\n')
    cssList.append('span.good {color:green;}\n')
    cssList.append('span.bad {color:red;}\n')
    return ''.join(cssList)

def renderAnswerList(
                    correctAnswer, 
                    topAnswers, 
                    topAnswerScores):
    htmlList = []
    for i, answer in enumerate(topAnswers):
        if answer == correctAnswer:
            colorStr = 'class="good"'
        elif i == 0:
            colorStr = 'class="bad"'
        else:
            colorStr = ''
        htmlList.append('<span %s>%d. %s %.4f</span><br/>' % \
                    (colorStr, i + 1, 
                    answer, topAnswerScores[i]))
    return ''.join(htmlList)

def renderSingleItem(
                    imageFilename,
                    questionIndex,
                    question,
                    correctAnswer,
                    topAnswers=None,
                    topAnswerScores=None,
                    modelNames=None):
    """
    Render a single item.
    topAnswers: a list of top answer strings
    topAnswerScores: a list of top answer scores
    modelNames: if multiple items, then above are list of lists.
    """
    htmlList = []
    htmlList.append('<td class="item">\
                    <div class="img">\
                    <img src="%s"/></div>\n' % \
                    imageFilename)
    htmlList.append('<div class="ans">Q%d: %s<br/>' % \
                    (questionIndex, question))
    htmlList.append('Correct answer: <span class="good">\
                    %s</span><br/>' % correctAnswer)
    if topAnswers is not None:
        for modelAnswer, modelAnswerScore, modelName in \
            zip(topAnswers, topAnswerScores, modelNames):
            htmlList.append('%s:<br/>' % modelName)
            htmlList.append(
                renderAnswerList(
                                 correctAnswer,
                                 modelAnswer,
                                 modelAnswerScore))
    htmlList.append('</div></td>')
    return ''.join(htmlList)

def pickTopAnswers(
                    ansIdict,
                    n,
                    topK=10,
                    modelOutputs=None, 
                    modelNames=None,
                    questionIds=None):
    if modelOutputs is not None:
        topAnswers = []
        topAnswerScores = []
        for j, modelOutput in enumerate(modelOutputs):
            sortIdx = np.argsort(modelOutput[n], axis=0)
            sortIdx = sortIdx[::-1]
            topAnswers.append([])
            topAnswerScores.append([])
            for i in range(0, topK):
                topAnswers[-1].append(ansIdict[sortIdx[i]])
                topAnswerScores[-1].append(modelOutput[n, sortIdx[i]])
    else:
        topAnswers = None
        topAnswerScores = None
    return topAnswers, topAnswerScores

def renderSinglePage(
                    inputData, 
                    targetData, 
                    questionIdict, 
                    ansIdict, 
                    urlDict,
                    iPage=0, 
                    numPages=1,
                    topK=10,
                    modelOutputs=None,
                    modelNames=None,
                    questionIds=None):
    htmlList = []
    htmlList.append('<html><head>\n')
    htmlList.append('<style>%s</style>' % renderCss())
    htmlList.append('</head><body>\n')
    htmlList.append('<table>')
    imgPerRow = 4
    htmlList.append(renderMenu(iPage, numPages))
    for n in range(inputData.shape[0]):
        if np.mod(n, imgPerRow) == 0:
            htmlList.append('<tr>')
        imageId = inputData[n, 0, 0]
        imageFilename = urlDict[imageId - 1]
        question = it.decodeQuestion(inputData[n], questionIdict)

        qid = questionIds[n] if questionIds is not None else n
        topAnswers, topAnswerScores = pickTopAnswers(
                                ansIdict, 
                                n,
                                topK=topK,
                                modelOutputs=modelOutputs, 
                                modelNames=modelNames,
                                questionIds=questionIds)
        htmlList.append(renderSingleItem(
                                imageFilename, 
                                qid, 
                                question, 
                                ansIdict[targetData[n, 0]], 
                                topAnswers=topAnswers, 
                                topAnswerScores=topAnswerScores, 
                                modelNames=modelNames))

        if np.mod(n, imgPerRow) == imgPerRow - 1:
            htmlList.append('</tr>')
    htmlList.append('</table>')
    htmlList.append(renderMenu(iPage, numPages))
    htmlList.append('</body></html>')
    return ''.join(htmlList)

def readImgDictCocoqa(imgidDict):
    with open(jsonTrainFilename) as f:
        captiontxt = f.read()
    urlDict = {}
    caption = json.loads(captiontxt)
    for item in caption['images']:
        urlDict[item['id']] = item['url']

    with open(jsonValidFilename) as f:
        captiontxt = f.read()
    caption = json.loads(captiontxt)
    for item in caption['images']:
        urlDict[item['id']] = item['url']
    urlList = [None] * len(imgidDict)
    for i, key in enumerate(imgidDict):
        urlList[i] = urlDict[int(key)]
    return urlList

def readImgDictDaquar():
    urlList = []
    for i in range(1, 1450):
        urlList.append(daquarImageFolder + 'image%d.jpg' % i)
    return urlList

def loadImgUrl(dataset, dataFolder):
    print 'Loading image urls...'
    if dataset == 'cocoqa':
        imgidDictFilename = \
            os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        urlDict = readImgDictCocoqa(imgidDict)
    elif dataset == 'daquar':
        urlDict = readImgDictDaquar()
    return urlDict

def loadImgPath(dataset, dataFolder):
    print 'Loading image paths...'
    cocoImgIdRegex = 'COCO_((train)|(val))2014_0*(?P<imgid>[1-9][0-9]*)'
    if dataset == 'cocoqa':
        imgidDictFilename = \
            os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        pathList = [None] * len(imgidDict)
        pathDict = {}
        with open('/u/mren/data/mscoco/train/image_list.txt') as f:
            imageList = f.readlines()
        with open('/u/mren/data/mscoco/valid/image_list.txt') as f:
            imageList.extend(f.readlines())
        for imgPath in imageList:
            match = re.search(cocoImgIdRegex, imgPath)
            imgid = match.group('imgid')
            pathDict[imgid] = imgPath[:-1]
        for i, key in enumerate(imgidDict):
            pathList[i] = pathDict[key]
        return pathList
    elif dataset == 'daquar':
        pathList = []
        for i in range(1, 1450):
            pathList.append('/u/mren/data/nyu-depth/jpg/image%d.jpg' % i)
        return pathList
    else:
        raise Exception('Unknown dataset: ' + dataset)

def escapeLatexIdict(idict):
    for i in range(len(idict)):
        if '_' in idict[i]:
            idict[i] = idict[i].replace('_', '\\_')
    return idict

def parseComparativeParams(argv):
    """
    Parse parameter list for rendering comparative results.
    Usage:
        -m[odel] {name1:modelId1}
        -m[odel] {name2:modelId2}
        -em[odel] {name3:ensembleModelId3,ensembleModelId4,...}
        -pem[odel] {name4:ensembleModelId5,ensembleModelId6,...}
        -aem[odel] {name4:ensembleModelId7,ensembleModelId8,...}
                
        ...
        -d[ata] {dataFolder}
        -i[nput] {listFile}
        -o[utput] {outputFolder}
        [-k {top K answers}]
        [-p[icture] {pictureFolder}]
        [-r[esults] {resultsFolder}]
        [-f[ile] {outputFilename}]
        [-dataset {daquar/cocoqa}]
        [-format {html/latex}]

    Parameters:
        -m[odel]: Normal models
        -em[odel]: Class-specific-ensemble models
        -pem[odel]: Class-specific-ensemble + prior models
        -aem[odel]: Ensemble average of regular models
        -d[ata]: Data folder
        -i[nput]: Task-specific input file
        -o[utput]: Output folder
        -k: Top K answers, default 1
        -p[icture]: Picture folder name, default "img"
        -r[esults]: Results folder, default "../results"
        -f[ile]: Output TeX file name, default "result"
        -dataset: daquar/cocoqa dataset, default "cocoqa"
        -format: Output format, default "html"

    Returns:
        A dictionary with following keys:
        models: List of dictionaries, with following keys:
            modelName: string
            isClassEnsemble: boolean
            runPrior: boolean
        dataFolder: string
        inputFile: string
        outputFolder: string
        topK: int
        pictureFolder: string
        outputFilename: string
        dataset: string
    """
    dataset = 'cocoqa'
    filename = 'result'
    pictureFolder = 'img'
    resultsFolder = '../results'
    K = 1
    models = []
    format = 'html'
    inputFile = None
    for i, flag in enumerate(argv):
        if flag == '-m' or flag == '-model':
            parts = sys.argv[i + 1].split(':')
            models.append({
                'name': parts[0],
                'id': parts[1],
                'isClassEnsemble': False,
                'isAverageEnsemble': False,
                'runPrior': False
            })
        elif flag == '-em' or flag == '-emodel':
            parts = sys.argv[i + 1].split(':')
            models.append({
                'name': parts[0],
                'id': parts[1],
                'isClassEnsemble': True,
                'isAverageEnsemble': False,
                'runPrior': False
            })
        elif flag == '-pem' or flag == '-pemodel':
            parts = sys.argv[i + 1].split(':')
            models.append({
                'name': parts[0],
                'id': parts[1],
                'isClassEnsemble': True,
                'isAverageEnsemble': False,
                'runPrior': True
            })
        elif flag == '-aem' or flag == '-aemodel':
            parts = sys.argv[i + 1].split(':')
            models.append({
                'name': parts[0],
                'id': parts[1],
                'isClassEnsemble': False,
                'isAverageEnsemble': True,
                'runPrior': False
            })
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-i' or flag == '-input':
            inputFile = sys.argv[i + 1]
        elif flag == '-k':
            K = int(sys.argv[i + 1])
        elif flag == '-p' or flag == '-picture':
            pictureFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-f' or flag == '-file':
            filename = sys.argv[i + 1]
        elif flag == '-dataset':
            dataset = sys.argv[i + 1]
        elif flag == '-format':
            format = sys.argv[i + 1]
    return {
        'models': models,
        'dataFolder': dataFolder,
        'inputFile': inputFile,
        'outputFolder': outputFolder,
        'resultsFolder': resultsFolder,
        'topK': K,
        'pictureFolder': pictureFolder,
        'outputFilename': filename,
        'dataset': dataset,
        'format': format
    }

def getModelNames(modelSpecs):
    modelNames = []
    for spec in modelSpecs:
        modelNames.append(spec['name'])
    return modelNames

if __name__ == '__main__':
    """
    Render HTML of a single model.
    Usage: python imageqa_render.py 
                        -m[odel] {name:modelId}
                        -em[odel] {name:ensembleModelId1,ensembleModelId2,...}
                        -pem[odel] {name:ensembleModelId3,ensembleModelId4,...}
                        -aem[odel] {name:ensembleModelId5,ensembleModelId6,...}
                        -d[ata] {data folder}
                        -o[utput] {output folder}
                        [-k {Top K answers}]
                        [-dataset {daquar/cocoqa}]
                        [-r[esults] {results folder}]
    Parameters:
        -m[odel]: Model ID
        -em[odel]: Ensemble model ID
        -pem[odel]: Ensemble model ID with prior
        -d[ata]: Data folder
        -o[utput]: Output folder
        -k: Top K answers
        -dataset: DAQUAR/COCO-QA dataset
    """
    params = parseComparativeParams(sys.argv)

    print('Loading test data...')
    urlDict = loadImgUrl(params['dataset'], params['dataFolder'])
    data = it.loadDataset(params['dataFolder'])

    if len(params['models']) > 0:
        print('Running models...')
        singleModel = [params['models'][0]]
        modelOutputs = ie.runAllModels(
                        data['testData'][0], 
                        data['questionTypeArray'], 
                        singleModel, 
                        params['resultsFolder'],
                        params['dataset'],
                        params['dataFolder'])
    else:
        modelOutputs = None

    # Render
    print('Rendering HTML to %s' % params['outputFolder'])
    if not os.path.exists(params['outputFolder']):
        os.makedirs(params['outputFolder'])

    pages = renderHtml(
                data['testData'][0], 
                data['testData'][1], 
                data['questionIdict'], 
                data['ansIdict'], 
                urlDict, 
                topK=params['topK'], 
                modelOutputs=modelOutputs,
                modelNames=getModelNames(singleModel),
                questionIds=np.arange(data['testData'][0].shape[0]))

    for i, page in enumerate(pages):
        with open(os.path.join(params['outputFolder'], 
                htmlHyperLink % i), 'w') as f:
            f.write(page)
