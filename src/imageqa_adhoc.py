import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_ensemble as ie
import imageqa_render as ir
import prep

def parseInputFile(filename):
    qids = []
    questions = []
    answers = []
    caption = ''
    i = 0
    with open(filename) as f:
        for line in f:
            if i == 0 and line.startswith('caption:'):
                caption = line[8:-1]
            else:
                parts = line.split(',')
                qids.append(int(parts[0]))
                questions.append(parts[1])
                answers.append(parts[2].strip('\n'))
            i += 1
    return caption, qids, questions, answers

if __name__ == '__main__':
    """
    Ask adhoc questions with trained models.

    Usage: python imageqa_adhoc.py  
                    -m[odel] {name1:modelId1}
                    -m[odel] {name2:modelId2}
                    -em[odel] {name3:ensembleModelId3,ensembleModelId4,...}
                    -pem[odel] {name3:ensembleModelId5,ensembleModelId6,...}
                    -aem[odel] {name3:ensembleModelId7,ensembleModelId8,...}
                    ...
                    -d[ata] {dataFolder}
                    -i[nput] {listFile}
                    -o[utput] {outputFolder}
                    [-k {top K answers}]
                    [-p[icture] {pictureFolder}]
                    [-r[esults] {resultsFolder}]
                    [-f[ile] {outputTexFilename}]
                    [-dataset {daquar/cocoqa}]
                    [-format {html/latex}]
    Parameters:
        -m[odel]: Model name and model ID
        -d[ata]: Dataset dataFolder
        -i[nput]: Adhoc question list filename
        -o[utput]: Output folder of the rendered results
        -k: Render top-K answers (default 1)
        -p[icture]: Picture folder, only required in LaTeX mode (default "img")
        -r[esults]: Results folder where trained models are stored (default "../results")
        -f[ile]: Output filename, only required in LaTex mode
        -dataset: Use DAQUAR/COCO-QA dataset (default "cocoqa")
        -format: Set output format to HTML/LaTeX (default "html")

    Input question list format:
        QID1,Question1,GroundTruthAnswer1
        QID2,Question2,GroundTruthAnswer2
        ...
    """
    params = ir.parseComparativeParams(sys.argv)

    urlDict = ir.loadImgUrl(params['dataset'], params['dataFolder'])
    data = it.loadDataset(params['dataFolder'])
    maxlen = data['testData'][0].shape[1]

    print('Parsing input file...')
    caption, qids, questions, answers = parseInputFile(params['inputFile'])
    idx = np.array(qids, dtype='int')
    #inputTestSel = data['testData'][0][idx]
    #targetTestSel = data['testData'][1][idx]
    imgids = qids
    #imgids = inputTestSel[:, 0, 0]
    inputTest = prep.combine(\
        prep.lookupQID(questions, data['questionDict'], maxlen), imgids)
    targetTest = prep.lookupAnsID(answers, data['ansDict'])
    questionTypeArray = data['questionTypeArray'][idx]

    print('Running models...')
    modelOutputs = ie.runAllModels(
                        inputTest, 
                        questionTypeArray, 
                        params['models'],
                        params['resultsFolder'],
                        params['dataset'],
                        params['dataFolder'])

    # Render
    if not os.path.exists(params['outputFolder']):
        os.makedirs(params['outputFolder'])
    if params['format'] == 'html':
        print('Rendering HTML...')
        pages = ir.renderHtml(
                    inputTest,
                    targetTest,
                    data['questionIdict'],
                    data['ansIdict'],
                    urlDict,
                    topK=params['topK'],
                    modelOutputs=modelOutputs,
                    modelNames=ir.getModelNames(params['models']),
                    questionIds=idx)
        for i, page in enumerate(pages):
            with open(os.path.join(params['outputFolder'],
                '%s-%d.html' % (params['outputFilename'], i)), 'w') as f:
                f.write(page)
    elif params['format'] == 'latex':
        # For LaTeX only, replace underscore in vocabulary.
        data['questionIdict'] = ir.escapeLatexIdict(data['questionIdict'])
        data['ansIdict'] = ir.escapeLatexIdict(data['ansIdict'])
        ir.renderLatex(
                inputTest,
                targetTest,
                data['questionIdict'],
                data['ansIdict'],
                urlDict, 
                topK=params['topK'],
                outputFolder=params['outputFolder'],
                pictureFolder=params['pictureFolder'],
                comments=None,
                caption=caption,
                modelOutputs=modelOutputs,
                modelNames=ir.getModelNames(params['models']),
                questionIds=idx,
                filename=params['outputFilename']+'.tex')
