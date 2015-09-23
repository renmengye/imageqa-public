import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_visprior as ip
import imageqa_ensemble as ie
import imageqa_render as ir

def parseInputFile(filename):
    caption = ''
    selIds = []
    selComments = []
    with open(filename) as f:
        i = 0
        for line in f:
            if i == 0 and line.startswith('caption:'):
                caption = line[8:-1]
            else:
                parts = line.split(',')
                selIds.append(int(parts[0]))
                if len(parts) > 1:
                    selComments.append(parts[1][:-1])
                else:
                    selComments.append('')
            i += 1
    return caption, selIds, selComments

if __name__ == '__main__':
    """
    Render a selection of examples into LaTeX.
    Usage: python imageqa_layout.py 
                    -m[odel] {name1:modelId1}
                    -m[odel] {name2:modelId2}
                    -em[odel] {name3:ensembleModelId3,ensembleModelId4,...}
                    -pem[odel] {name4:ensembleModelId5,ensembleModelId6,...}
                    ...
                    -d[ata] {dataFolder}
                    -i[nput] {listFile}
                    -o[utput] {outputFolder}
                    [-k {top K answers}]
                    [-p[icture] {pictureFolder}]
                    [-f[ile] {outputFilename}]
                    [-daquar/-cocoqa]
    Input file format:
    QID1[,Comment1]
    QID2[,Comment2]
    ...
    """
    params = ir.parseComparativeParams(sys.argv)

    urlDict = ir.loadImgUrl(params['dataset'], params['dataFolder'])
    data = it.loadDataset(params['dataFolder'])

    print('Parsing input file...')
    caption, selIds, selComments = parseInputFile(params['inputFile'])

    print('Running models...')
    idx = np.array(selIds, dtype='int')
    inputTestSel = inputTest[idx]
    targetTestSel = targetTest[idx]
    inputTest = data['testData'][0]
    questionTypeArray = data['questionTypeArray']
    modelOutputs = ie.runAllModels(
                inputTestSel, 
                questionTypeArray[idx], 
                params['models'], 
                params['resultsFolder'],
                params['dataset'],
                params['dataFolder']):

    # Render
    print('Rendering LaTeX...')
    
    # Replace escape char
    data['questionIdict'] = ir.escapeLatexIdict(data['questionIdict'])
    data['ansIdict'] = ir.escapeLatexIdict(data['ansIdict'])

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    ir.renderLatex(
                inputTestSel, 
                targetTestSel, 
                data['questionIdict'], 
                data['ansIdict'],
                urlDict, 
                topK=params['topK'],
                outputFolder=params['outputFolder'],
                pictureFolder=params['pictureFolder'],
                comments=selComments,
                caption=caption,
                modelOutputs=modelOutputs,
                modelNames=ir.getModelNames(params['models']),
                questionIds=idx,
                filename=params['outputFilename'] + '.tex')