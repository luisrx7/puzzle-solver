'''
Helper (but not required) methods for the solver
'''

import os
import cv2
import random

def show (img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def verify_dory(pieceList):
    print "begin counting"
    count_wrong = 0
    for piece in pieceList:
        sexes = []
        for edge in piece.edges:
            sexes.append(edge.sex)
        real_sexes = get_real_sexes(piece.pieceNumber)

        for i in range(len(sexes)):
            if (sexes[i] != real_sexes[i]):
                count_wrong += 1
                print piece
    print 'count wrong', count_wrong


def get_real_sexes (pieceNumber):
    switcher = {
        1: ['n','f','f','n'],
        2: ['m','f','m','n'],
        3: ['f','f','m','n'],
        4: ['f','m','f','n'],
        5: ['m','f','m','n'],
        6: ['f','m','n','n'],
        7: ['n','f','m','m'],
        8: ['f','m','m','m'],
        9: ['f','m','m','m'],
        10: ['f','m','f','f'],
        11: ['m','f','f','m'],
        12: ['m','m','n','f'],
        13: ['n','m','f','m'],
        14: ['m','f','f','f'],
        15: ['m','m','f','f'],
        16: ['m','f','m','f'],
        17: ['f','m','f','m'],
        18: ['m','f','n','f'],
        19: ['n','n','m','f'],
        20: ['f','n','f','m'],
        21: ['m','n','f','f'],
        22: ['m','n','m','m'],
        23: ['f','n','f','f'],
        24: ['m','n','n','m']
    }
    return switcher.get(pieceNumber, "Invalid piece number")


def jumble_pieces(puzzleName):
    inputFolder = "input/" + puzzleName + '/'  # rename to 'path'? since it includes folders and files
    files = sorted(os.listdir(inputFolder))
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    if 'global' in files:
        files.remove('global')
    if 'all' in files:
        files.remove('all')
    alist = range(len(files))
    newList = []

    while (len(alist) > 0):
        i = random.randint(0,len(alist)-1)
        p = alist.pop(i)
        newList.append(p)

    count = "a"
    for j in range(len(newList)):
        k = files[newList[j]]
        img = cv2.imread("input/" + puzzleName + "/" + k)

        # Rotate it a random number of degrees
        rows, cols, color = img.shape
        degrees = random.randint(0,360)
        M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        # Write the file and increment the loop variable
        cv2.imwrite("input/" + puzzleName + "-jumble/" + count + ".jpg", img)
        count = chr(ord(count) + 1)
