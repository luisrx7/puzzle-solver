import os
import cv2
import sys
import numpy as np
from edge import Edge
from piece import Piece

USE_MASKED_IMAGE = True

''' Assemble puzzle one piece at a time '''
def assemble_puzzle_sequentially (puzzleName):
    tm = cv2.TickMeter(); tm2 = cv2.TickMeter();
    tm2.start();
    # 1) Get the list of puzzle pieces
    inputFolder = "input/" + puzzleName + '/'
    files = sorted(os.listdir(inputFolder))
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    pieceList = []

    for afile in files:
        # 2) Pull out one piece at a time and analyze it
        tm.start();
        newPiece = Piece(puzzleName, afile)
        pieceList.append(newPiece)
        pieceList[0].isPinned = True  # Always pin the first piece
        tm.stop(); print "analysis", tm.getTimeMilli();

        tm.reset(); tm.start();
        # 3) Rebuild the puzzle
        for piece in pieceList:
            pinDirection(piece, pieceList, "right")
            pinDirection(piece, pieceList, "below")

        # 4) Draw after every new file
        draw_matrix_puzzle(puzzleName, pieceList)
        tm.stop(); print "pinned and drawn", tm.getTimeMilli(), "\n";

    tm2.stop(); print "total", tm2.getTimeSec(), "seconds";
    print "total", tm2.getTimeSec()/60.0, "minutes"


''' Pin an individual puzzle piece '''
def pinDirection (setPiece, pieceList, direction):
    if (direction == "right"):
        (i, j) = (2, 0)  # setPiece's right and testPiece's left
    elif (direction == "below"):
        (i, j) = (1, 3)  # setPiece's bottom and testPiece's top
    else:
        return

    if (setPiece.edges[i].isConnected is not True):
        for testPiece in pieceList:
            if ( (setPiece.pieceNumber != testPiece.pieceNumber) and
               (testPiece.isPinned is not True) and
               (setPiece.edges[i].does_match_edge(testPiece.edges[j])) ):
                # Set connections
                connect_pieces(setPiece, i, testPiece, j)
                testPiece.isPinned = True

                # Update its location within the puzzle
                (testPiece.x, testPiece.y) = (setPiece.x, setPiece.y)
                if (direction == "right"):
                    testPiece.x += 1   # Double check x-y conventions
                elif (direction == "below"):
                    testPiece.y += 1

                # Connect to neighboring pieces so edges aren't left dangling open
                connect_in_direction (testPiece, pieceList, -1, 0, 2, 0)  # Look left
                connect_in_direction (testPiece, pieceList,  0, 1, 3, 1)  # Look below
                connect_in_direction (testPiece, pieceList,  1, 0, 0, 2)  # Look right
                connect_in_direction (testPiece, pieceList,  0,-1, 1, 3)  # Look above

                # Debug print statements
                print "success-" + direction
                print "setPiece  " + str(setPiece)
                print "testPiece " + str(testPiece)
                break


def connect_in_direction (testPiece, pieceList, xOffset, yOffset, edgeA, edgeB):
    (xprime, yprime) = testPiece.x + xOffset, testPiece.y + yOffset
    for p in pieceList:
        if (p.x == xprime and p.y == yprime):
            connect_pieces(p, edgeA, testPiece, edgeB)
            break  # why is this necessary again?


def connect_pieces (pieceA, positionA, pieceB, positionB):
    pieceA.edges[positionA].isConnected = True
    pieceA.edges[positionA].pieceThisEdgeConnectsTo = pieceB
    pieceB.edges[positionB].isConnected = True
    pieceB.edges[positionB].pieceThisEdgeConnectsTo = pieceA
    pieceB.isPinned = True


def draw_matrix_puzzle(puzzleName, pieceList):
    # Find the dimensions of a single puzzle piece
    inputFolder = "input/" + puzzleName
    files = sorted(os.listdir(inputFolder))
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    img = cv2.imread("input/" + puzzleName + "/" + files[0], cv2.IMREAD_REDUCED_COLOR_8)
    (height, width) = img.shape[:2]

    # Set the dimensions of the entire canvas
    puzzleDimensions = (4, 6)  # Hardcoded for now
    canvas = np.zeros((puzzleDimensions[0] * width, puzzleDimensions[1] * height, 3))

    for piece in pieceList:
        # Read file, resize it, and then get the piece's location
        if USE_MASKED_IMAGE is True:
            readFile = 'output' + '/' + piece.puzzleName + '/' + piece.pieceName + "-mask.jpg"
        else:
            readFile = inputFolder + '/' + str(piece.pieceName) + ".jpg"
        img = cv2.imread(readFile, cv2.IMREAD_REDUCED_COLOR_8)
        resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
        (xcount, ycount) = (piece.x, piece.y)

        if USE_MASKED_IMAGE is True:
            # Draw the images overlapped
            x_start = int(width * 0.5 * xcount)
            x_end = x_start + width
            y_start = int(height * 0.5 * ycount)
            y_end = y_start + height
            canvas[y_start:y_end, x_start:x_end] += resized[:,:]
        else:
            # Draw the images side-by-side
            canvas[height*ycount:height*(ycount+1), width*xcount:width*(xcount+1)] = resized[:,:]

    # Looks better on white background
    if len(pieceList) > 23: # Only paint white at the very end because it takes a while
        h, w, c = canvas.shape
        for i in range(h):
            for j in range(w):
                if (canvas[i][j][0] == 0 and canvas[i][j][1] == 0 and canvas[i][j][2] == 0):
                    canvas[i][j] = [255, 255, 255]

    cv2.imwrite('output/'+ puzzleName + "/canvas/" + str(len(pieceList)) + '.jpg', canvas)


if __name__ == "__main__":
    puzzleName = sys.argv[1]
    if (puzzleName == "dory-single"):
        Piece(puzzleName, "e.jpg")
        Piece(puzzleName, "f.jpg")
        Piece(puzzleName, "u.jpg")
        Piece(puzzleName, "v.jpg")
    else:
        assemble_puzzle_sequentially(puzzleName)
