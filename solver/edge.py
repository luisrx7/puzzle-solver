import cv2
import numpy as np

class Edge:
    ''' One edge in a piece '''

    def __init__ (self, imageFile, rotation, belongingPiece):
        self.imageFile = imageFile  # Path to image file of this edge
        self.sex = self.determine_sex(imageFile, rotation)
        if (self.sex == "n"):
            self.isConnected = True
            # Flat edges are on the perimeter and do not match any other edges.
            # So they are 'already connected' and you do not need to match them.
        else:
            self.isConnected = False
        self.pieceThisEdgeBelongsTo = belongingPiece
        self.pieceThisEdgeConnectsTo = None

    def __str__(self):
        return self.sex

    def determine_sex (self, fileName, degrees):
        # Open file and find the edges
        image = cv2.imread(fileName)
        if image is None:
            return 'n'  # If trying to read an empty file
        image = cv2.bilateralFilter(image, 9, 150, 150)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, 100, 200)

        # Rotate so the center of the piece is on the bottom
        if (degrees != 0):
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
            image = cv2.warpAffine(image, M, (cols, rows))

        # Crop the extra from when it was rotated
        m, n = image.shape
        image = image[int(m*0.15):m, :]  # Modify later?

        # Convert the 2D edge into a one-dimensional line
        m, n = image.shape
        line = []
        for j in range (0, n, 10):  # For each row. Sample, don't need every one.
            for i in range (0, m, 1):  # Go down the column
                if (image[i][j] != 0):
                    line.append(i)  # Record the first time it is not 0 (you reach the edge)
                    break
        # https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Matrix.svg/1200px-Matrix.svg.png

        # Compare mean of first third to second third
        n = len(line)
        n1, n2 = (n/3, n*2/3)
        firstThirdMean = np.mean(line[0:n1])
        secondThirdMean = np.mean(line[n1:n2])
        lineDeviation = np.std(line)

        if (lineDeviation < 50): # First determine if line is flat. This nubmer is experimentally determined.
            return "n"
        else:
            if (secondThirdMean < firstThirdMean):
                return "m"  # Piece is closer to the image axis, so it 'sticks out'
            else:
                return "f"  # Piece is farther from the image axis, so it is an 'indentation'


    ''' Determines if two edges fit together based on their sex and shape. '''
    def does_match_edge (self, testEdge):
        if (self.sex == 'n' or testEdge.sex == 'n'):
            return False  # Flat edges are on the perimeter and don't match anything
        if (self.sex == testEdge.sex):
            return False  # Same-sex puzzle pieces don't match
        return True  # For now, assume all opposite sex edges match.
        # Later, match by color or shape (contour hu value)
