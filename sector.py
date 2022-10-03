import os
import cv2

class Sector:
    ''' Rectangular/square area of the empty puzzle board '''

    def __init__ (self, hasPiece=False, addPiece=None):
        self.hasPiece = hasPiece  # Has a pinned piece?
        self.pieces = []
        if addPiece is not None:
            self.pieces.append(addPiece)
            hasPiece = True

    def __str__(self):
        line = str(self.hasPiece)
        for p in self.pieces:
            line += '\n' + str(p)
        return line
