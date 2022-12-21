import numpy as np

def symmetry_maker(o):
    symmetry = np.zeros(13)

    symmetry[0] = float(o[0])
    symmetry[1] = float(o[1])
    symmetry[2] = float(o[2])

    symmetry[3] = float(o[4])
    symmetry[4] = float(o[3])

    symmetry[5] = float(o[6])
    symmetry[6] = float(o[5])

    symmetry[7] = float(o[8])
    symmetry[8] = float(o[7])

    symmetry[9] = float(o[10])
    symmetry[10] = float(o[9])

    symmetry[11] = float(o[12])
    symmetry[12] = float(o[11])

    o_sym = symmetry
    return o_sym