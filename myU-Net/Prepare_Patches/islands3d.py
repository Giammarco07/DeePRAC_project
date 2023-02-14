from collections import deque
import numpy as np
#https://www.techiedelight.com/count-the-number-of-islands/
# Below lists detail all 26 possible movements from a cell
row = [-1, -1, -1, 0, 1, 0, 1, 1, -1, -1, -1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 1, 0, 1, 1, 0]
col = [-1, 1, 0, -1, -1, 1, 0, 1, -1, 1, 0, -1, -1, 1, 0, 1, 0, -1, 1, 0, -1, -1, 1, 0, 1, 0]
slc = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]



# Function to check if it is safe to go to position `(x, y, z)`
# from the current position. The function returns false if `(x, y, z)`
# is not valid matrix coordinates or `(x, y, z)` represents water or
# position `(x, y, z)` is already processed.

def isSafe(mat, x, y, z, processed):
    return (x >= 0) and (x < processed.shape[0]) and \
           (y >= 0) and (y < processed.shape[1]) and \
           (z >= 0) and (z < processed.shape[2]) and \
           (mat[x][y][z] == 1 and not processed[x][y][z])


def BFS(mat, processed, i, j, k):
    # create an empty queue and enqueue source node
    q = deque()
    q.append((i, j, k))

    # mark source node as processed
    processed[i][j][k] = True

    # loop till queue is empty
    while q:

        # dequeue front node and process it
        x, y, z = q.popleft()

        # check for all 26 possible movements from the current cell
        # and enqueue each valid movement
        for p in range(26):
            # skip if the location is invalid, or already processed, or has water
            if isSafe(mat, x + row[p], y + col[p], z + slc[p], processed):
                # skip if the location is invalid, or it is already
                # processed, or consists of water
                processed[x + row[p]][y + col[p]][z + slc[p]] = True
                q.append((x + row[p], y + col[p], z + slc[p]))


def islands3d(mat):

    (M, N, O) = (mat.shape[0],  mat.shape[1], mat.shape[2])

    # stores if a cell is processed or not
    processed = np.full((M, N, O), False)
    newmat = np.zeros((M,N,O))

    island = 0

    for i in range(M):
        for j in range(N):
            for k in range(O):
                # start BFS from each unprocessed node and increment island count
                if mat[i][j][k] == 1 and not processed[i][j][k]:
                    newprocessed, tmp = np.zeros((M, N, O)), np.zeros((M, N, O))
                    tmp[processed] = 1
                    BFS(mat, processed, i, j, k)
                    island = island + 1
                    newprocessed[processed] = 1
                    newprocessed -= tmp
                    newmat[newprocessed==1]=island
                    del tmp,newprocessed


    print("The total number of islands is", island)

    return newmat

def direction(p0,p1):
    a = p1[0] - p0[0]
    b = p1[1] - p0[1]
    c = p1[2] - p0[2]

    return a,b,c

def plane(p0,a,b,c):
    d = - a*p0[0] - b*p0[1] - c*p0[2]

    return d

def forcing(perp,dimension,alfa,beta,c0,d):
    return round((- d - (c0[dimension[0]]*alfa) - (c0[dimension[1]]*beta))/c0[perp])

def minimo(piano,shape):
    x = shape[0] - abs(piano[0].max() - piano[0].min())
    y = shape[1] - abs(piano[1].max() - piano[1].min())
    z = shape[2] - abs(piano[2].max() - piano[2].min())
    pos = np.array([x,y,z])
    return np.argmax(pos)

def end_bif_points_2d(skeleton):
    import scipy.ndimage as ndi

    selems = list()
    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
    selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))

    branches = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(skeleton, selem)

    selems_end = list()
    selems_end.append(np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]))
    selems_end.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
    selems_end  = [np.rot90(selems_end [i], k=j) for i in range(2) for j in range(4)]

    ends = np.zeros_like(skeleton, dtype=bool)
    for selem in selems_end:
        ends |= ndi.binary_hit_or_miss(skeleton, selem)

    return ends,branches