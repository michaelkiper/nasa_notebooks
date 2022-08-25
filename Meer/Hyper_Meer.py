from re import M
import numpy as np
from PIL import Image
from matplotlib import image
import os

def HyperMeer(img):
    if type(img) == Image:
        data = np.asarray(img)
        return HyperMeerEngineThread(data)

    elif type(img) == np.ndarray:
        return HyperMeerEngineThread(img)

    elif type(img) == str:
            if not os.path.exists(str):
                raise Exception("File does not exist")
            try:
                img = Image.open(img)
            except Exception:
                try:
                    img = image.imread(img)
                except Exception as e:
                    raise e
            
            data = np.asarray(img)
            return HyperMeerEngineThread(data)
        

import concurrent
def HyperMeerEngineThread(img:np.array):
    step = 25
    id_series = []
    with concurrent.futures.ThreadPoolExecutor() as executer:
        # rfls = [(i, rfl) for i, rfl in enumerate(img)]
        
        rfls = []
        index = 0
        for r in range(0, img.shape[-1], step):
            rfls.append((index, img[:, :, r:r+step]))
            index += 1
        print("Total Runs:", index, "\n")

        results = executer.map(HyperMeerEngineThreadExec, rfls)

        for result in results:
            id_series.append(result)
            
        s = sorted(id_series, key=lambda x: x[0])
        r = [j for i, j in s]
        r_noise = [noise for noise, level in r]
        total = []
        for n in r_noise:
            for n_i in n:
                total.append(n_i)
        return total
    
def HyperMeerEngineThreadExec(s):
    i, img = s[0], s[1]
    print("Starting run number:", i)
    return (i, HyperMeerEngine(img))


def HyperMeerEngine(img:np.array):
    T = -0.01
    bands = img.shape[-1] #the last element in the image should be the bands
    v = np.zeros(bands)
    L = np.zeros(bands)

    for loop in range(bands):
        try:
            v[loop], L[loop] = Meer(img, img[:,:,loop:loop+1], T, loop+1)
        except Exception:
            raise Exception
            v[loop] = 0
            L[loop] = 0
    
    return v, L

def Meer(full_image, band, T, index):
    # currently only works on band sizes of 1
    if len(band.shape) != 3:
        raise Exception("Improper 'band' shape: ", band)
    if band.shape[2] != 1:
        raise Exception("Only works for single bands. Currently inputted: ",band.shape[2], "bands")
    # if band.shape[0]<=19 or band.shape[1]<=19:
    #     raise Exception("Too small of a band size")
    
    full_image = np.array(full_image)
    
    x, y, b = full_image.shape
    m = min(x, y)

    L = 0
    left = m
    while left >= 2:
        L += 1
        left = left//2

    noise = 0
    level = 0
    br = False

    V = []
    for loop in range(1, L-1+1):
        cell_size = (2**L) / (2**(L-loop))
        next = 0
        v = []

        for inner in range(1, int((2**L)/cell_size+1)): 
            start = (inner-1) * cell_size + 1

            for inner1  in range(1, int((2**L)/cell_size+1)):
                start1 = (inner1-1) * cell_size + 1

                # cell = full_image[int(start-1):int(start+cell_size-1), int(start1-1):int(start1+cell_size-1), 0]
                cell = band[int(start-1):int(start+cell_size-1), int(start1-1):int(start1+cell_size-1), 0]

                m = np.nanmean(np.nanmean(cell))

                var = 1 / (4**loop-1) * np.nansum(np.nansum((cell-m)**2))
                v.append(var)

                next += 1

        q = sorted(v)

        r1 = (q[2-1]-q[1-1])/(q[4-1]-q[1-1])
        r2 = (q[3-1]-q[2-1])/(q[4-1]-q[2-1])
        r3 = 1-r2
        
        
        if r1<=0.5 or loop==L-1:
            V.append(1/4*np.nansum(q[0:4]))
        elif r2<=0.7:
            V.append(1/3*np.nansum(q[1:4]))
        elif r3<=0.7:
            V.append(1/2*np.nansum(q[2:4]))
        else: 
            V.append(q[4-1])
        
    br = False
    if not br:
        m = np.nanmean(np.nanmean(band))
        var = 1 / ((4**L)-1) * np.nansum(np.nansum((band-m)**2))
        V.append(var)

        B = []
        for loop in range(1, L):
            r_item = V[loop-1] / V[loop] - (1 - 0.1 * 2**(-loop+5))
            B.append(r_item)
    
        B = np.array(B)
        if len(B[3-1:])>0 and np.nanmin(B[3-1:]) > 0:
            noise = V[L-1]
            level = L
        else:
            lt = [i for i in range(len(B[2:])) if B[2:][i] < 0]
            l_u = lt[0] + 2 + 1

            l=l_u
            
            while (np.nansum(B[l_u-1:l]) >= -0.1)  or l == L: # check to make sure that this is right
                l += 1

            l_0=l
            
            if l_0==L:
                noise = V[L-1]
                level = L
                print('Rule 2')

            if l_0==3 or l_0==4:
                noise = V[3-1]
                level = 1
                print('Cannot determine noise')
                print(noise)

            elif l_0 == 5:
                rho = B[5-1] + B[6-1]

                if rho<=-1.5 and rho>-2:
                    noise = V[3-1]
                    level = 3
                    print('Rule 5_1')

                elif rho <= -1 and rho >= -1.5:
                    delta = (rho + 2)/-0.5
                    noise = delta * V[3-1] + (1-delta) * V[4-1]
                    level = 3.5
                    print('Rule 5_2')

                elif rho<=-0.5 and rho>=-1 :
                    delta = (rho + 0.5) / -0.5
                    noise = delta * V[4-1] + (1-delta) * V[5-1]
                    level = 4.5
                    print('Rule 5_3')

                elif rho <= T and rho >= -0.5 :
                    delta = (rho - T)/(-0.5 - T)
                    noise = delta * V[5-1] + (1-delta) * V[6-1]
                    level = 5.5
                    print('Rule 5_4')
                else :
                    print('Error at 5')

            elif l_0 == 6 or l_0 == 7:
                if B[l_0-1-1] <= 0 and B[l_0-1-1] > T :
                    delta = abs(B[l_0-1])
                    noise = delta * V[l_0-2-1] + (1-delta) * V[l_0-1-1]
                    level = 5
                    print('Rule 6_1')

                elif B[l_0-1-1] < 1 and B[l_0-1-1] >= 0 :
                    if B[l_0-1] <= -0.5 and B[l_0-1] > -1 :
                        delta = (B[l_0-1] + 0.5) / -0.5
                        noise = delta * V[l_0-2-1] + (1-delta) * V[l_0-1-1]
                        level = 5
                        print('Rule 6_2')

                    elif B[l_0-1] <= T and B[l_0-1] > -0.5 :
                        delta = (B[l_0-1] - T) / (-0.5 - T)
                        noise = 0.5*((1+delta) * V[l_0+1-1] + (1-delta) * V[l_0-1])
                        level = 7
                        print('Rule 6_3')
                    else:
                        print('Error at 6_2 or 6_3')
                else:
                    print('Error at 6_1')

    return noise, level
