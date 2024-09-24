# %%
import random as rnd

# %%
def _minHash(toHash, hashFunc):
    D = dict(zip(hashFunc, toHash))

    for i in range(len(hashFunc)):
        if D[i] == 1:
            return i

    return -1

# %%
def genHashFuncs(lenOfVector, hash_len=4):
    D = []

    for i in range(hash_len):
        h = [i for i in range(lenOfVector)]
        rnd.shuffle(h)

        D.append(h)

    return D

# %%
def minHash(vector:list, hashFuncs) -> list:
    return [_minHash(vector, h) for h in hashFuncs]



# %%
if __name__ == "__main__": 
    
    vector = [rnd.randint(0,1) for i in range(12)]

    hashFuncs = genHashFuncs( len(vector) )

    print( minHash(vector, hashFuncs) )


