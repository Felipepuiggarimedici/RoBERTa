'''
Code by Yinfei Yang, PhD candidate Imperial College London, co-supervisor of thesis.
'''
import numpy as np
curr_float = np.float32
curr_int = np.int16
from numba import prange
import json

def get_aa_count(vocab_path="tokenizer/vocab.json"):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    special_tokens = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}
    amino_acids = [token for token in vocab if token not in special_tokens]
    return len(amino_acids)

q = get_aa_count()

def convert_number(peptides, vocab_path="tokenizer/vocab.json"):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Define special tokens to ignore for mapping amino acids
    special_tokens = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}

    # Keep only tokens that are not special tokens and are single-letter (assumed AA)
    amino_acids = [token for token in vocab if token not in special_tokens]

    # Map amino acids to zero-based indices (compact)
    aa_to_idx = {aa: i for i, aa in enumerate(sorted(amino_acids))}

    converted = []
    for pep in peptides:
        numeric = []
        for aa in pep:
            if aa not in aa_to_idx:
                raise ValueError(f"Unknown amino acid '{aa}' in peptide '{pep}'")
            numeric.append(aa_to_idx[aa])
        converted.append(numeric)

    return np.array(converted, dtype=np.int16)



def calculate_error_3(data_tr, data_gen, L, q = q, ifprint=False, ifmat=False):
    # Means
    if data_tr.dtype != 'int16':
        data_tr = convert_number(list(data_tr))
    if data_gen.dtype != 'int16':
        data_gen = convert_number(list(data_gen))
    
    mudata, fdata, f3data, covdata, cov3data = calculate_mat_3(data_tr, L, q)
    mugen, fgen, f3gen, covgen, cov3gen = calculate_mat_3(data_gen, L, q)

    M = len(data_tr)
    maxp = float(1)/float(M)
    ps = 0.001

    # single site frequency
    errm = 0
    neffm = 0
    for i in range(L):
        for a in range(q):
            neffm += 1
            if mudata[i,a] < maxp:
                errm += np.power((mugen[i,a] - mudata[i,a]),2)/(float(1-maxp/q)*float(maxp/q))
            elif mudata[i,a] == 1.0:
                errm += np.power((mugen[i,a] - mudata[i,a]),2)/(float(1-mudata[i,a]+ps)*float(mudata[i,a]-ps))  
            else:
                errm += np.power((mugen[i,a] - mudata[i,a]),2)/(float(1-mudata[i,a])*float(mudata[i,a])) 
            if mudata[i,a] < maxp:
                denom = (1-maxp/q)*(maxp/q)
            else:
                denom = (1-mudata[i,a])*(mudata[i,a])
            if denom < 1e-6:
                print("Tiny denom at i,a:", i, a, "→", denom)
    errmt = np.sqrt(float(errm)/(float(neffm)*float(maxp)))

    # 2-sites frequency
    errm2 = 0
    neffm2 = 0
    for i in range(L):
        for j in range(i+1,L):
            for a in range(q):
                for b in range(q):
                    neffm2+=1
                    if fdata[i,j,a,b] < maxp:
                        den = float(1-maxp/q**2)*float(maxp/q**2)
                    else:
                        den = float(1-fdata[i,j,a,b])*float(fdata[i,j,a,b])
                    errm2 += np.power((fgen[i,j,a,b] - fdata[i,j,a,b]),2)/float(den)
    errmt2 = np.sqrt(float(errm2)/(float(neffm2)*float(maxp)))

    # 3-sites frequency
    errm3 = 0
    neffm3 = 0
    for i in range(L):
        for j in range(i+1,L):
            for k in range(j+1,L):
                for a in range(q):
                    for b in range(q):
                        for c in range(q):
                            neffm3+=1
                            if f3data[i,j,k,a,b,c] < maxp:
                                den = float(1-maxp/q**3)*float(maxp/q**3)
                                errm3 += np.power((f3gen[i,j,k,a,b,c] - f3data[i,j,k,a,b,c]),2)/float(den)
                            else:
                                den = float(1-f3data[i,j,k,a,b,c])*float(f3data[i,j,k,a,b,c])
                                errm3 += np.power((f3gen[i,j,k,a,b,c] - f3data[i,j,k,a,b,c]),2)/float(den)
    errmt3 = np.sqrt(float(errm3)/(float(neffm3)*float(maxp)))
    
    # 2-sites correlations
    errc = 0
    neffc = 0
    for i in range(L):
        for j in range(i+1,L):
            for a in range(q):
                for b in range(q):
                    neffc += 1
                    if covdata[i,j,a,b] < maxp:
                        var0 = np.sqrt(float(1-maxp/q**2)*float(maxp/q**2))
                    else:
                        var0 = np.sqrt(float(1-fdata[i,j,a,b])*float(fdata[i,j,a,b]))
                    var1 = mudata[i,a] * np.sqrt(mudata[j,b] * (1 - mudata[j,b])) + mudata[j,b] * np.sqrt(mudata[i,a] * (1 - mudata[i,a]))
                    den = np.power(var0 + var1, 2)
                    errc += np.power((covgen[i,j,a,b] - covdata[i,j,a,b]),2)/float(den)
    errct = np.sqrt(float(errc)/(float(neffc)*float(maxp)))

    # 3-sites correlations
    errc2 = 0
    neffc2 = 0
    for i in range(L):
        for j in range(i+1,L):
            for k in range(j+1,L):
                for a in range(q):
                    for b in range(q):
                        for c in range(q):
                            neffc2 += 1
                            if f3data[i,j,k,a,b,c] < maxp:
                                var0 = np.sqrt(float(1-maxp/q**3)*float(maxp/q**3))
                            else:
                                var0 = np.sqrt(float(1-f3data[i,j,k,a,b,c])*float(f3data[i,j,k,a,b,c]))
                            var1 = mudata[i,a]*np.sqrt(fdata[j,k,b,c]*(1 - fdata[j,k,b,c])) + mudata[j,b]*np.sqrt(fdata[i,k,a,c]*(1 - fdata[i,k,a,c])) + mudata[k,c]*np.sqrt(fdata[i,j,b,c] * (1 - fdata[i,j,b,c]))
                            var2 = mudata[i,a]*mudata[j,b]*np.sqrt(mudata[k,c]*(1 - mudata[k,c])) + mudata[i,a]*mudata[k,c]*np.sqrt(mudata[j,b]*(1 - mudata[j,b])) + mudata[j,b]*mudata[k,c]*np.sqrt(mudata[i,a]*(1 - mudata[i,a]))
                            den = np.power(var0 + var1 + 4*var2, 2)
                            errc2 += np.power((cov3gen[i,j,k,a,b,c] - cov3data[i,j,k,a,b,c]),2)/float(den)
    errct2 = np.sqrt(float(errc2)/float(neffc2)*float(maxp))

    if ifprint:
        print(f'The error on frequency is {errmt} (single site), {errmt2} (2 sites) and {errmt3} (3 sites)')
        print(f'the error on correlations is {errct} (2 sites), {errct2} (3 sites)')
    if ifmat:
        return [errmt, errmt2, errmt3, errct, errct2, mugen, mudata, fgen, fdata, f3gen, f3data, covgen, covdata, cov3gen, cov3data]
    else:
        return [errmt, errmt2, errmt3, errct, errct2]

def calculate_mat_3(data_tr, L, q):
    # Means
    if data_tr.dtype != 'int16':
        data_tr = convert_number(list(data_tr))
    
    mudata = average(data_tr, q) # empirical averages
    
    # training data
    fdata = average_product(data_tr, data_tr, q)  # fij 9 x 9 x 20 x 20
    covdata = fdata - mudata[:,np.newaxis,:,np.newaxis] * mudata[np.newaxis,:,np.newaxis,:]
    muf1data = mudata[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * fdata[np.newaxis,:,:,np.newaxis,:,:]
    muf2data = mudata[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] * fdata[:,np.newaxis,:,:,np.newaxis,:]
    muf3data = mudata[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :] * fdata[:,:,np.newaxis,:,:,np.newaxis]
    mu3data = mudata[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * mudata[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] * mudata[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    f3data = frequency_3_site(data_tr, q)
    cov3data = f3data - muf1data - muf2data - muf3data + 2*mu3data

    # put to zero the diagonal elements of the covariance
    for i in range(L):
        covdata[i,i,:,:] = np.zeros((q,q))
        fdata[i,i,:,:] = np.zeros((q,q))
        cov3data[i,i,:,:,:,:] = np.zeros([q,q,q])
        cov3data[i,:,i,:,:,:] = np.zeros([q,q,q])
        cov3data[:,i,i,:,:,:] = np.zeros([q,q,q])

    return mudata, fdata, f3data, covdata, cov3data

def frequency_1_site(X, q=20):
    N = X.shape[0]
    L = X.shape[1]
    out = np.zeros((L, q), dtype=curr_float)
    for i in range(L):
        for n in range(N):
            out[i, X[n,i]] += 1
    out /= N
    return out

def frequency_2_site(X, q=20):
    N = X.shape[0]
    L = X.shape[1]
    out = np.zeros((L, L, q, q), dtype=curr_float)
    for i in range(L):
        for j in range(L):
            for n in range(N):
                out[i, j, X[n,i], X[n,j]] += 1
    out /= N
    for i in range(L):
        out[i,i,:,:] = np.zeros((q,q))
    return out

def frequency_3_site(X, q=20):
    N = X.shape[0]
    L = X.shape[1]
    out = np.zeros((L, L, L, q, q, q), dtype=curr_float)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for n in range(N):
                    out[i, j, k, X[n,i], X[n,j], X[n,k]] += 1
    out /= N
    for i in range(L):
        out[i,i,:,:,:,:] = np.zeros([q,q,q])
        out[i,:,i,:,:,:] = np.zeros([q,q,q])
        out[:,i,i,:,:,:] = np.zeros([q,q,q])
    return out

def covariance_3(X, L, q):
    mu1 = frequency_1_site(X, c=q)
    f2 = frequency_2_site(X,X,c1=q,c2=q)
    muf1 = mu1[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * f2[np.newaxis,:,:,np.newaxis,:,:]
    muf2 = mu1[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] * f2[:,np.newaxis,:,:,np.newaxis,:]
    muf3 = mu1[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :] * f2[:,:,np.newaxis,:,:,np.newaxis]
    mu3 = mu1[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * mu1[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis] * mu1[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    f3 = frequency_3_site(X, q)
    cov3 = f3 - muf1 - muf2 - muf3 + 2*mu3
    for i in range(L):
        cov3[i,i,:,:,:,:] = np.zeros([q,q,q])
        cov3[i,:,i,:,:,:] = np.zeros([q,q,q])
        cov3[:,i,i,:,:,:] = np.zeros([q,q,q])
    return cov3

def get_upper_flatten(data):
    L = data.shape[0]
    if len(data.shape) == 2:
        return data.flatten()
    elif len(data.shape) == 4:
        return data[np.triu_indices(data.shape[0], k=1)].flatten()
    elif len(data.shape) == 6:
        mask = np.zeros(data.shape[:3], dtype=bool)
        for i in range(L):
            for j in range(i + 1, L):
                for k in range(j + 1, L):
                    mask[i, j, k] = True
        return data[mask].flatten()

def average(config, q):
    B = config.shape[0]
    N = config.shape[1]
    out = np.zeros((N, q), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            idx = config[b, n]
            if 0 <= idx < q:
                out[n, idx] += 1
            else:
                raise ValueError(f"Invalid index {idx} in config at position ({b}, {n}). Expected range: 0 ≤ idx < {q}")
    out /= B
    return out

def average_product(config1, config2, q):
    B = config1.shape[0]
    M = config1.shape[1]
    N = config2.shape[1]
    out = np.zeros((M, N, q, q), dtype=curr_float)
    for b in prange(B):
        for m in prange(M):
            for n in prange(N):
                a = config1[b, m]
                b_ = config2[b, n]
                if 0 <= a < q and 0 <= b_ < q:
                    out[m, n, a, b_] += 1
                else:
                    raise ValueError(f"Invalid indices a={a}, b={b_} at (b={b}, m={m}, n={n}). Expected: 0 ≤ a,b < {q}")
    out /= B
    return out
