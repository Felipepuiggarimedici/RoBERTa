import os
nHeads = [14,16,18,20]
nLayers = [1,2,3,4,5]
batch_sizes = [128]
hidden_sizes = [510, 768, 864, 940]
d_ffs =[2048, 3072, 4096]
mlmProbs = [0.10, 0.20, 0.4, 0.6] 

for head in nHeads:
    for layer in nLayers:
        for b in batch_sizes:
            for mlmProb in mlmProbs:
                for hs in hidden_sizes:
                    for d_ff in d_ffs:
                        command = (
                            f"qsub -q v1_limited "
                            f"-v nHeads={head},nLayers={layer},mlmProb={mlmProb},batch_size={b},hidden_size={hs},d_ff={d_ff} "
                            f"run5FoldAntiberta.pbs"
                        )
                        print("Submitting:", command)
                        os.system(command)