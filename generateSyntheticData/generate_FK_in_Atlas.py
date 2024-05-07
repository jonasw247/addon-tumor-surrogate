#%%
from TumorGrowthToolkit.FK import Solver as FKSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib

import multiprocessing
import json

def getOrigin(gm_data, wm_data):
    #check if origin is in the white or gray matter
    for i in range(300000):
        origin = np.random.rand(3)
        intOrigin = (origin * gm_data.shape).astype(int)
        if gm_data[intOrigin[0], intOrigin[1], intOrigin[2]] == 0 and wm_data[intOrigin[0], intOrigin[1], intOrigin[2]]== 0:
            print("Origin is in the background")
        else:
            print("Origin is in the brain")
            return origin
    #throw error
    raise Exception("Could not find origin in the brain")

def simulateOneSampleTumor(savePath):
    atlasPath = "../DTIAtlas/sub-mni152_tissues_space-sri.nii.gz"
    atlasTissue = nib.load(atlasPath).get_fdata()
    affine = nib.load(atlasPath).affine

    wm_data = atlasTissue == 3
    gm_data = atlasTissue == 2

    origin = getOrigin(gm_data, wm_data)

    randomRanges = {
        'Dw_range': [0.001, 10.0],
        'rho_range': [0.001, 10.0],
        'stopping_volume_range': [1000, 400000],#based on the brats dataset tumors range till 250ml so lets take 400ml as max
        }

    stopping_volume = np.random.randint(randomRanges['stopping_volume_range'][0], randomRanges['stopping_volume_range'][1]) 
    dw = np.random.uniform(randomRanges['Dw_range'][0], randomRanges['Dw_range'][1])
    rho = np.random.uniform(randomRanges['rho_range'][0], randomRanges['rho_range'][1])

    # Set up parameters
    parameters = {
        'Dw': dw,          # Diffusion coefficient for white matter
        'rho': rho,         # Proliferation rate
        'RatioDw_Dg': 10,  # Ratio of diffusion coefficients in white and grey matter
        'gm': gm_data,      # Grey matter data
        'wm': wm_data,      # White matter data
        'NxT1_pct': origin[0],    # tumor position [%]
        'NyT1_pct': origin[1],
        'NzT1_pct': origin[2],
        'init_scale': 1., #scale of the initial gaussian
        'resolution_factor': 1, #resultion scaling for calculations
        'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
        'verbose': True, #printing timesteps 
        'time_series_solution_Nt': None,#64, #64, # number of timesteps in the output
        'stopping_volume' : stopping_volume, #stop when the volume of the tumor is less than this value
        #'stopping_time' : 100000, #stop when the time is greater than this value
    }        

    # Run the FK_solver and plot the results
    start_time = time.time()
    fk_solver = FKSolver(parameters)
    result = fk_solver.solve()
    end_time = time.time()  # Store the end time
    execution_time = int(end_time - start_time)  # Calculate the difference

    print(f"Execution Time: {execution_time} seconds")
    if result['success']:
        print("Simulation successful!")
        #plot_tumor_states(wm_data, result['initial_state'], result['final_state'], NzT)
    else:
        print("Error occurred:", result['error'])


    nii = nib.Nifti1Image(result['final_state'], affine)

    os.makedirs(savePath, exist_ok=True)
    nib.save(nii, savePath + 'tumor_concentration.nii.gz')

    del result['final_state']
    del result['initial_state']
    del parameters['gm']
    del parameters['wm']
    saveDict = {
        "parameters": parameters,
        "results": result,
        "execution_time": execution_time,
        "randomRanges": randomRanges
    }


    np.save(savePath + 'saveDict.npy', saveDict)
    #save as json
    with open(savePath + 'saveDict.json', 'w') as fp:
        json.dump(saveDict, fp)


"""for i in range(10):

    simulateOneSampleTumor("/mnt/8tb_slot8/jonas/workingDirDatasets/synthetic_FK_Michals_solver/patient_" + str(i) + "/")

"""
# %%

def process_patient(i):
    seed = (os.getpid() + int(time.time() *10000)) % 2**30  # Using the process ID as a seed and also time
    np.random.seed(seed)
    path = "/mnt/8tb_slot8/jonas/workingDirDatasets/synthetic_FK_Michals_solver/patient_" + ("000000000"  + str(i))[-7:] + "/"
    simulateOneSampleTumor(path)


if __name__ == '__main__':
    number_of_patients = 10000
    number_of_processes = 10
    # Number of processes

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=number_of_processes)
    # Range of patients
    patient_indices = range(number_of_patients)

    # Map process_patient function over the range of patient indices
    pool.map(process_patient, patient_indices)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

# %%
