# Description: Functions for inference

def find_checkpoint_with_lowest_val_mse(base_folder):
    import os
    checkpoint = None
    # get previous 5 characters before ckpt
    files = os.listdir(base_folder)
    for file in files:
        if 'last' not in file:
            if file.endswith('.ckpt'):
                val_mse = file.split('=')[-1].split('.ckpt')[0]
                val_mse = val_mse.strip('-')[0]
                # convert string to float
                val_mse = float(val_mse)
                if checkpoint is None:
                    checkpoint = file
                    min_val_mse = float(val_mse)
                else:
                    if float(val_mse) < min_val_mse:
                        checkpoint = file
                        min_val_mse = float(val_mse)
                    elif float(val_mse) == min_val_mse:
                        # print('val_mse is the same, taking the one with the highest step.')
                        # get the one with the highest step
                        step = file.split('step=')[-1].split('-val_mse')[0]
                        if int(step) > int(checkpoint.split('step=')[-1].split('-val_mse')[0]):
                            checkpoint = file
                            min_val_mse = float(val_mse)
    return base_folder + checkpoint
