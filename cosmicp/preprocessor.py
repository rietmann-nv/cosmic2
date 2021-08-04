import sys
import os
import jax.numpy as np
import jax
import jax.ops
import jax.dlpack
import scipy
import scipy.constants
import scipy.interpolate
import scipy.signal
import numpy as npo
import math
from .fccd import imgXraw as cleanXraw
from .common import printd, printv, rank, gather, color, bcolors
from .common import  size as mpi_size
from .diskIO import IO, frames_out

from timeit import default_timer as timer


@jax.jit
def combine_double_exposure(data0, data1, double_exp_time_ratio, thres=3e3):

    msk=data0<thres    

    return (double_exp_time_ratio+1)*(data0*msk+data1)/(double_exp_time_ratio*msk+1)
@jax.jit
def resolution2frame_width(final_res, detector_distance, energy, detector_pixel_size, frame_width):

    hc=scipy.constants.Planck*scipy.constants.c/scipy.constants.elementary_charge
   
    wavelength = hc/energy
    padded_frame_width = (detector_distance*wavelength) /(detector_pixel_size*final_res)

    return padded_frame_width # cropped (TODO:or padded?) width of the raw clean frames

#Computes a weighted average of the coordinates, where if the image is stronger you have more weight.
@jax.jit
def center_of_mass(img, coord_array_1d):
    return np.array([np.sum(img*coord_array_1d)/np.sum(img), np.sum(img*coord_array_1d.T)/np.sum(img)])

@jax.jit
def filter_frame(frame, bbox):
    return jax.scipy.signal.convolve2d(frame, bbox, mode='same', boundary='fill')


#Interpolation around the center of mass, thus centering. This downsamples into the output frame width
@jax.partial(jax.jit, static_argnums=2)
def shift_rescale(img, center_of_mass, out_frame_shape, scale):

    img_out = jax.image.scale_and_translate(img, [out_frame_shape, out_frame_shape], [0,1], jax.numpy.array([scale, scale]), jax.numpy.array([center_of_mass[1], center_of_mass[0]]) , method = "bilinear", antialias = False).T

    img_out*=(img_out>0)

    img_out = np.float32(img_out)
            
    img_out = np.reshape(img_out, (1, out_frame_shape, out_frame_shape))

    return img_out

@jax.jit
def split_background(background_double_exp):

    # split the average from 2 exposures:
    bkg_avg0=np.average(background_double_exp[0::2],axis=0)
    bkg_avg1=np.average(background_double_exp[1::2],axis=0)

    return np.array([bkg_avg0, bkg_avg1])


def compute_background_metadata(metadata, frames, dark_frames):

    ## get one frame to compute center

    clean_frame = [] 
    if metadata["double_exposure"]:

        background_avg = split_background(dark_frames)
        for i in range(0, frames.shape[0], 2):

            frame_exp1 = frames[i] - background_avg[0]
            frame_exp2 = frames[i + 1] - background_avg[1]
            # get clean frames
            clean_frame.append(combine_double_exposure(cleanXraw(frame_exp1), cleanXraw(frame_exp2), metadata["double_exp_time_ratio"]))
            
    else:
        background_avg = np.average(dark_frames,axis=0)
        for i in range(0, frames.shape[0], 1):

            # get clean frames
            clean_frame.append(cleanXraw(frames[i] - background_avg))

    clean_frame = npo.array(clean_frame)

    metadata["frame_width"] = clean_frame.shape[0]

    #Coordinates from 0 to frame width, 1 dimension
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))
    yy=np.reshape(np.arange(metadata["output_frame_width"]),(metadata["output_frame_width"],1))

    # cropped width of the raw clean frames
    if metadata["desired_padded_input_frame_width"]:
        metadata["padded_frame_width"] = metadata["desired_padded_input_frame_width"]

    else:
        metadata["padded_frame_width"] = resolution2frame_width(metadata["final_res"], metadata["detector_distance"], metadata["energy"], metadata["detector_pixel_size"], metadata["frame_width"]) 
    
    # modify pixel size; the pixel size is rescaled
    metadata["x_pixel_size"] = metadata["detector_pixel_size"] * metadata["padded_frame_width"] / metadata["output_frame_width"]
    metadata["y_pixel_size"] = metadata["x_pixel_size"]

    # frame corner
    corner_x = metadata['x_pixel_size']*metadata['output_frame_width']/2  
    corner_z = metadata['detector_distance']                
    metadata['corner_position'] = [corner_x, corner_x, corner_z]
    metadata["energy"] = metadata["energy"]*scipy.constants.elementary_charge

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    bbox = np.ones((kernel_width,kernel_width))

    filtered_frames = []
    for i in range(0, clean_frame.shape[0]):
        clean_frame[i] = filter_frame(clean_frame[i], bbox)

        filtered_frames.append(shift_rescale(clean_frame[i], (0,0), metadata["output_frame_width"], metadata["output_frame_width"]/metadata["padded_frame_width"])[0])

    filtered_frames = npo.array(filtered_frames)

    com = center_of_mass(filtered_frames*(filtered_frames>0), yy)

    com = np.round(com)

    metadata["center_of_mass"] = metadata["output_frame_width"]//2 - com
    metadata["output_padded_ratio"] = metadata["output_frame_width"]/metadata["padded_frame_width"]

    return metadata, background_avg


def prepare(metadata, dark_frames, raw_frames):

    dark_frames = np.array(dark_frames)

    n_frames = raw_frames.shape[0]
    n_total_frames = metadata["translations"].shape[0]
    if metadata["double_exposure"]: n_total_frames *= 2

    #This takes the center of the stack as a center frame(s)
    center = int(n_total_frames)//2

    #Check this in case we are in double exposure
    if center % 2 == 1:
        center -= 1

    extra_frames = 2  #we take a couple more frames to compute the center of mass, careful not to start on an odd exposure frame here

    center_frames = []
    if metadata["double_exposure"]:
        metadata["double_exp_time_ratio"] = metadata["dwell1"] // metadata["dwell2"] # time ratio between long and short exposure
        
        for i in range(center, center + extra_frames + 1, 2):
            center_frames.append(raw_frames[i])
            center_frames.append(raw_frames[i + 1])

    else:

        for i in range(center, center + extra_frames + 1, 1):
            center_frames.append(raw_frames[i])

    center_frames = np.array(center_frames)
    
    metadata, background_avg =  compute_background_metadata(metadata, center_frames, dark_frames)

    return metadata, background_avg


def process_batch(metadata, frames_batch, background_avg, out_data, out_index, kernel_box, local_batch_size):


    for i in range(0, local_batch_size, metadata['double_exposure']+1):       
        
        if metadata["double_exposure"]:
            clean_frame = combine_double_exposure(cleanXraw(frames_batch[i]-background_avg[0]), \
                                                  cleanXraw(frames_batch[i+1]-background_avg[1]), metadata["double_exp_time_ratio"])           
        else:
            clean_frame = cleanXraw(frames_batch[i]-background_avg)
          
        filtered_frame = filter_frame(clean_frame, kernel_box)

        centered_rescaled_frame = shift_rescale(filtered_frame, metadata["center_of_mass"], metadata["output_frame_width"], metadata["output_padded_ratio"])

        out_data[out_index] = centered_rescaled_frame

        out_index += 1 


def process(metadata, raw_frames_tiff, background_avg, local_batch_size):

    n_total_frames = raw_frames_tiff.shape[0]
    n_frames = n_total_frames

    #if the batch size is not even in double exposure we fix that
    if local_batch_size % 2 != 0 and metadata['double_exposure']:
        local_batch_size += 1

    #If the batch size is not given or it is too big, we set it up to give work to every rank
    if local_batch_size == None or local_batch_size * mpi_size > n_total_frames:
        local_batch_size = n_total_frames // mpi_size

    batch_size = mpi_size * local_batch_size
    
    if metadata["double_exposure"]:
        printv(color("\nProcessing the stack of raw frames as a double exposure scan...\n", bcolors.OKGREEN))
        n_frames //= 2 # 1/2 for double exposure
    else:
        printv(color("\nProcessing the stack of raw frames as a single exposure scan...\n", bcolors.OKGREEN))

    printv(color("\r Processing a stack of frames of size: {}".format((raw_frames_tiff.shape[0], raw_frames_tiff[0].shape[0], raw_frames_tiff[0].shape[1])), bcolors.HEADER))
    printv(color("\r Using a local batch size per MPI rank = " + str(local_batch_size), bcolors.HEADER))

    #This stores the frames indexes that are being process by this mpi rank
    my_indexes = []
    import cosmicp.diskIO as diskIO
    n_batches = raw_frames_tiff.shape[0] // batch_size

    #Here we correct if the total number of frames is not a multiple of batch_size  
    extra = raw_frames_tiff.shape[0] - (n_batches * batch_size)

    extra_last_batch = None
    if rank * local_batch_size < extra: 
        n_batches = n_batches + 1
        n_ranks_extra = int(np.ceil(extra/local_batch_size))
        #We always overshot the batch sizes if they don't match perfectly (that is when extra % local_batch_size != 0)
        #To account for this, we need to have an index substraction (extra_last_batch) for last rank accross the ones having extra work
        if rank == n_ranks_extra - 1 and extra % local_batch_size != 0: 
            extra_last_batch = - (local_batch_size - (extra % local_batch_size)) // (metadata['double_exposure']+1)     
    
    out_data_shape = (n_batches * local_batch_size //(metadata['double_exposure']+1) , metadata["output_frame_width"], metadata["output_frame_width"])
    out_data = np.empty(out_data_shape,dtype=np.float32)
    frames_batch = npo.empty((local_batch_size, raw_frames_tiff[0].shape[0], raw_frames_tiff[0].shape[1]))

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    kernel_box = np.ones((kernel_width,kernel_width))

#---------------------------------------------
    
    cleanXraw_vmap = jax.vmap(lambda x: cleanXraw(x - background_avg))
    cleanXraw_vmap_d1 = jax.vmap(lambda x: cleanXraw(x - background_avg[0]))
    cleanXraw_vmap_d2 = jax.vmap(lambda x: cleanXraw(x - background_avg[1]))

    combine_double_exposure_vmapf = jax.vmap(lambda x, y: combine_double_exposure(x, y, metadata["double_exp_time_ratio"]))

    #single and double exposure functions
    f_cleanframes = jax.jit(lambda x: cleanXraw_vmap(x))
    f_cleanframes_d = jax.jit(lambda x, y: combine_double_exposure_vmapf(cleanXraw_vmap_d1(x), cleanXraw_vmap_d2(y)))

    def f(clean_frame):
        filtered_frame = filter_frame(clean_frame, kernel_box)
        centered_rescaled_frame = shift_rescale(filtered_frame, metadata["center_of_mass"], metadata["output_frame_width"], metadata["output_padded_ratio"])
        return centered_rescaled_frame

    process_batch_vmapf = jax.vmap(f)

    f_all = jax.jit(lambda x: process_batch_vmapf(f_cleanframes(x)))
    f_all_d = jax.jit(lambda x, y: process_batch_vmapf(f_cleanframes_d(x, y)))

    for i in range(0, n_batches):

        local_i = ((i * batch_size) + (rank * local_batch_size)) 

        #we handle uneven shapes here
        upper_bound = min(local_i  + local_batch_size, n_total_frames)

        local_range = range(local_i // (metadata['double_exposure']+1) , upper_bound // (metadata['double_exposure']+1))

        my_indexes.extend(local_range)

        i_s = i * local_batch_size // (metadata['double_exposure']+1)
        i_e = i_s + local_batch_size // (metadata['double_exposure']+1)

        for j in range(local_i, upper_bound) : 
            frames_batch[j % local_batch_size] = raw_frames_tiff[j][:, :]

        if metadata["double_exposure"]:
            centered_rescaled_frames_jax = f_all_d(frames_batch[:-1:2], frames_batch[1::2])
        else:
            centered_rescaled_frames_jax = f_all(frames_batch)

        # TODO: 'centered_rescaled_frames_jax' picks up an additional dimension somehow, should fix this...
        out_data = jax.ops.index_update(out_data, jax.ops.index[i_s:i_e, :, :], centered_rescaled_frames_jax[:,0,:,:])

        if rank == 0:
            sys.stdout.write(color("\r Computing batch = %s/%s " %(i+1,n_batches), bcolors.HEADER))
            sys.stdout.flush()

    if rank == 0: print("\n")
    return out_data[:extra_last_batch], my_indexes


#------------------------
#This is the baseline no-vmap version of the above, slightly slower
    '''
    printd(n_batches)

    for i in range(0, n_batches):

        local_i = ((i * batch_size) + (rank * local_batch_size)) 

        #we handle uneven shapes here
        upper_bound = min(local_i  + local_batch_size, n_total_frames)

        local_range = range(local_i // (metadata['double_exposure']+1) , upper_bound // (metadata['double_exposure']+1))

        my_indexes.extend(local_range)
        #if rank == 0: print(my_indexes)

        for j in range(local_i, upper_bound) : 
            printd(j)
            frames_batch[j % local_batch_size] = raw_frames_tiff[j][:, :]

        process_batch(metadata, frames_batch, background_avg, out_data, i * local_batch_size // (metadata['double_exposure']+1), kernel_box, local_batch_size)

    return out_data[:extra_last_batch], my_indexes
'''


def save_results(fname, metadata, local_data, my_indexes, n_frames):

    n_elements = npo.prod([i for i in local_data.shape])

    frames_gather = gather(local_data, (n_frames, local_data[0].shape[0], local_data[0].shape[1]), n_elements, npo.float32)  

    #we need the indexes too to map properly each gathered frame
    index_gather = gather(my_indexes, n_frames, len(my_indexes), npo.int32)

    if rank == 0:

        #Here we generate a proper index pull map, with respect to the input
        index_gather = npo.array([ npo.where(index_gather==i)[0][0] for i in range(0,len(index_gather))])

        frames_gather[:,:,:] = frames_gather[index_gather,:,:]

        printv(color("\r Final output data size: {}".format(frames_gather.shape), bcolors.HEADER))

        io = IO()
        output_filename = os.path.splitext(fname)[:-1][0][:-4] + "cosmic2.cxi"

        printv(color("\nSaving cxi file: " + output_filename + "\n", bcolors.OKGREEN))

        #This deletes and rewrites a previous file with the same name
        try:
            os.remove(output_filename)
        except OSError:
            pass

        io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data

        data_shape = frames_gather.shape
        out_frames, fid = frames_out(output_filename, data_shape)  

        dataAve = frames_gather[()].mean(0)
        pMask = np.fft.fftshift((dataAve > 0.1 * dataAve.max()))
        probe = np.sqrt(np.fft.fftshift(dataAve)) * pMask
        probe = np.fft.ifftshift(np.fft.ifftn(probe))
        dset = fid.create_dataset('entry_1/instrument_1/detector_1/probe', data = probe)
        dset = fid.create_dataset('entry_1/instrument_1/detector_1/data_illumination', data = probe)
        dset = fid.create_dataset('entry_1/instrument_1/source_1/probe', data = probe)
        dset = fid.create_dataset('entry_1/instrument_1/source_1/data_illumination', data = probe)
        dset = fid.create_dataset('entry_1/instrument_1/source_1/illumination', data = probe)
        dset = fid.create_dataset('entry_1/instrument_1/detector_1/probe_mask', data = pMask)

        out_frames[:, :, :] = frames_gather[:, :, :]

        fid.close()


        
