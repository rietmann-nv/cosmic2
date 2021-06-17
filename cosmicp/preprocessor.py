import sys
import os
import jax.numpy as np
import cupy as cp
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
from .common import printd, printv, rank, igatherv, gather
from .common import  size as mpi_size
from .diskIO import IO, frames_out

from timeit import default_timer as timer
from cupy.cuda.nvtx import RangePush, RangePop

def get_chunk_slices(n_slices):

    chunk_size =np.int32( np.ceil(n_slices/mpi_size)) # ceil for better load balance
    nreduce=(chunk_size*(mpi_size)-n_slices)  # how much we overshoot the size
    start = np.concatenate((np.arange(mpi_size-nreduce)*chunk_size,
                            (mpi_size-nreduce)*chunk_size+np.arange(nreduce)*(chunk_size-1)))
    stop = np.append(start[1:],n_slices)

    start=start.reshape((mpi_size,1))
    stop=stop.reshape((mpi_size,1))
    slices=np.uint32(np.concatenate((start,stop),axis=1))
    return slices 


def get_loop_chunk_slices(ns, ms, mc ):
    # ns: num_slices, ms=mpi_size, mc=max_chunk
    # loop_chunks size
    if np.isinf(mc): 
#        print("ms",ms)
        return np.array([0,ns],dtype='int32')
#    print("glc ms",ms)

    ls=np.int32(np.ceil(ns/(ms*mc)))    
    # nreduce: how many points we overshoot if we use max_chunk everywhere
    nr=ls*mc*ms-ns
    #print(nr,ls,mc)
    # number of reduced loop_chunks:
    
    cr=np.ceil(nr/ms/ls)
    # make it a multiple of 2 since we do 2 slices at once
    #cr=np.ceil(nr/ms/ls/2)*2

    if nr==0:
        rl=0
    else:
        rl=np.int32(np.floor((nr/ms)/cr))
    
    loop_chunks=np.concatenate((np.arange(ls-rl)*ms*mc,(ls-rl)*ms*mc+np.arange(rl)*ms*(mc-cr),np.array([ns])))
    return np.int32(loop_chunks)


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


def prepare(metadata, frames, dark_frames):

    ## get one frame to compute center

    if metadata["double_exposure"]:

        background_avg = split_background(dark_frames)

        frame_exp1 = (frames[0::2] - background_avg[0])[0]
        frame_exp2 = (frames[1::2] - background_avg[1])[0]

        # get one clean frame
        clean_frame = combine_double_exposure(cleanXraw(frame_exp1), cleanXraw(frame_exp2), metadata["double_exp_time_ratio"])

    else:        
        background_avg = np.average(dark_frames,axis=0)
        clean_frame = cleanXraw(frames - background_avg)


    #TODO: When do we know this one? Is the one original shape from the beginning right?
    metadata["frame_width"] = clean_frame.shape[0]

    #Coordinates from 0 to frame width, 1 dimension
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))
    yy=np.reshape(np.arange(metadata["output_frame_width"]),(metadata["output_frame_width"],1))


    #metadata["energy"]=metadata["energy"]/scipy.constants.elementary_charge
    # cropped width of the raw clean frames
    if metadata["desired_padded_input_frame_width"]:
        metadata["padded_frame_width"] = metadata["desired_padded_input_frame_width"]

    else:
        metadata["padded_frame_width"] = resolution2frame_width(metadata["final_res"], metadata["detector_distance"], metadata["energy"], metadata["detector_pixel_size"], metadata["frame_width"]) 
    
    # modify pixel size; the pixel size is rescaled
    metadata["x_pixel_size"] = metadata["detector_pixel_size"] * metadata["padded_frame_width"] / metadata["output_frame_width"]
    metadata["y_pixel_size"] = metadata["x_pixel_size"]

    #metadata['detector_pixel_size'] = metadata["x_pixel_size"]

    # frame corner
    corner_x = metadata['x_pixel_size']*metadata['output_frame_width']/2  
    corner_z = metadata['detector_distance']                
    metadata['corner_position'] = [corner_x, corner_x, corner_z]
    metadata["energy"] = metadata["energy"]*scipy.constants.elementary_charge

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    bbox = np.ones((kernel_width,kernel_width))

    filtered_frame = filter_frame(clean_frame, bbox)

    filtered_frame = shift_rescale(filtered_frame, (0,0), metadata["output_frame_width"], metadata["output_frame_width"]/metadata["padded_frame_width"])[0]

    # we need a shift, we take it from the first frame:
    com = center_of_mass(filtered_frame*(filtered_frame>0), yy)

    com = np.round(com)

    metadata["center_of_mass"] = metadata["output_frame_width"]//2 - com

    metadata["output_padded_ratio"] = metadata["output_frame_width"]/metadata["padded_frame_width"]


    return metadata, background_avg

@jax.jit
def calculate_mpi_chunk(n_total_frames, my_rank, n_ranks):

    frames_per_rank = n_total_frames//n_ranks

    #we always make chunks multiple of 2 because of double exposure
    if frames_per_rank % 2 == 1:
        frames_per_rank -= 1

    extra_work = 0

    if  rank == mpi_size - 1:
        extra_work =  n_total_frames - (n_ranks * frames_per_rank) 

    printv("Frames to compute per rank: " + str(frames_per_rank))

    frames_range = slice(my_rank * frames_per_rank, ((my_rank + 1) * frames_per_rank) + extra_work)

    printd("My range of ranks: " + str(frames_range) + ", my extra work: " + str(extra_work))

    return frames_range



# loop through all frames and save the result

def process_stack(metadata, frames_stack, background_avg, out_data):
    #n_frames = my_indices.stop-my_indices.start

    n_frames = frames_stack.shape[0]
    #my_indices = calculate_mpi_chunk(n_frames, rank, mpi_size)
    #n_frames = my_indices.stop-my_indices.start+1
    
    if metadata["double_exposure"]:    
        n_frames //= 2 # 1/2 for double exposure

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    bbox = np.ones((kernel_width,kernel_width))

    if metadata['double_exposure']:
        printv("\nProcessing the stack of raw frames - double exposure...\n")
    else:
        printv("\nProcessing the stack of raw frames...\n")

    max_chunk_slice = 1
    loop_chunks=get_loop_chunk_slices(n_frames, mpi_size, max_chunk_slice )

    frames_local=None
    pgather = None
    
    out_data_shape = (max_chunk_slice*mpi_size, metadata["output_frame_width"], metadata["output_frame_width"])
    if rank == 0:      
        frames_chunks = npo.empty(out_data_shape,dtype=np.float32)

    output_padded_ratio = metadata["output_frame_width"]/metadata["padded_frame_width"]

    chunk_slices = []
    chunks = []
    for ii in range(loop_chunks.size-1):

        nslices = loop_chunks[ii+1]-loop_chunks[ii]
        chunk_slices.append(get_chunk_slices(nslices)) 
        chunks.append(chunk_slices[-1][rank,:]+loop_chunks[ii])
    #This stored the frames indexes are being process by this mpi rank
    my_indexes = []
    for ii in range(loop_chunks.size-1):
        
        # only one frame per chunk
        ii_frames= chunks[ii][0]
        my_indexes.append(ii_frames)
        # empty 
        if chunks[ii][1]-chunks[ii][0] == 0:
            centered_rescaled_frame = np.empty((0),dtype = np.float32)
        else:            
        
            if metadata["double_exposure"]:
                clean_frame = combine_double_exposure(cleanXraw(frames_stack[ii_frames*2]-background_avg[0]), cleanXraw(frames_stack[ii_frames*2+1]-background_avg[1]), metadata["double_exp_time_ratio"])
                
            else:
                clean_frame = cleanXraw(frames_stack[ii_frames]-background_avg)
          
            filtered_frame = filter_frame(clean_frame, bbox)

            centered_rescaled_frame = shift_rescale(filtered_frame, metadata["center_of_mass"], metadata["output_frame_width"], output_padded_ratio)

        
        if rank ==0:
            frames_local =  frames_chunks[0:loop_chunks[ii+1]-loop_chunks[ii],:,:]
        
        pgather = igatherv(centered_rescaled_frame,chunk_slices[ii],data=frames_local)   
        
        if mpi_size > 1:
            pgather.Wait()

        if rank == 0:
            # out_data[loop_chunks[ii]:loop_chunks[ii+1],:,:] = frames_local
            out_data = jax.ops.index_update(out_data, jax.ops.index[loop_chunks[ii]:loop_chunks[ii+1],:,:], frames_local)
                   
        if rank == 0 :
 
            ii_rframe = ii_frames#*(metadata["double_exposure"]+1) 
            sys.stdout.write('\r frame {}/{}, loop_chunk {}/{}:{}, mpi chunks {}'.format(ii_rframe+1, n_frames, ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[ii][:,0],chunk_slices[ii][-1,1]).ravel()))
#            sys.stdout.write('\r frame = %s/%s ' %(ii_frames+1, n_frames))
            sys.stdout.flush()
            #print("\n")

    return out_data, my_indexes



def prepare_2(metadata, dark_frames, raw_frames):

    dark_frames = np.array(dark_frames)

    n_frames = raw_frames.shape[0]
    n_total_frames = metadata["translations"].shape[0]
    if metadata["double_exposure"]: n_total_frames *= 2

    #This takes the center of the stack as a center frame(s)
    center = int(n_total_frames)//2

    #Check this in case we are in double exposure
    if center % 2 == 1:
        center -= 1

    if metadata["double_exposure"]:
        metadata["double_exp_time_ratio"] = metadata["dwell1"] // metadata["dwell2"] # time ratio between long and short exposure
        center_frames = np.array([raw_frames[center], raw_frames[center + 1]])
    else:
        center_frames = raw_frames[center]
   
    
    metadata, background_avg =  prepare(metadata, center_frames, dark_frames)

    return metadata, background_avg


def process_jax(metadata, raw_frames_tiff, background_avg_np):

    # larger batch size is usually better...
    local_batch_size = 100

    #if the batch size is not even in double exposure we fix that
    if local_batch_size % 2 != 0 and metadata['double_exposure']:
        local_batch_size += 1

    batch_size = mpi_size * local_batch_size

    n_frames = raw_frames_tiff.shape[0]

    printv(n_frames)
    
    if metadata["double_exposure"]:
        printv("\nProcessing the stack of raw frames - double exposure...\n")
        n_frames //= 2 # 1/2 for double exposure
    else:
        printv("\nProcessing the stack of raw frames...\n")

    #This stores the frames indexes that are being process by this mpi rank
    my_indexes = []
    import cosmicp.diskIO as diskIO
    n_batches = raw_frames_tiff.shape[0] // batch_size

    #Here we correct if the total number of frames is not a multiple of batch_size  
    extra = raw_frames_tiff.shape[0] - (n_batches * batch_size)
    print(extra)
    if rank * local_batch_size < extra: 
        n_batches = n_batches + 1
    
    out_data_shape = (n_batches * local_batch_size //(metadata['double_exposure']+1) , metadata["output_frame_width"], metadata["output_frame_width"])
    out_data = np.empty(out_data_shape,dtype=np.float32)

    frames_batch_np = npo.empty((local_batch_size, raw_frames_tiff[0].shape[0], raw_frames_tiff[0].shape[1]), dtype=npo.float32)
    print("Number of frames ({}x{}): ", raw_frames_tiff[0].shape[0], raw_frames_tiff[0].shape[0], raw_frames_tiff.shape[0])
    print("Output shape: ", out_data_shape)

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    kernel_box = np.ones((kernel_width,kernel_width))

    printd(n_batches)

    # jax jit & vmap preparations
    # These functions have to be persistent across loop iterations, because their first-use compilation (expensive) is cached.
    background_avg = np.asarray(background_avg_np)
    cleanXraw_vmap_f0 = jax.vmap(lambda x: cleanXraw(x - background_avg[0]))
    cleanXraw_vmap_f1 = jax.vmap(lambda x: cleanXraw(x - background_avg[1]))
    combine_double_exposure_vmapf = jax.vmap(lambda x, y: combine_double_exposure(x, y, metadata["double_exp_time_ratio"]))
    f_cleanframes = jax.jit(lambda x, y: combine_double_exposure_vmapf(cleanXraw_vmap_f0(x), cleanXraw_vmap_f1(y)))
    def f(clean_frame):
        filtered_frame = filter_frame(clean_frame, kernel_box)
        centered_rescaled_frame = shift_rescale(filtered_frame, metadata["center_of_mass"], metadata["output_frame_width"], metadata["output_padded_ratio"])
        return centered_rescaled_frame
    process_batch_vmapf = jax.vmap(f)
    f_all = jax.jit(lambda x, y: process_batch_vmapf(f_cleanframes(x, y)))

    # we schedule the transfer ahead, to enable compute overlap with the disk I/O
    local_i = ((0 * batch_size) + (rank * local_batch_size))
    start_hdf5 = timer()
    for j in range(local_i, local_i  + local_batch_size) : frames_batch_np[j % local_batch_size] = raw_frames_tiff[j][:, :]
    end_hdf5 = timer()
    frames_batch = np.array(frames_batch_np)

    start = timer()
    for i in range(0, n_batches):
        # RangePush("preprocessing")
        # start_i = timer()
        

        local_range = range(local_i // (metadata['double_exposure']+1) , (local_i  + local_batch_size) // (metadata['double_exposure']+1))

        my_indexes.extend(local_range)
        #if rank == 0: print(my_indexes)

        i_s = i * local_batch_size // (metadata['double_exposure']+1)
        i_e = i_s + local_batch_size // (metadata['double_exposure']+1)

        # non-vmap version
        # process_batch(metadata, frames_batch, background_avg, out_data, i * local_batch_size // (metadata['double_exposure']+1), kernel_box, local_batch_size)        
        # these JAX GPU calls are async, so the will overlap with the Disk I/O in next step (hopefully)
        centered_rescaled_frames_jax = f_all(frames_batch[:-1:2], frames_batch[1::2])
        
        if i<n_batches-1:
            # RangePush("HDF5 -> NumPy")
            # start_hdf5 = timer()
            local_i = (((i+1) * batch_size) + (rank * local_batch_size))
            for j in range(local_i, local_i  + local_batch_size) : frames_batch_np[j % local_batch_size] = raw_frames_tiff[j][:, :]
            # end_hdf5 = timer()
            # RangePop()
            frames_batch = np.array(frames_batch_np)


        # TODO: 'centered_rescaled_frames_jax' picks up an additional dimension somehow, should fix this...
        # Put this at the end, because it is blocking (I think)
        out_data = jax.ops.index_update(out_data, jax.ops.index[i_s:i_e, :, :], centered_rescaled_frames_jax[:,0,:,:])

        # RangePop()

    end = timer()
    print("Total time preprocessing: ", end-start)

    return out_data, my_indexes


def process_batch(metadata, frames_batch, background_avg, out_data, out_index, kernel_box, local_batch_size):

    #printd(out_index)
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


def process(metadata, raw_frames_tiff, background_avg):

    n_total_frames = raw_frames_tiff.shape[0]
    n_frames = n_total_frames

    local_batch_size = 500

    #if the batch size is not even in double exposure we fix that
    if local_batch_size % 2 != 0 and metadata['double_exposure']:
        local_batch_size += 1

    #If the batch size is not given or it is too big, we set it up to give work to every rank
    if local_batch_size == None or local_batch_size * mpi_size > n_total_frames:
        local_batch_size = n_total_frames // mpi_size
        
    printv("Using local batch size per MPI rank = " + str(local_batch_size))

    batch_size = mpi_size * local_batch_size

    printv(n_frames)
    
    if metadata["double_exposure"]:
        printv("\nProcessing the stack of raw frames - double exposure...\n")
        n_frames //= 2 # 1/2 for double exposure
    else:
        printv("\nProcessing the stack of raw frames...\n")

    #This stores the frames indexes that are being process by this mpi rank
    my_indexes = []
    import cosmicp.diskIO as diskIO
    n_batches = raw_frames_tiff.shape[0] // batch_size

    #Here we correct if the total number of frames is not a multiple of batch_size  
    extra = raw_frames_tiff.shape[0] - (n_batches * batch_size)
    print(extra)
    extra_last_batch = None
    if rank * local_batch_size < extra: 
        n_batches = n_batches + 1
        n_ranks_extra = int(np.ceil(extra/local_batch_size))
        print(n_ranks_extra)
        #We always overshot the batch sizes if they don't match perfectly (that is when extra % local_batch_size != 0)
        #To account for this, we need to have an index substraction (extra_last_batch) for last rank accross the ones having extra work
        if rank == n_ranks_extra - 1 and extra % local_batch_size != 0: 
            extra_last_batch = - (local_batch_size - (extra % local_batch_size)) // (metadata['double_exposure']+1)

    printd(extra_last_batch)        
    
    out_data_shape = (n_batches * local_batch_size //(metadata['double_exposure']+1) , metadata["output_frame_width"], metadata["output_frame_width"])
    out_data = npo.empty(out_data_shape,dtype=np.float32)
    frames_batch = npo.empty((local_batch_size, raw_frames_tiff[0].shape[0], raw_frames_tiff[0].shape[1]))

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    kernel_box = np.ones((kernel_width,kernel_width))

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
	


def save_results(fname, metadata, local_data, my_indexes, n_frames):

    print(len(my_indexes))
    print(n_frames)
    printd(local_data.shape)
    printd(my_indexes)
    

    #local = np.array([[[(rank + 1)*1, (rank + 1)*2],[(rank + 1)*3, (rank + 1)*4]], [[(rank + 1)*5, (rank + 1)*6],[(rank + 1)*7, (rank + 1)*8]]], dtype='int32')

    #print(local)
    #print(local.shape)

    #test_gather = gather(local, (4, local[0].shape[0], local[0].shape[1]), npo.int32)

    print("gather-------------------------------")

    n_elements = npo.prod([i for i in local_data.shape])

    frames_gather = gather(local_data, (n_frames, local_data[0].shape[0], local_data[0].shape[1]), n_elements, npo.float32)  

    #we need the indexes too to map properly each gathered frame
    print(type(my_indexes[0]))
    index_gather = gather(my_indexes, n_frames, len(my_indexes), npo.int32)

    if rank == 0:
   
        #print(test_gather)
        #print(test_gather.shape)

        import sys
        import numpy
        numpy.set_printoptions(threshold=sys.maxsize)
        print(frames_gather[:,0,0])
        print(index_gather)
        print(index_gather.shape)

        #Here we generate a proper index pull map, with respect to the input
        index_gather = npo.array([ npo.where(index_gather==i)[0][0] for i in range(0,len(index_gather))])

        print(index_gather)

        frames_gather[:,:,:] = frames_gather[index_gather,:,:]

        print(frames_gather[:,0,0])

        print("Output_data size: {}".format(frames_gather.shape))

        io = IO()
        output_filename = os.path.splitext(fname)[:-1][0][:-4] + "cosmic2.cxi"

        printv("\nSaving cxi file metadata: " + output_filename + "\n")

        #This deletes and rewrites a previous file with the same name
        try:
            os.remove(output_filename)
        except OSError:
            pass

        io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data
   

        data_shape = frames_gather.shape
        out_frames, fid = frames_out(output_filename, data_shape)  

        out_frames[:, :, :] = frames_gather[:, :, :]

        fid.close()


        
