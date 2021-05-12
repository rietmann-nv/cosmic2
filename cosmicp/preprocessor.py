import sys
import jax.numpy as np
import jax
import scipy
import scipy.constants
import scipy.interpolate
import scipy.signal
import numpy as npo
from .fccd import imgXraw as cleanXraw
from .common import printd, printv, rank, igatherv
from .common import  size as mpi_size


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

    for ii in range(loop_chunks.size-1):
        
        # only one frame per chunk
        ii_frames= chunks[ii][0]
        
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

    return out_data

        
