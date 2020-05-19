import sys
import numpy as np
import scipy
from fccd import imgXraw as cleanXraw


def combine_double_exposure(data0, data1, double_exp_time_ratio, thres=3e3):

    msk=data0<thres    

    return (double_exp_time_ratio+1)*(data0*msk+data1)/(double_exp_time_ratio*msk+1)

def resolution2frame_width(final_res, detector_distance, energy, detector_pixel_size, frame_width):

    hc=scipy.constants.Planck*scipy.constants.c/scipy.constants.elementary_charge
   
    wavelength = hc/energy
    padded_frame_width = width**2*detector_pixel_size*final_res/(detector_distance*wavelength)

    return padded_frame_width # cropped (TODO:or padded?) width of the raw clean frames

#TODO: WHAT IS THIS
def center_of_mass(img, xx):
    return np.array([np.sum(img*xx)/np.sum(img), np.sum(img*xx.T)/np.sum(img)])


#TODO: WHAT IS THIS
def filter_frame(frame, bbox):
    return scipy.signal.convolve2d(frame, bbox, mode='same', boundary='fill')


#TODO: WHAT IS THIS (center frames?)
def shift_rescale(img, coord, center_of_mass):
    img_out=(scipy.interpolate.interp2d(xx, xx, img, fill_value=0)(coord+center_of_mass[1],coord+center_of_mass[0])).T
    img_out*=(img_out>0)
    return img_out

def split_background(background_double_exp):

    # split the average from 2 exposures:
    bkg_avg0=np.average(background_double_exp[0::2],axis=0)
    bkg_avg1=np.average(background_double_exp[1::2],axis=0)

    return np.array([bkg_avg0, bkg_avg1])


def prepare(metadata, frames, dark_frames):

    ## get one frame to compute center

    if metadata["double_exposure"]:

        background_avg = split_background(dark_frames)

        frame_exp1 = frames[0::2] - background_avg[0]
        frame_exp2 = frames[1::2] - background_avg[1]

        # get one clean frame
        clean_frame = combine_double_exposure(cleanXraw(frame_exp1), cleanXraw(frame_exp2), metadata["double_exp_time_ratio"])

    else:        
        background_avg = np.average(dark_frames,axis=0)
        clean_frame = cleanXraw(frames - background_avg)


    #TODO: When do we know this one? Is the one original shape from the beginning right?
    metadata["frame_width"] = clean_frame.shape[0]

    #TODO: WHAT IS THIS
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))

    # we need a shift, we take it from the first frame:
    center_of_mass = center_of_mass(clean_frame*(clean_frame>0), xx) - metadata["frame_width"]//2
    center_of_mass = np.round(com)

    metadata["center_of_mass"] = center_of_mass

    # cropped width of the raw clean frames

    if metadata["desired_padded_input_frame_width"]:
        metadata["padded_frame_width"] = metadata["desired_padded_input_frame_width"]

    else:
        metadata["padded_frame_width"] = resolution2frame_width(metadata["final_res"], metadata["detector_distance"], metadata["energy"], metadata["detector_pixel_size"], metadata["frame_width"]) 
        
    # modify pixel size; the pixel size is rescaled
    metadata["x_pixel_size"] = detector_pixel_size * metadata["padded_frame_width"] / metadata["output_frame_width"]
    metadata["y_pixel_size"] = metadata["x_pixel_size"]

    return metadata, background_avg


# loop through all frames and save the result
def process_stack(metadata, frames_stack, background_avg):

    if metadata["double_exposure"]:    
        n_frames = frames_stack.shape[0]//2 # 1/2 for double exposure
    else:
        n_frames = frames_stack.shape[0]

    #TODO: WHAT IS THIS
    box_width = np.max([np.int(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1])
    bbox = np.ones((box_width,box_width))

    #TODO: WHAT IS THIS, use parentesis or break it down into multiple lines...
    coord = np.arange(-metadata["output_frame_width"]//2, metadata["output_frame_width"]//2) / metadata["output_frame_width"] * metadata["padded_frame_width"] + metadata["frame_width"]//2

    out_data_shape = (n_frames, metadata["output_frame_width"], metadata["output_frame_width"])

    out_data = np.zeros(out_data_shape, dtype= np.float32)

    for ii in np.arange(n_frames):

        if metadata["double_exposure"]:
            clean_frame = combine_double_exposure(cleanXraw(frames_stack[ii*2]-background_avg[0]), cleanXraw(frames_stack[ii*2+1]-background_avg[1]), metadata["double_exp_time_ratio"])

        else:
            clean_frame = cleanXraw(frames_stack[ii]-background_avg)
      
        filtered_frame = filter_frame(clean_frame, bbox)

        #Center and downsample a clean frame
        centered_rescaled_frame = shift_rescale(filtered_frame, coord, metadata["center_of_mass"])
    
        out_data[ii] = centered_rescaled_frame
        #print('hello')
        sys.stdout.write('\r frame = %s/%s ' %(ii+1, n_frames))
        sys.stdout.flush()

    return out_data



