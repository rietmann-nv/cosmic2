import jax.numpy as jnp
import jax

@jax.jit
def combine_double_exposure(data0, data1, double_exp_time_ratio, thres=3e3):

    msk=data0<thres    

    return (double_exp_time_ratio+1)*(data0*msk+data1)/(double_exp_time_ratio*msk+1)


@jax.jit
def split_background(background_double_exp):

    # split the average from 2 exposures:
    bkg_avg0=jnp.average(background_double_exp[0::2],axis=0)
    bkg_avg1=jnp.average(background_double_exp[1::2],axis=0)

    return jnp.array([bkg_avg0, bkg_avg1])
