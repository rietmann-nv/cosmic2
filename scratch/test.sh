# install:
# pip install -e .

# python generate_raw_cxi.py /data/smarchesini/newlens_20/200220/NS_200220033/NS_200220033_002.cxi /data/smarchesini/newlens_20/200220/NS_200220033/200220033/001/ /data/smarchesini/newlens_20/200220/NS_200220033/200220033/002/ 2

# python preprocess.py raw_NS_200220033_002.cxi
# cp raw_NS_200220033_002.cxi filtered_NS_200220033_002.cxi 

# mpirun -n 4 ptycho.py -i 100 -T 2 -r 2 -M data_set.cxi

