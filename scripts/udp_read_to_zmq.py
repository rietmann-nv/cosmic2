"""
Experimental framgrabber that should work without pyQT4 and zmq

Can actualluy be deleted once the framegrabbing is stable
"""


import socket
import time
import urllib.request, urllib.error, urllib.parse
import zmq
import cosmicp.udpframereader as udpr

import numpy as np
import cupy as cp

def splitaddr(addr):
    host, port = urllib.parse.splitport(addr)
    if port is None:
        return host, port
    else:
        return host,int(port)


class Framegrabber:
    """Grab frames from the camera and publishes assembled frames over 0MQ."""

    def __init__(self, fsize,
                       read_addr= "localhost:49205",
                       send_addr= "127.0.0.1:50000",
                       udp_addr ="10.0.5.207:49203"):

        # Configuration
        self.read_addr = splitaddr(read_addr)
        self.send_addr = splitaddr(send_addr) if send_addr is not None else None
        self.udp_addr = splitaddr(udp_addr) 

        # Size of frame (unit16)
        self.fsize = fsize
        self.fbytes = None

        # Buffers and counters
        self.fbuffer  = None
        self.gpubuffers = []
        self.lastbuffer = None
        self.fnumber = -1
        self.fnumber1 = 0
        self.nreceive = 0
        self.nrecord = 0
        self.nsend = 0
        self.updaterate = 100
        self.t0 = time.time()
        self.t1 = time.time()
        
        print("-- Starting ZMQ Frame publisher --")

    def createReadFrameSocket(self):
        # A socket for reading from camera
        self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.camera_socket.setblocking(1)
        self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        self.camera_socket.bind(self.read_addr)
        self.camera_socket.sendto(b"dummy_data", self.udp_addr)
        print("Framegrabber is listening to data from the camera on ip %s port %d" % self.read_addr)
        
    def udpdump_to_file(self,filename='/tmp/udpdump.hex',npackets = 2000, packet_size=4104):
        
        f = open(filename,'w')
        for i in range(npackets):
            f.write(self.camera_socket.recv(packet_size))
        f.close()
            
    def createSendFrameSocket(self):
        # A socket for sending data (including meta data) to the backend
        context = zmq.Context()
        self.backend_socket = context.socket(zmq.PUB)
        self.backend_socket.bind('tcp://%s:%d' % self.send_addr)
        self.backend_socket.set_hwm(10000)
        print("Framegrabber is sending data to backend on ip %s port %d" % self.send_addr)
    
    def _recvframe(self):
        """receive frames from the FCCD. Return as soon as a new frame is ready, otherwise it is blocking."""
        #print "RECV: ", self.camera_socket.fileno(), self.fsize
        
        pkg = udpr.read_frame(self.camera_socket.fileno(), self.fsize)
        self.fbuffer, fnumber, self.fbytes = pkg
        self.nreceive += 1

        if fnumber > self.fnumber1:
            print("Dropped Frame(s) range %d - %d" % (self.fnumber1, fnumber))
        self.fnumber1 = fnumber + 1
        
        if (self.nreceive == self.updaterate):
            t1 = time.time()
            print("Reading at %.2fHz" % (self.nreceive/(t1-self.t0)))
            self.t0 = t1
            self.nreceive = 0
        
        self.fnumber = fnumber
        
    def _sendframe(self):
        """Sending frames to the processing backend."""
        if self.fbuffer is not None:

            # transfer to GPU
            self.gpubuffers.append(cp.array(np.frombuffer(self.fbuffer, '<u2')))

            cuda_ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self.gpubuffers[-1].data.ptr)
            # print("Sending IPC Handle for buffer with length: ", self.gpubuffers[-1].shape, self.gpubuffers[-1].dtype, cuda_ipc_handle)
            self.backend_socket.send_multipart([b'%d' % self.fnumber, b'%d' % len(self.gpubuffers[-1]),
                                                self.gpubuffers[-1].dtype.str.encode(), cuda_ipc_handle])

    def run(self):
        print("Framegrabber started")
        tsend = 0
        ii = 0
        while True:
            ii +=1
            self._recvframe()
            t = time.time()
            self._sendframe()
            ts = time.time()-t
            tsend = 0.05*ts + 0.95*tsend
            if ii % 40 == 0: 
                print("UDP load is %.2f ms" % (tsend * 1000) )


if __name__=='__main__':

    #FG=Framegrabber(2*1152*2000,read_addr= "127.0.0.1:49205",send_addr ="10.0.0.16:49206",udp_addr ="127.0.0.1:49203")
    ## With CIN
    #FG=Framegrabber(2*1152*1040,read_addr= "10.0.5.55:49205",send_addr ="127.0.0.1:49206",udp_addr ="10.0.5.207:49203")
    ## Simulation with test_stxmcontrol:
    FG=Framegrabber(2*1152*1040,read_addr="127.0.0.1:49205",send_addr="127.0.0.1:49206",udp_addr ="127.0.0.1:49203")
    FG.createReadFrameSocket()
    FG.createSendFrameSocket()
    FG.run()

