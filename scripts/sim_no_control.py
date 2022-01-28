import time
import socket
import struct
import datetime
import numpy as np
import cosmic.camera.fccd as fccd

today = datetime.date.fromtimestamp(time.time())

PARAMS = dict(
    # parameters for the frontend
    nstream=dict(
        step=0.03,  # mum
        num=15,  # number of
        bnum=5,  # num of dark points for each axes
        dwell=(100, 500), #(100, 100),  # msec
        energy=800,  # ev
    ),
    comm=dict(
        udp_ip='127.0.0.1',
        udp_port=49203,
    )
)

DELAY_100Hz = 1.8e-5
DELAY_100Hz = 1.8e-5


def eV2mum(eV):
    """\
    Convert photon energy in eV to wavelength (in vacuum) in micrometers.
    """
    wl = 1. / eV * 4.1356e-7 * 2.9998 * 1e6

    return wl


def abs2(X):
    return np.abs(X) ** 2

def stxm_probe(shape, inner=8, outer=25):

    # d = np.float(np.min(shape))
    X, Y = np.indices(shape).astype(float)
    X -= X.mean()
    Y -= Y.mean()
    R = (np.sqrt(X ** 2 + Y ** 2) < outer).astype(complex)
    r = (np.sqrt(X ** 2 + Y ** 2) > inner).astype(complex)
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(R * r)))

def nanoball_object(shape, rad=5, num=40):
    """ creates nanoballs as transmission """

    def cluster_coords(shape, rad=5, num=40):
        sh = shape

        def pick():
            return np.array([np.random.uniform(0, sh[0] - 1), np.random.uniform(0, sh[1] - 1)])

        coords = [np.array(
            [np.random.randint(sh[0] / 3, 2 * sh[0] / 3 - 1), np.random.randint(sh[0] / 3, 2 * sh[0] / 3 - 1)])]
        # np.rand.uniform(0,1.,tuple(sh)):
        for ii in range(num - 1):
            noresult = True
            for k in range(10000):
                c = pick()
                dist = np.sqrt(np.sum(abs2(np.array(coords) - c), axis=1))
                if (dist < 2 * rad).any():
                    continue
                elif (dist >= 2 * rad).any() and (dist <= 3 * rad).any():
                    break
                elif (0.001 + np.sum(8 / (dist ** 2)) > np.random.uniform(0, 1.)):
                    break

            coords.append(c)
        return np.array(coords)

    sh = shape
    out = np.zeros(sh)
    xx, yy = np.indices(sh)
    coords = cluster_coords(sh, rad, num)
    for c in coords:
        h = rad ** 2 - (xx - c[0]) ** 2 - (yy - c[1]) ** 2
        h[h < 0] = 0.
        out += np.sqrt(h)

    return out

class FccdSimulator:
    """Simulates STXM control and FCCD camera."""
    def __init__(self, params=None):
        super(FccdSimulator, self).__init__()

        # these could be part of input but not entirely important
        self.seed = 1983
        self.udp_packet_size = 4104
        self.udp_header_size = 8
        #self.shape = (1920, 960)  # (1940,1152)
        #self.fCCD = fccd.FCCD()
        self.shape = (980, 960)  # (1940,1152)
        self.fCCD = fccd.FCCD(nrows=self.shape[0]//2)
        self.psize = 30
        self.offset = 10000
        self.io_noise = 5
        self.photons_per_sec = 2e7
        self.resolution = 0.005
        self.dist = 80000
        self.adu_per_photon = 34
        self.zp_dia_outer = 30  # pixel  on screen
        self.zp_dia_inner = 12  # pixel on screen
        self.nanoball_rad = 5  # nanoball radius  (pixel)

        self.delay = 1e-7

        # DEADFOOD
        self.end_of_frame_msg = b"\xf1\xf2" + b"\xde\xad\xf0\x0d" + b"\x00\x00"

        # load parameters
        if params is not None:
            self.p = params
        else:
            from copy import deepcopy
            self.p = deepcopy(PARAMS)

        self.energy = self.p['nstream']['energy']

        self.photons = [self.photons_per_sec * d / 1000 for d in self.p['nstream']['dwell']]

        # Configuration
        self.udp_address = (self.p['comm']['udp_ip'], self.p['comm']['udp_port'])

        # calculate sim shape to resolution
        a = np.int(self.dist * eV2mum(self.energy) / (self.psize * self.resolution))
        assert a < np.min(self.shape), 'Too many pixel, choose larger resolution'
        for i in [256, 384, 512, 640, 768, 896, 1024]:
            if i > a:
                a = i
                break

        a = 384
        self.sim_shape = (a, a)
        print("Simulation resolution is %d x %d" % self.sim_shape)

        # make test frame
        X, Y = np.indices(self.shape)
        Y = (Y // 144) * 10
        Y[self.shape[0] // 2:, :] *= -1
        Y += Y.min()
        self.testframe = Y.astype(np.uint16)

        self.dark_frames = None
        self.exp_frames = None

    def make_ptycho_data(self):
        print("Preparing ptycho data ..")
        self.create_darks()
        print("Prepared dark data ..")
        self.create_exps()
        print("Prepared ptycho data ..")

    def create_darks(self):
        N = len(self.photons) * self.p['nstream']['bnum'] ** 2
        self.dark_frames = self._draw(np.zeros((N,) + self.shape).astype(int))

    def create_exps(self):
        """ makes a raster ptycho scan """
        # seed the random generatot to fixed value
        np.random.seed(self.seed)

        sh = self.sim_shape

        # positions
        num = self.p['nstream']['num']
        step = self.p['nstream']['step']

        pos = np.array([(step * k, step * j) for j in range(num) for k in range(num)])
        pixelpos = np.round(pos / self.resolution).astype(int)
        pixelpos -= pixelpos.min()
        pixelpos += 5

        # make object
        print("Preparing exit waves ..")

        osh = pixelpos.max(0) + np.array(sh) + 10
        nb = nanoball_object(osh, rad=self.nanoball_rad, num=400)
        nb /= nb.max()
        # nb = np.resize(nb,osh)
        self.ob = np.exp(0.2j * nb - nb / 2.)

        pr = stxm_probe(sh, outer=self.zp_dia_outer, inner=self.zp_dia_inner)
        pr /= np.sqrt(abs2(pr).sum())
        self.pr = pr

        a, b = sh
        exits = np.array([self.pr * self.ob[pr:pr + a, pc:pc + b] for (pr, pc) in pixelpos])
        fs = lambda e: np.fft.fftshift(np.fft.fft2(np.fft.fftshift(e))) / np.sqrt(sh[0] * sh[1])

        print("Propagating waves ..")
        stack = np.array([abs2(fs(e)) * ph for e in exits for ph in self.photons])

        # an ideal measurement
        diffstack = np.random.poisson(stack) * self.adu_per_photon


        print("Frame waves to larger detector shape ..")
        self.exp_frames = [self._draw(self._embed_frame(frame)) for frame in diffstack]


    def _embed_frame(self, frame):
        sh = frame.shape
        out = np.zeros(self.shape).astype(int)
        off = [(a - b) // 2 for (a, b) in zip(self.shape, sh)]
        out[off[0]:off[0] + sh[0], off[1]:off[1] + sh[1]] = frame
        return out

    def _draw(self, frames):
        # add background noise
        res = frames + np.random.normal(loc=self.offset, scale=self.io_noise, size=frames.shape).astype(int)
        # cut at saturation
        res[res > 63000] = 63000
        return res

    def _frame2bytes(self, frame):
        c = self.fCCD
        # scramble
        res = c._rawXclock(c._clockXrow(c._rowXccd(frame)))
        # convert to byte stream, assure uint16. Cut off a bit at the end and attach ending message
        res = res.astype(np.uint16).byteswap().tobytes()[:-200] + self.end_of_frame_msg
        return res

    def listen_for_greeting(self):
        """Open a socket for sending UDP packets to the framegrabber."""

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.udp_address)
        print("Socket open on (%s) on %d " % self.udp_address)

        print("Awaiting Handshake")
        data, addr = self.sock.recvfrom(255)

        print("Received %s from (%s) on %d,  " % (str(data), addr[0], addr[1]))
        self.fg_addr = addr

        self.sock.connect(addr)

    def make_header(self, port, length, frame_number, packet_number):
        """Returns header."""
        header = struct.pack('!BBHHH', packet_number, 0, port, length, frame_number)
        return header

    def udpsend(self, packet):
        # self.sock.sendto(packet,self.fg_addr)
        self.sock.send(packet)

    def send_frame_in_udp_packets(self, frame, frame_number):
        """Chop frame into small UDP packets and send them out through connected socket."""
        frame = self._frame2bytes(frame)
        psize = self.udp_packet_size - self.udp_header_size
        print(frame_number, (len(frame) // psize + 1))
        ip, port = self.udp_address
        try:
            for i in range(len(frame) // psize + 1):
                time.sleep(self.delay)
                h = self.make_header(port, psize + self.udp_header_size, frame_number, i % 256)
                packet = h + frame[i * psize:(i + 1) * psize]
                self.udpsend(packet)

            # optional blurbs / hickups
            packet = self.make_header(port, 48, frame_number, 0) + 32 * b"\x00" + self.end_of_frame_msg
            self.udpsend(packet)
            self.udpsend(packet)
        except socket.error:
            print("Connection error. Restarting ...")
            self.sock.close()
            self.listen_for_greeting()

    def run(self, num=1, triggers=20):
        """This triggers the event loop."""
        j = 0
        t0 = datetime.datetime.now()
        conn = self.listen_for_greeting()
        # Start the event loop
        for scan_num in range(num):

            # pause for moving in the detector
            print("Moving in CCD detector")
            time.sleep(.3)

            j = 0
            print("Closing shutter")
            time.sleep(.2)

            # dark frames
            print("Taking dark frames")
            time.sleep(.5)

            for k, frame in enumerate(self.dark_frames):
                self.send_frame_in_udp_packets(frame, j + k)

            j += k
            # actual frames
            print("Opening shutter")
            time.sleep(.2)

            print("Taking exp frames")
            time.sleep(.5)

            for k, frame in enumerate(self.exp_frames):
                self.send_frame_in_udp_packets(frame, j + k)

            j += k

            print("Moving detector out")
            time.sleep(.3)

            # the regular trigger pulse of the the system
            for tr in range(triggers):
                self.send_frame_in_udp_packets(self.testframe, j)
                time.sleep(0.5)
                j += 1
            # Update counters
            j += 1

if __name__ == '__main__':
    S = FccdSimulator()
    S.make_ptycho_data()
    S.run()
