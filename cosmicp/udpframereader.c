#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sys/socket.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>

// Copy data and swap bytes. Both pointers must be 64bit aligned and size must be multiple of 8
static void memcpy_ntohs(void *restrict dst, const void *restrict src, size_t n){
  // 
  assert(n & 0x7 == 0);
  assert((uintptr_t)dst & (uintptr_t)0x7 == 0);
  assert((uintptr_t)src & (uintptr_t)0x7 == 0);
  uint64_t * restrict to = (uint64_t * restrict)dst;
  uint64_t * restrict from = (uint64_t * restrict)src;
  while(n >= 8){
    *to = ((*from << 8) & 0xFF00FF00FF00FF00) | ((*from >> 8) & 0x00FF00FF00FF00FF);
    to++;
    from++;
    n-=8;
  }
}


static void descramble(void *restrict frame8, size_t rows){
 const int des[192] = { 
    188, 172, 156, 140, 124, 108, 92, 76, 60, 44, 28, 12, 
    189, 173, 157, 141, 125, 109, 93, 77, 61, 45, 29, 13, 
    190, 174, 158, 142, 126, 110, 94, 78, 62, 46, 30, 14, 
    191, 175, 159, 143, 127, 111, 95, 79, 63, 47, 31, 15, 
    184, 168, 152, 136, 120, 104, 88, 72, 56, 40, 24, 8, 
    185, 169, 153, 137, 121, 105, 89, 73, 57, 41, 25, 9, 
    186, 170, 154, 138, 122, 106, 90, 74, 58, 42, 26, 10, 
    187, 171, 155, 139, 123, 107, 91, 75, 59, 43, 27, 11, 
    180, 164, 148, 132, 116, 100, 84, 68, 52, 36, 20, 4, 
    181, 165, 149, 133, 117, 101, 85, 69, 53, 37, 21, 5, 
    182, 166, 150, 134, 118, 102, 86, 70, 54, 38, 22, 6, 
    183, 167, 151, 135, 119, 103, 87, 71, 55, 39, 23, 7, 
    176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0, 
    177, 161, 145, 129, 113, 97, 81, 65, 49, 33, 17, 1, 
    178, 162, 146, 130, 114, 98, 82, 66, 50, 34, 18, 2, 
    179, 163, 147, 131, 115, 99, 83, 67, 51, 35, 19, 3};

  uint16_t * restrict frame = (uint16_t * restrict)frame8;
  //uint16_t * restrict from = (uint16_t * restrict)src;
  uint16_t row[192]; // = malloc(packet_size)
  for (size_t off=0;off < rows;off++){
    for (size_t n =0;n<192;n++){
        *(row+n) = *(frame+n+off*192) ;
    }
    for (size_t n =0;n<192;n++){
        *(frame+n+off*192) = *(row+des[n]);
    }
    //rest
    //for (int n =0;n<sze % 192;n++){
    //    *(dest+n+off) = *(src+des[n]+off) ;
    //}
  }
}


static int guess_packet_n(int packet_small_n, int prev_packet_n, int n_packets){
  // The package_small_n is only 8 bits. We'll have to
  // estimate the higher bits
  int packet_n = 0;
  if(prev_packet_n == -1){
    // We don't know any better so let's just assume the 
    // highs are 0
    packet_n = packet_small_n;
  }else if(abs(packet_small_n - (prev_packet_n % 256)) < 100){
    // The new low bits are similar to the previous.
    // Keep the same high bits
    packet_n = 256 * (prev_packet_n/256) + (packet_small_n);
  }else if((packet_small_n - (prev_packet_n % 256)) > 100){
    // The new low bits are much higher than the previous.
    // We probably got a few delayed packages from a previous batch
    packet_n = 256*(prev_packet_n/256 - 1) + packet_small_n;
  }else if((packet_small_n - (prev_packet_n % 256)) < -100){
    // The new low bits are much lower than the previous.
    // We probably got a few packages from a new batch
    packet_n = 256*(prev_packet_n/256 + 1) + packet_small_n;
  }
  while(packet_n < 0){
    packet_n += 256;
  }
  while(packet_n >= n_packets){
    packet_n -= 256;
  }
  return packet_n;
}

static PyObject *
ufr_read_frame(PyObject *self, PyObject *args)
{
  // We're going to assume all packets are at most 8192 bytes
  const int packet_size = 4104;
  const int header_size = 8;
  int sock;
  int size;
  //int exclude;
  PyObject* out;
  PyObject* frame_n_out;
  PyObject* bytes_in_frame_out;

  int status = 0;
  int bytes_in_frame = 0;

  // The frame we'll eventually return
  static uint8_t * restrict frame; 
  //static uint8_t * restrict spill = malloc(packet_size);
  static int use_old_buffer = 0;
  static int last_frame_n = -1;
   
  // The frame we're currently building
  int frame_n = -1;

  if (!PyArg_ParseTuple(args, "ii", &sock, &size))
    return NULL;
  Py_BEGIN_ALLOW_THREADS
  // The buffer where we first receive the data
  static uint8_t * restrict buffer;
    
  // The total number of packets we'll receive
  int n_packets = 1+(size-1)/(packet_size-header_size);
  // The number of packages we have received for this frame
  int packets_received = 0;
  int prev_packet_n = 0;
  int frame_allocated = 0;
  int buffer_allocated = 0;
  size_t rows = size / 384; //(12 * 960)
  while(1){
    ssize_t bytes_recv = 0;
    if (use_old_buffer==0){
        //printf("about to read\n");
        if (buffer_allocated == 0){
            //free(buffer);
            //printf("buffer allocated");
            buffer = malloc(packet_size);
            buffer_allocated = 1;
        }
        bytes_recv = recv(sock, buffer, packet_size, 0);
        //printf("read %ld\n", bytes_recv);
    }
    else {
        buffer_allocated = 1;
        use_old_buffer = 0;
        bytes_recv = packet_size;
    }
    if(bytes_recv < 0){
      // If we get an error during recv return None, error_value
      status = ntohs(bytes_recv);
      break;
    }

    int fn = ntohs(*(uint16_t*)(buffer+6));
    int packet_small_n = buffer[0];


    //printf("Packet\t Frame\t Bytes\n");
    //printf("%d\t %d\t %ld\n", packet_small_n, fn,bytes_recv);

    //if((bytes_recv != 8192) && (packets_received< n_packets/2)){
    if (fn == last_frame_n){
      // First packet must be full
      // But ideally we would just drop packets of a frame already pushed
      // out
      continue;
    }
    
    if(fn != frame_n){
      if(fn < frame_n && fn != 0){
        // We got some left overs from a previous frame. Ignore it.
        // printf("Got old packets for frame %d\n", fn);
      continue;
      }

      
      if (frame_allocated == 1){
        // packet of next frame. Push out current frame, remember that
        // the buffer holds actual data and should not be freed.
        use_old_buffer = 1;
        break;
      }
      else{
        // Allocate a new frame, this will be in-place as an overwrite!
        free(frame);
        frame = calloc(n_packets*(packet_size-header_size),sizeof(uint8_t));
        //printf("New frame %d at %16lu\n", fn,packet_small_n,frame);
        frame_n = fn;
        packets_received = 0;
        prev_packet_n = -1;
        frame_allocated = 1;
      }
    }


    int packet_n = guess_packet_n(packet_small_n, prev_packet_n, n_packets);
    prev_packet_n = packet_n;
    
    //if (packet_n == 1) {printf("Frame %d at %lu\n",fn,frame);}
    // Copy data appropriately, and swap endianess
    //int addr = frame+packet_n*(packet_size-header_size);
    //printf("Frame %d at %lu\n",fn,addr);
    memcpy_ntohs(frame+packet_n*(packet_size-header_size), buffer+header_size,bytes_recv-header_size);
    packets_received++;
    bytes_in_frame += bytes_recv-header_size;
    
    //printf("looking for packet %d max is %d - size = %d\n", packets_received, n_packets, size);
    if (bytes_recv < packet_size){
      // we should check here for DEADFOOD
      //printf("Out %d after %d packets because small packet at %16lu \n", fn, packets_received,frame);
      break;
    }
    if(packets_received == n_packets){
      // We are done!
      //printf("Break cause enough packets %d\n", packets_received);
      break;
    }
  } 
  if (use_old_buffer==0){
    free(buffer);
  }
  //printf("Descrambling!\n");
  descramble(frame,rows);
  Py_END_ALLOW_THREADS
  if(status == 0){
    //printf("End !\n");
    //printf("Read from %16lu\n",frame);
    //*(uint16_t*)(frame+10) = htons(frame_n);
    //printf("%lu %d\n" ,frame+10, ntohs(*(uint16_t*)(frame+10)));
    last_frame_n = frame_n;
    //out = PyBuffer_FromReadWriteMemory(frame, size);
    out = PyMemoryView_FromMemory(frame, size, PyBUF_WRITE);
    frame_n_out = Py_BuildValue("i", frame_n);
    bytes_in_frame_out = Py_BuildValue("i", bytes_in_frame);
    return PyTuple_Pack(3, out, frame_n_out, bytes_in_frame_out);
  }else{
    // If we get an error during recv return None, error_value
    //printf("Error\n");
    return PyTuple_Pack(3, Py_None, Py_BuildValue("i", status), Py_None);
  }
}

static PyMethodDef UDPReaderMethods[] = {
  {"read_frame",  ufr_read_frame, METH_VARARGS,
   "read_frame(socket_fd, frame_nbytes)\n\n"
   "Returns the first frame read from the given socket file descriptor.\n"
   "The function will not return until it manages to successfully\n"
   "read a whole frame without dropping any packets.\n"
   "\n"
   "Parameters\n"
   "----------\n"
   "socket_fd : int\n"
   "    The socket file descriptor usually obtained from socket.fileno()\n"
   "frame_nbytes : int\n"
   "    The size, in bytes, of the frame to read.\n"
   "\n"
   "Returns\n"
   "----------\n"
   "frame : buffer\n"
   "    The frame that was read or None in case of failure.\n"
   "    N.B.: This buffer is freed() in subsequent call to this function!\n"
   "frame_number : int\n"
   "    The frame number, according to the packet headers.\n"
   "byte_number : int\n"
   "    Number of bytes read into frame.\n"
  },
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

/*
PyMODINIT_FUNC
initudpframereader(void)
{
  (void) Py_InitModule("udpframereader", UDPReaderMethods);
}
*/

static struct PyModuleDef udpframereader =
{
    PyModuleDef_HEAD_INIT,
    "udpframereader", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    UDPReaderMethods
};

PyMODINIT_FUNC
PyInit_udpframereader(void)
{
    return PyModule_Create(&udpframereader);
}
