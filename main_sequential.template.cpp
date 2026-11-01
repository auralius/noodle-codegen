// R15 v0.6.2 Noodle-runtime-first LeNet5 on Uno - EXPERIMENTAL
// Static topology. All arithmetic uses noodle_i8_* operators (single SdFat handle).
// SD inputs and outputs are preallocated at Codegen export time.
// Generated from the original TFLite INT8 weights and quantization metadata.
#include <Arduino.h>
#include <noodle.h>
#include <noodle_int8.h>
#include "generated_params.h"
#include <limits.h>
#if defined(__AVR_ATmega328P__)
#include "bare_uart.h"
#define Serial r15_uart
#endif

// Two tensor descriptors follow the complete topology. After SD flattening,
// they are rebound as externally owned SRAM views for FC1/FC2/FC3/Softmax.
// Each operator still receives two distinct input/output descriptors.
static NoodleTensor tensor_a;
static NoodleTensor tensor_b;
// Fixed 150-byte work array handles the largest 5x5x6 convolution filter.
static int8_t r15_filter[150];
static uint8_t r15_io[32];
// Lifetime-specific SRAM reuse: Conv filter bytes are dead before FC1 starts;
// FC1 output is dead before FC3 overwrites it; I/O scratch is dead at Softmax.
// Descriptors are rebound after their old values die. The two activation
// buffers alternate during the dense stages; serial/SD I/O uses r15_io.
static int8_t r15_hidden84[84];
static void r15_bind_scratch(NoodleTensor &t,int8_t *buffer,uint16_t cap,uint16_t n,float scale,int32_t zp){
  // These are non-owning views into persistent scratch. Do not free or grow them.
  noodle_buffer_bind_external(&t.buffer,buffer,cap);
  t.C=n;t.W=1;t.rank=NOODLE_TENSOR_1D;t.scale=scale;t.zero_point=zp;
  t.file_path=nullptr;
}
static void r15_bind_scratch(NoodleTensor &t,uint8_t *buffer,uint16_t cap,uint16_t n,float scale,int32_t zp){
  r15_bind_scratch(t,(int8_t*)buffer,cap,n,scale,zp);
}

// Parameter bindings only. All operator arithmetic runs in Noodle.
// One reusable Conv configuration and one reusable FCNFile configuration.
// Layers execute sequentially; nothing keeps a reference after the call returns.
static Conv conv_cfg;
static FCNFile fcn_cfg;
static NoodleI8Pool mean_pool;
static void bind_pool_parameters(){
  mean_pool.M=2;mean_pool.T=2;mean_pool.mode=NOODLE_I8_POOL_MEAN;
}
static void bind_conv1(){
  @@BIND_CONV1@@
}
static void bind_conv2(){
  @@BIND_CONV2@@
}
static void bind_fc1(){
  @@BIND_FC1@@
}
static void bind_fc2(){
  @@BIND_FC2@@
}
static void bind_fc3(){
  @@BIND_FC3@@
}
static bool init_storage(){
  noodle_tensor_init(&tensor_a);
  noodle_tensor_init(&tensor_b);
  // SRAM scratch is bound only after the SD stages have finished using these
  // descriptors. No arena allocation is required for the activation tensors.
  if(!noodle_i8_tensor_bind_file(&tensor_a,"@@PATH_INPUT@@")){Serial.println(F("ERR_INIT_BIND_A"));return false;}
  if(!noodle_i8_tensor_bind_file(&tensor_b,"@@PATH_POOL1@@")){Serial.println(F("ERR_INIT_BIND_B"));return false;}
  return true;
}
// The input SD file MUST be open before the RDYIMG response. Otherwise the
// host can fill the Uno's 32-byte RX FIFO during a blocking SdFat open.
// Python sends the next block only after the single-byte K acknowledgment.
static bool receive_input(NDL_File &f){
  for(uint16_t start=0;start<784;start+=32){
    const uint8_t len=(784-start<32)?(uint8_t)(784-start):32;
    for(uint8_t k=0;k<len;++k){
      const unsigned long began=millis();
      while(!Serial.available()){
        if(millis()-began>3000ul){
          Serial.print(F("ERR_INPUT_TIMEOUT_BLOCK "));
          Serial.print(start/32);
          Serial.print(F(" OFFSET "));
          Serial.print(k);
          return false;
        }
      }
      const int b=Serial.read();
      r15_io[k]=(uint8_t)(int8_t)(b-128);
    }
    const size_t n=f.write(r15_io,len);
    if(n!=len){
      Serial.print(F("ERR_INPUT_WRITE_BLOCK "));
      Serial.print(start/32);
      Serial.print(F(" WRITTEN "));
      Serial.print((unsigned)n);
      return false;
    }
    if(len==32)Serial.write('K');
  }
  // Caller closes the file before starting the convolution stages.
  // tensor_a carried logits from the previous inference, if any.
  if(!noodle_i8_tensor_bind_file(&tensor_a,"@@PATH_INPUT@@") ||
     !noodle_i8_tensor_file_2d(&tensor_a,1,28,@@INPUT_SCALE@@f,@@INPUT_ZP@@)){
    Serial.println(F("ERR_INPUT_TENSOR_BIND"));
    return false;
  }
  return true;
}

static bool run_user_topology(){
  // SD: in.bin (A) -> p1.bin (B) -> p2.bin (A) -> flat.bin (B).
  // File names differ for every operation, but only two descriptors exist.
  if(!noodle_i8_tensor_bind_file(&tensor_b,"@@PATH_POOL1@@")) return false;
  Serial.println(F("STAGE CONV1_POOL1"));
  bind_conv1();
  if (noodle_i8_conv2d_fused(&tensor_a,conv_cfg,mean_pool,&tensor_b)!=14)return false;
  if(!noodle_i8_tensor_bind_file(&tensor_a,"@@PATH_POOL2@@")) return false;
  Serial.println(F("STAGE CONV2_POOL2"));
  bind_conv2();
  if (noodle_i8_conv2d_fused(&tensor_b,conv_cfg,mean_pool,&tensor_a)!=5)return false;
  if(!noodle_i8_tensor_bind_file(&tensor_b,"@@PATH_RESHAPE@@")) return false;
  Serial.println(F("STAGE RESHAPE"));
  if (noodle_i8_flat_file(&tensor_a,&tensor_b)!=400)return false;

  // The last SD input remains live in tensor_b during FC1. Tensor_a is dead,
  // so rebind it to the 150-byte Conv filter, now free to serve as SRAM output.
  r15_bind_scratch(tensor_a,r15_filter,sizeof(r15_filter),120,@@H1_SCALE@@f,@@H1_ZP@@);
  Serial.println(F("STAGE FC1"));
  bind_fc1();
  if (noodle_i8_fcn(&tensor_b,fcn_cfg,&tensor_a)!=120)return false;

  // The flattened SD input is dead after FC1. Rebind tensor_b to hidden84.
  r15_bind_scratch(tensor_b,r15_hidden84,sizeof(r15_hidden84),84,@@H2_SCALE@@f,@@H2_ZP@@);
  Serial.println(F("STAGE FC2"));
  bind_fc2();
  if (noodle_i8_fcn(&tensor_a,fcn_cfg,&tensor_b)!=84)return false;
  Serial.println(F("STAGE FC3"));
  bind_fc3();
  if (noodle_i8_fcn(&tensor_b,fcn_cfg,&tensor_a)!=10)return false;
  Serial.println(F("STAGE SOFTMAX"));
  @@SOFTMAX_EXEC@@
}
void setup(){
  Serial.begin(115200);
  bind_pool_parameters();
  noodle_i8_avr_bind_scratch(r15_filter,sizeof(r15_filter),r15_io,sizeof(r15_io));
  if(!init_storage()){Serial.println(F("ERR_INIT_MEMORY"));while(true){}}
  if(!noodle_fs_init(10)){Serial.println(F("ERR_SD_INIT"));while(true){}}

  Serial.println(F("READY"));
}
void loop(){
  static uint8_t cmd=0;
  while(Serial.available()){
    const char ch=(char)Serial.read();
    const char expected[]={'I','M','G','\n'};
    if(ch==expected[cmd])++cmd;else cmd=(ch=='I'?1:0);
    if(cmd<4)continue;
    cmd=0;
    // Pre-open input file BEFORE telling the host to transmit any bytes.
    // This prevents input transfer while SdFat opens the SD file.

    NDL_File input_file;


    // The 784-byte file must already exist on the SD card.
    // Open for writing, but do not create/truncate it (lower SdFat stack cost).
    const bool opened = noodle_fs_open_write_only_existing_into(input_file, "@@PATH_INPUT@@");


    if (!opened) {
      Serial.println(F("ERR_INPUT_OPEN"));
      Serial.println(F("READY"));
      return;
    }

    // Keep the validated initial 32-byte write and rewind before UART transfer.
    // No new buffer is allocated; r15_io is the shared I/O scratch.
    for (uint8_t i=0; i<sizeof(r15_io); ++i) r15_io[i]=0x80;
    if (input_file.write(r15_io,sizeof(r15_io)) != sizeof(r15_io) ||
        !input_file.seekSet(0)) {
      input_file.close();
      Serial.println(F("ERR_INPUT_PREPARE"));
      Serial.println(F("READY"));
      return;
    }

    // Keep input_file open while receive_input() writes its 32-byte blocks.
    Serial.println(F("RDYIMG"));
    const bool input_ok=receive_input(input_file);
    input_file.close();
    if(!input_ok){Serial.println(F("ERR_INPUT"));Serial.println(F("READY"));return;}
    const unsigned long start=millis();
    if(!run_user_topology()){Serial.println(F("ERR_INFERENCE"));Serial.println(F("READY"));return;}
    const unsigned long dt=millis()-start;
    uint16_t index=0;int8_t value=0;
    noodle_i8_find_max(&@@TERMINAL@@,&index,&value);
    Serial.print(F("PRED "));Serial.print(index);Serial.print(F(" MS "));Serial.println(dt);
    Serial.print(F("OUTPUT_INT8"));
    for(uint8_t i=0;i<10;++i){Serial.print(' ');Serial.print((int)@@TERMINAL@@.buffer.data[i]);}
    Serial.println();Serial.println(F("READY"));
    return;
  }
}
