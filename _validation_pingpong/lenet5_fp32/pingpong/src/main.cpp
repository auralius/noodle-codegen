// Noodle Studio ESP32-S3 generic FP32, serial test.
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_partition.h>
#include <esp_spi_flash.h>
#include <string.h>
#include <math.h>
#include <noodle_s3_nhwc_fp32.h>

static constexpr uint32_t EXPECTED_BYTES=246840u;
static constexpr uint32_t EXPECTED_CRC=0x6a228e5du;
static constexpr uint32_t INPUT_IDX=0u,OUTPUT_IDX=9u;
static constexpr uint32_t SERIAL_BYTES=3136u,CHUNK=32u;
static const char *opNames[]={"conv1","avgpool1","conv2","avgpool2","reshape1","fc1","fc2","fc3","softmax1"};
static NoodleS3F32Tensor tensors[]={
  {nullptr, 784u, 28, 28, 1}, // input1
  {nullptr, 4704u, 28, 28, 6}, // tmp1
  {nullptr, 1176u, 14, 14, 6}, // tmp2
  {nullptr, 1600u, 10, 10, 16}, // tmp3
  {nullptr, 400u, 5, 5, 16}, // tmp4
  {nullptr, 400u, 1, 1, 400}, // tmp5
  {nullptr, 120u, 1, 1, 120}, // hidden120
  {nullptr, 84u, 1, 1, 84}, // hidden84
  {nullptr, 10u, 1, 1, 10}, // logits10
  {nullptr, 10u, 1, 1, 10}, // probabilities10
};
static const uint32_t offsets[]={0u,4704u,0u,4704u,0u,4704u,0u,4704u,0u,4704u};
static const NoodleS3F32Op operators[]={
  {1, 1, 0, 0, 1, 5, 5, 1, 1, 16u, 616u}, // conv1
  {8, 0, 1, 0, 2, 2, 2, 2, 2, 4294967295u, 4294967295u}, // avgpool1
  {1, 1, 2, 0, 3, 5, 5, 1, 1, 640u, 10240u}, // conv2
  {8, 0, 3, 0, 4, 2, 2, 2, 2, 4294967295u, 4294967295u}, // avgpool2
  {7, 0, 4, 0, 5, 0, 0, 1, 1, 4294967295u, 4294967295u}, // reshape1
  {9, 1, 5, 0, 6, 0, 0, 1, 1, 10304u, 202304u}, // fc1
  {9, 1, 6, 0, 7, 0, 0, 1, 1, 202784u, 243104u}, // fc2
  {9, 0, 7, 0, 8, 0, 0, 1, 1, 243440u, 246800u}, // fc3
  {10, 0, 8, 0, 9, 0, 0, 1, 1, 4294967295u, 4294967295u}, // softmax1
};
static uint8_t*arena=nullptr;
static const uint8_t*flash=nullptr;
static spi_flash_mmap_handle_t mappedHandle=0;
static bool initialized=false;
static uint32_t crc32(const uint8_t*data,size_t len){uint32_t c=0xffffffffu;for(size_t i=0;i<len;++i){c^=data[i];for(int bit=0;bit<8;++bit)c=(c>>1)^((c&1u)?0xedb88320u:0u);}return ~c;}
static bool setupMemory(){
 if(!psramFound()){Serial.println("ERR_PSRAM_MISSING");return false;}
 Serial.printf("PSRAM total=%u free=%u\n",(unsigned)ESP.getPsramSize(),(unsigned)ESP.getFreePsram());
 const auto*part=esp_partition_find_first(ESP_PARTITION_TYPE_DATA,(esp_partition_subtype_t)0x40,"model");
 if(!part||part->size<EXPECTED_BYTES){Serial.println("ERR_MODEL_PARTITION");return false;}
 const void*mapped=nullptr;
 if(esp_partition_mmap(part,0,EXPECTED_BYTES,SPI_FLASH_MMAP_DATA,&mapped,&mappedHandle)!=ESP_OK){Serial.println("ERR_FLASH_MMAP");return false;}
 flash=(const uint8_t*)mapped;
 if(memcmp(flash,"NDLFXIP",7)!=0||flash[7]!=0||(uint32_t(flash[8])|(uint32_t(flash[9])<<8)|(uint32_t(flash[10])<<16)|(uint32_t(flash[11])<<24))!=EXPECTED_BYTES){Serial.println("ERR_MODEL_HEADER");return false;}
 if(crc32(flash+16,EXPECTED_BYTES-16)!=EXPECTED_CRC){Serial.println("ERR_MODEL_CRC");return false;}
 arena=(uint8_t*)heap_caps_malloc(23520u,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
 if(!arena){Serial.println("ERR_PSRAM_ARENA_ALLOC");return false;}
 for(size_t k=0;k<sizeof(tensors)/sizeof(*tensors);++k)tensors[k].data=(float*)(arena+offsets[k]);
 Serial.printf("TENSORS=%u OPS=%u PARAMS=%u ARENA=%u FREE_PSRAM=%u\n",(unsigned)(sizeof(tensors)/sizeof(*tensors)),(unsigned)(sizeof(operators)/sizeof(*operators)),(unsigned)EXPECTED_BYTES,(unsigned)23520u,(unsigned)ESP.getFreePsram());
 return true;
}
static bool receiveInput(){
 float*input=tensors[INPUT_IDX].data;
 uint8_t*raw=(uint8_t*)input;
 Serial.println("RDYIMG");
 uint32_t off=0;
 while(off<SERIAL_BYTES){
  uint32_t n=SERIAL_BYTES-off<CHUNK?SERIAL_BYTES-off:CHUNK,k=0;
  uint32_t last=millis();
  while(k<n){int b=Serial.read();if(b>=0){raw[off+k++]=(uint8_t)b;last=millis();}else if(millis()-last>15000){Serial.println("ERR_INPUT_TIMEOUT");return false;}else delay(1);}
  off+=n;if(n==CHUNK){Serial.write('K');Serial.flush();}
 }
 // Raw little-endian FLOAT32 tensor bytes: no model-specific RGB normalization.
 return true;
}
static void runModel(){
 uint32_t t0=millis();
 for(size_t i=0;i<sizeof(operators)/sizeof(*operators);++i){
  uint32_t step=millis();
  if(!noodle_s3_f32_execute(operators[i],tensors,sizeof(tensors)/sizeof(*tensors),flash,EXPECTED_BYTES)){
   Serial.printf("ERR_INFERENCE_OP %u %s\n",(unsigned)i,opNames[i]);Serial.println("READY");return;
  }
  Serial.printf("OP %u %-15s %lu ms\n",(unsigned)i,opNames[i],(unsigned long)(millis()-step));
 }
 auto&out=tensors[OUTPUT_IDX];uint32_t best=0;
 for(uint32_t k=1;k<out.elements;++k)if(out.data[k]>out.data[best])best=k;
 Serial.printf("PRED %u MS %lu\n",(unsigned)best,(unsigned long)(millis()-t0));
 Serial.print("OUTPUT_FLOAT32");
 for(uint32_t k=0;k<out.elements;++k){Serial.write(' ');Serial.print(out.data[k],7);}Serial.println();Serial.println("READY");
}
void setup(){
 Serial.begin(115200);delay(1200);Serial.println("NOODLE_GENERIC_FP32_N16R8");
 initialized=setupMemory();Serial.println(initialized?"READY":"BOOT_FAILED");
}
void loop(){
 if(!initialized){delay(1000);return;}
 static char cmd[12];static uint8_t used=0;
 while(Serial.available()){
  int c=Serial.read();if(c==13)continue;
  if(c=='\n'){cmd[used]=0;used=0;
   if(strcmp(cmd,"HELLO")==0)Serial.println("READY");
   else if(strcmp(cmd,"IMG")==0){if(receiveInput())runModel();else Serial.println("READY");}
   else{Serial.println("ERR_CMD");Serial.println("READY");}
  }else if(used<sizeof(cmd)-1)cmd[used++]=(char)c;
  else{used=0;Serial.println("ERR_CMD_LENGTH");Serial.println("READY");}
 }
 delay(1);
}
