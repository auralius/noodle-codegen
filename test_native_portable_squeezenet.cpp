#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <iterator>
#include "noodle_i8_nhwc.h"

static const uint32_t bufferSizes[]={788544u,387200u,193600u};
static const uint16_t tensorBufferIds[]={0,1,0,1,0,1,0,1,2,0,1,2,0,1,2,1,2,0,1,2,1,0,2,1,2,1,0,2,1,2,0,1,2,1,0,2,1,2,0,1,2,1,2,1};
static NoodleTensor physicalTensors[sizeof(bufferSizes)/sizeof(*bufferSizes)]={};
static NoodleI8NHWCTensor tensors[]={
  {nullptr, 150528u, 224u, 224u, 3u, 0, 3.921568859e-3f, true}, // input3
  {nullptr, 150528u, 224u, 224u, 3u, -128, 3.921568859e-3f, false}, // tmp1
  {nullptr, 150528u, 224u, 224u, 3u, -13, 4.231372382e-3f, false}, // tmp2
  {nullptr, 150528u, 224u, 224u, 3u, -14, 1.865844801e-2f, false}, // tmp3
  {nullptr, 788544u, 111u, 111u, 64u, -128, 7.200364769e-2f, false}, // tmp4
  {nullptr, 193600u, 55u, 55u, 64u, -128, 7.200364769e-2f, false}, // tmp5
  {nullptr, 48400u, 55u, 55u, 16u, -128, 1.453656256e-1f, false}, // tmp6
  {nullptr, 193600u, 55u, 55u, 64u, -128, 1.507989466e-1f, false}, // tmp7
  {nullptr, 193600u, 55u, 55u, 64u, -128, 1.507989466e-1f, false}, // tmp8
  {nullptr, 387200u, 55u, 55u, 128u, -128, 1.507989466e-1f, false}, // tmp9
  {nullptr, 48400u, 55u, 55u, 16u, -128, 2.309362143e-1f, false}, // tmp10
  {nullptr, 193600u, 55u, 55u, 64u, -128, 2.972879708e-1f, false}, // tmp11
  {nullptr, 193600u, 55u, 55u, 64u, -128, 2.972879708e-1f, false}, // tmp12
  {nullptr, 387200u, 55u, 55u, 128u, -128, 2.972879708e-1f, false}, // tmp13
  {nullptr, 93312u, 27u, 27u, 128u, -128, 2.972879708e-1f, false}, // tmp14
  {nullptr, 23328u, 27u, 27u, 32u, -128, 8.045994043e-1f, false}, // tmp15
  {nullptr, 93312u, 27u, 27u, 128u, -128, 6.423013210e-1f, false}, // tmp16
  {nullptr, 93312u, 27u, 27u, 128u, -128, 6.423013210e-1f, false}, // tmp17
  {nullptr, 186624u, 27u, 27u, 256u, -128, 6.423013210e-1f, false}, // tmp18
  {nullptr, 23328u, 27u, 27u, 32u, -128, 1.204994082e+0f, false}, // tmp19
  {nullptr, 93312u, 27u, 27u, 128u, -128, 9.627363682e-1f, false}, // tmp20
  {nullptr, 93312u, 27u, 27u, 128u, -128, 9.627363682e-1f, false}, // tmp21
  {nullptr, 186624u, 27u, 27u, 256u, -128, 9.627363682e-1f, false}, // tmp22
  {nullptr, 43264u, 13u, 13u, 256u, -128, 9.627363682e-1f, false}, // tmp23
  {nullptr, 8112u, 13u, 13u, 48u, -128, 2.578848600e+0f, false}, // tmp24
  {nullptr, 32448u, 13u, 13u, 192u, -128, 2.143035412e+0f, false}, // tmp25
  {nullptr, 32448u, 13u, 13u, 192u, -128, 2.143035412e+0f, false}, // tmp26
  {nullptr, 64896u, 13u, 13u, 384u, -128, 2.143035412e+0f, false}, // tmp27
  {nullptr, 8112u, 13u, 13u, 48u, -128, 3.610392094e+0f, false}, // tmp28
  {nullptr, 32448u, 13u, 13u, 192u, -128, 3.265003920e+0f, false}, // tmp29
  {nullptr, 32448u, 13u, 13u, 192u, -128, 3.265003920e+0f, false}, // tmp30
  {nullptr, 64896u, 13u, 13u, 384u, -128, 3.265003920e+0f, false}, // tmp31
  {nullptr, 10816u, 13u, 13u, 64u, -128, 3.625778198e+0f, false}, // tmp32
  {nullptr, 43264u, 13u, 13u, 256u, -128, 2.431860685e+0f, false}, // tmp33
  {nullptr, 43264u, 13u, 13u, 256u, -128, 2.431860685e+0f, false}, // tmp34
  {nullptr, 86528u, 13u, 13u, 512u, -128, 2.431860685e+0f, false}, // tmp35
  {nullptr, 10816u, 13u, 13u, 64u, -128, 2.933948517e+0f, false}, // tmp36
  {nullptr, 43264u, 13u, 13u, 256u, -128, 1.623080730e+0f, false}, // tmp37
  {nullptr, 43264u, 13u, 13u, 256u, -128, 1.623080730e+0f, false}, // tmp38
  {nullptr, 86528u, 13u, 13u, 512u, -128, 1.623080730e+0f, false}, // tmp39
  {nullptr, 169000u, 13u, 13u, 1000u, -128, 8.214161992e-1f, false}, // tmp40
  {nullptr, 1000u, 1u, 1u, 1000u, -128, 1.550151855e-1f, false}, // tmp41
  {nullptr, 1000u, 1u, 1u, 1000u, -128, 1.550151855e-1f, false}, // tmp42
  {nullptr, 1000u, 1u, 1u, 1000u, 0, 1.550151855e-1f, true}, // output1000
};
static const NoodleI8NHWCOp ops[]={
  {4, 0, 0, 0, 1, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // opquantize1
  {5, 0, 1, 0, 2, 0, 0, 1, 1, 16u, 4294967295u, 4294967295u, 4294967295u, 1.901960815e-3f, -128}, // opsub1
  {6, 0, 2, 0, 3, 0, 0, 1, 1, 20u, 4294967295u, 4294967295u, 4294967295u, 1.753740013e-2f, -128}, // opmul1
  {1, 1, 3, 0, 4, 3, 3, 2, 2, 24u, 1752u, 2008u, 2264u, 1.000000000e+0f, 0}, // conv1
  {2, 0, 4, 0, 5, 3, 3, 2, 2, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // pool1
  {1, 1, 5, 0, 6, 1, 1, 1, 1, 2520u, 3544u, 3608u, 3672u, 1.000000000e+0f, 0}, // conv2
  {1, 1, 6, 0, 7, 1, 1, 1, 1, 3736u, 4760u, 5016u, 5272u, 1.000000000e+0f, 0}, // conv3
  {1, 1, 6, 0, 8, 3, 3, 1, 1, 5528u, 14744u, 15000u, 15256u, 1.000000000e+0f, 0}, // conv4
  {3, 0, 7, 8, 9, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat1
  {1, 1, 9, 0, 10, 1, 1, 1, 1, 15512u, 17560u, 17624u, 17688u, 1.000000000e+0f, 0}, // conv5
  {1, 1, 10, 0, 11, 1, 1, 1, 1, 17752u, 18776u, 19032u, 19288u, 1.000000000e+0f, 0}, // conv6
  {1, 1, 10, 0, 12, 3, 3, 1, 1, 19544u, 28760u, 29016u, 29272u, 1.000000000e+0f, 0}, // conv7
  {3, 0, 11, 12, 13, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat2
  {2, 0, 13, 0, 14, 3, 3, 2, 2, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // pool2
  {1, 1, 14, 0, 15, 1, 1, 1, 1, 29528u, 33624u, 33752u, 33880u, 1.000000000e+0f, 0}, // conv8
  {1, 1, 15, 0, 16, 1, 1, 1, 1, 34008u, 38104u, 38616u, 39128u, 1.000000000e+0f, 0}, // conv9
  {1, 1, 15, 0, 17, 3, 3, 1, 1, 39640u, 76504u, 77016u, 77528u, 1.000000000e+0f, 0}, // conv10
  {3, 0, 16, 17, 18, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat3
  {1, 1, 18, 0, 19, 1, 1, 1, 1, 78040u, 86232u, 86360u, 86488u, 1.000000000e+0f, 0}, // conv11
  {1, 1, 19, 0, 20, 1, 1, 1, 1, 86616u, 90712u, 91224u, 91736u, 1.000000000e+0f, 0}, // conv12
  {1, 1, 19, 0, 21, 3, 3, 1, 1, 92248u, 129112u, 129624u, 130136u, 1.000000000e+0f, 0}, // conv13
  {3, 0, 20, 21, 22, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat4
  {2, 0, 22, 0, 23, 3, 3, 2, 2, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // pool3
  {1, 1, 23, 0, 24, 1, 1, 1, 1, 130648u, 142936u, 143128u, 143320u, 1.000000000e+0f, 0}, // conv14
  {1, 1, 24, 0, 25, 1, 1, 1, 1, 143512u, 152728u, 153496u, 154264u, 1.000000000e+0f, 0}, // conv15
  {1, 1, 24, 0, 26, 3, 3, 1, 1, 155032u, 237976u, 238744u, 239512u, 1.000000000e+0f, 0}, // conv16
  {3, 0, 25, 26, 27, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat5
  {1, 1, 27, 0, 28, 1, 1, 1, 1, 240280u, 258712u, 258904u, 259096u, 1.000000000e+0f, 0}, // conv17
  {1, 1, 28, 0, 29, 1, 1, 1, 1, 259288u, 268504u, 269272u, 270040u, 1.000000000e+0f, 0}, // conv18
  {1, 1, 28, 0, 30, 3, 3, 1, 1, 270808u, 353752u, 354520u, 355288u, 1.000000000e+0f, 0}, // conv19
  {3, 0, 29, 30, 31, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat6
  {1, 1, 31, 0, 32, 1, 1, 1, 1, 356056u, 380632u, 380888u, 381144u, 1.000000000e+0f, 0}, // conv20
  {1, 1, 32, 0, 33, 1, 1, 1, 1, 381400u, 397784u, 398808u, 399832u, 1.000000000e+0f, 0}, // conv21
  {1, 1, 32, 0, 34, 3, 3, 1, 1, 400856u, 548312u, 549336u, 550360u, 1.000000000e+0f, 0}, // conv22
  {3, 0, 33, 34, 35, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat7
  {1, 1, 35, 0, 36, 1, 1, 1, 1, 551384u, 584152u, 584408u, 584664u, 1.000000000e+0f, 0}, // conv23
  {1, 1, 36, 0, 37, 1, 1, 1, 1, 584920u, 601304u, 602328u, 603352u, 1.000000000e+0f, 0}, // conv24
  {1, 1, 36, 0, 38, 3, 3, 1, 1, 604376u, 751832u, 752856u, 753880u, 1.000000000e+0f, 0}, // conv25
  {3, 0, 37, 38, 39, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // concat8
  {1, 1, 39, 0, 40, 1, 1, 1, 1, 754904u, 1266904u, 1270904u, 1274904u, 1.000000000e+0f, 0}, // conv26
  {7, 0, 40, 0, 41, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // mean1
  {8, 0, 41, 0, 42, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // reshape1
  {4, 0, 42, 0, 43, 0, 0, 1, 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 1.000000000e+0f, 0}, // opquantize2
};
int main(int argc,char**argv){
  if(argc<3){std::fprintf(stderr,"usage: runner params.bin pattern\n");return 2;}
  std::ifstream f(argv[1],std::ios::binary);
  std::vector<uint8_t> weights((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
  if(weights.empty()){std::fprintf(stderr,"missing weights\n");return 2;}
  constexpr size_t tensorCount=sizeof(tensors)/sizeof(*tensors);
  std::vector<std::vector<uint8_t>> blocks;
  for(auto capacity: bufferSizes)blocks.emplace_back(capacity);
  for(size_t i=0;i<blocks.size();i++){
    if(!noodle_buffer_bind_external(&physicalTensors[i].buffer,reinterpret_cast<NoodleData*>(blocks[i].data()),bufferSizes[i]))return 3;
  }
  for(size_t i=0;i<tensorCount;i++)tensors[i].storage=&physicalTensors[tensorBufferIds[i]];

  auto &in=tensors[0];
  int pattern=std::atoi(argv[2]);
  for(size_t i=0;i<in.bytes;i++)noodle_i8_nhwc_data(in)[i]=(uint8_t)(pattern==0?0:pattern==1?(i*37u+11u)&255u:255);
  for(size_t i=0;i<sizeof(ops)/sizeof(*ops);i++){
    if(!noodle_i8_nhwc_execute(ops[i],tensors,tensorCount,weights.data(),weights.size())){std::fprintf(stderr,"operator fail %zu\n",i);return 1;}
    const auto &t=tensors[ops[i].dst];uint32_t hash=2166136261u;
    for(size_t j=0;j<t.bytes;j++)hash=(hash^noodle_i8_nhwc_data(t)[j])*16777619u;
    std::printf("%zu %u %08x\n",i,(unsigned)t.bytes,hash);
  }
  return 0;
}
