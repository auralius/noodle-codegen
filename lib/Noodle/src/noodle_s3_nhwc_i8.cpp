#include "noodle_s3_nhwc_i8.h"
#include "noodle_int8.h"
#include <math.h>
#include <string.h>
#include <limits.h>

namespace {
static constexpr uint32_t no_param = 0xffffffffu;
static inline int32_t getq(const NoodleS3Tensor &t, uint32_t i) {
  return t.is_uint8 ? t.data[i] : static_cast<int8_t>(t.data[i]);
}
static inline void setq(NoodleS3Tensor &t, uint32_t i, int32_t q) {
  if(t.is_uint8) { if(q<0)q=0; if(q>255)q=255; t.data[i]=static_cast<uint8_t>(q); }
  else { if(q< -128)q=-128; if(q>127)q=127; t.data[i]=static_cast<uint8_t>(static_cast<int8_t>(q)); }
}
static inline int32_t roundq(float v) {
  if(!isfinite(v)) return 0;
  if(v >= float(INT32_MAX)) return INT32_MAX;
  if(v <= float(INT32_MIN)) return INT32_MIN;
  return static_cast<int32_t>(nearbyintf(v));
}
static inline uint32_t count(const NoodleS3Tensor &t) {
  return uint32_t(t.h)*t.w*t.c;
}
static bool valid(const NoodleS3Tensor &t) {return t.data && t.scale > 0.0f && count(t) == t.bytes;}
static bool span(uint32_t off, uint32_t n, size_t size) {
  return off != no_param && off <= size && n <= size - off;
}
static bool requant(const NoodleS3Tensor &a, NoodleS3Tensor &b) {
  if(count(a)!=count(b))return false;
  const float ratio=a.scale/b.scale;
  for(uint32_t i=0;i<count(a);++i)
    setq(b,i,roundq(float(getq(a,i)-a.zero_point)*ratio)+b.zero_point);
  return true;
}
static bool binary(const NoodleS3Tensor &a, NoodleS3Tensor &b,
                   const int8_t *p, float scale, int32_t zp, bool mul) {
  if(count(a)!=count(b) || !p || a.c!=3 || b.c!=3)return false;
  for(uint32_t i=0;i<count(a);++i) {
    const float lhs=float(getq(a,i)-a.zero_point)*a.scale;
    const float rhs=float(int32_t(p[i%a.c])-zp)*scale;
    const float v=mul?lhs*rhs:lhs-rhs;
    setq(b,i,roundq(v/b.scale)+b.zero_point);
  }
  return true;
}
// Pointwise4: reuse each XIP 1x1 weight across four adjacent output pixels.
// The original scalar kernel remains available with NOODLE_S3_I8_POINTWISE_PIXELS=0.
// Normal build defaults to 4; all non-1x1 convolutions retain the original path.
#ifndef NOODLE_S3_I8_POINTWISE_PIXELS
#define NOODLE_S3_I8_POINTWISE_PIXELS 4
#endif
static bool pointwise_1x1(const NoodleS3Op &op, const NoodleS3Tensor &a,
                          NoodleS3Tensor &b, const int8_t *weights,
                          const int32_t *bias, const int32_t *mult,
                          const int32_t *shift) {
  // The caller invokes this only for 1x1 / stride-1 / same-size tensors.
  // Each output channel uses one contiguous input-channel weight vector.
  const uint32_t pixels = uint32_t(a.h) * a.w;
  const uint32_t inC = a.c, outC = b.c;
  const int32_t zp = a.zero_point;
  for (uint32_t oc = 0; oc < outC; ++oc) {
    const int8_t *w = weights + oc * inC;
#if NOODLE_S3_I8_POINTWISE_PIXELS == 4
    uint32_t p = 0;
    for (; p + 3 < pixels; p += 4) {
      const int8_t *i0 = reinterpret_cast<const int8_t*>(a.data + (p + 0) * inC);
      const int8_t *i1 = reinterpret_cast<const int8_t*>(a.data + (p + 1) * inC);
      const int8_t *i2 = reinterpret_cast<const int8_t*>(a.data + (p + 2) * inC);
      const int8_t *i3 = reinterpret_cast<const int8_t*>(a.data + (p + 3) * inC);
      int32_t s0 = bias[oc], s1 = s0, s2 = s0, s3 = s0;
      for (uint32_t ic = 0; ic < inC; ++ic) {
        const int32_t weight = int32_t(w[ic]);
        s0 += (int32_t(i0[ic]) - zp) * weight;
        s1 += (int32_t(i1[ic]) - zp) * weight;
        s2 += (int32_t(i2[ic]) - zp) * weight;
        s3 += (int32_t(i3[ic]) - zp) * weight;
      }
      const int32_t m = mult[oc], sh = shift[oc];
      int32_t q0 = noodle_i8_multiply_by_quantized_multiplier(s0, m, sh) + b.zero_point;
      int32_t q1 = noodle_i8_multiply_by_quantized_multiplier(s1, m, sh) + b.zero_point;
      int32_t q2 = noodle_i8_multiply_by_quantized_multiplier(s2, m, sh) + b.zero_point;
      int32_t q3 = noodle_i8_multiply_by_quantized_multiplier(s3, m, sh) + b.zero_point;
      if (op.relu) {
        if(q0 < b.zero_point) q0 = b.zero_point;
        if(q1 < b.zero_point) q1 = b.zero_point;
        if(q2 < b.zero_point) q2 = b.zero_point;
        if(q3 < b.zero_point) q3 = b.zero_point;
      }
      setq(b, (p+0)*outC+oc, q0);
      setq(b, (p+1)*outC+oc, q1);
      setq(b, (p+2)*outC+oc, q2);
      setq(b, (p+3)*outC+oc, q3);
    }
#else
    uint32_t p = 0;
#endif
    for (; p < pixels; ++p) {
      const int8_t *input = reinterpret_cast<const int8_t*>(a.data + p*inC);
      int32_t sum = bias[oc];
      for (uint32_t ic = 0; ic < inC; ++ic)
        sum += (int32_t(input[ic])-zp) * int32_t(w[ic]);
      int32_t q = noodle_i8_multiply_by_quantized_multiplier(sum,mult[oc],shift[oc])+b.zero_point;
      if(op.relu && q < b.zero_point) q=b.zero_point;
      setq(b,p*outC+oc,q);
    }
  }
  return true;
}

static bool convolution(const NoodleS3Op &op, const NoodleS3Tensor &a, NoodleS3Tensor &b,
                        const uint8_t *xip, size_t size) {
  if(a.is_uint8 || b.is_uint8 || !op.kh || !op.kw || !op.stride_h || !op.stride_w)return false;
  const uint32_t nweights=uint32_t(b.c)*op.kh*op.kw*a.c;
  if(!span(op.weights,nweights,size) || !span(op.bias,b.c*4u,size) ||
     !span(op.multipliers,b.c*4u,size)||!span(op.shifts,b.c*4u,size))return false;
  if((op.bias|op.multipliers|op.shifts)&3)return false;
  const int8_t *weights=reinterpret_cast<const int8_t*>(xip+op.weights);
  const int32_t *bias=reinterpret_cast<const int32_t*>(xip+op.bias);
  const int32_t *mult=reinterpret_cast<const int32_t*>(xip+op.multipliers);
  const int32_t *shift=reinterpret_cast<const int32_t*>(xip+op.shifts);
#if defined(NOODLE_S3_I8_POINTWISE_PIXELS) && NOODLE_S3_I8_POINTWISE_PIXELS > 0
  if(op.kh==1 && op.kw==1 && op.stride_h==1 && op.stride_w==1 &&
     a.h==b.h && a.w==b.w)
    return pointwise_1x1(op,a,b,weights,bias,mult,shift);
#endif
  const int32_t total_h=(int32_t(b.h)-1)*op.stride_h+op.kh-a.h;
  const int32_t total_w=(int32_t(b.w)-1)*op.stride_w+op.kw-a.w;
  const int32_t pad_h=(total_h>0?total_h:0)/2;
  const int32_t pad_w=(total_w>0?total_w:0)/2;
  for(uint16_t y=0;y<b.h;y++)for(uint16_t x=0;x<b.w;x++){
    const int32_t y0=int32_t(y)*op.stride_h-pad_h;
    const int32_t x0=int32_t(x)*op.stride_w-pad_w;
    for(uint16_t oc=0;oc<b.c;oc++){
      int32_t sum=bias[oc];
      for(uint16_t ky=0;ky<op.kh;ky++){
        const int32_t iy=y0+ky;if(iy<0||iy>=a.h)continue;
        for(uint16_t kx=0;kx<op.kw;kx++){
          const int32_t ix=x0+kx;if(ix<0||ix>=a.w)continue;
          const uint32_t ai=(uint32_t(iy)*a.w+ix)*a.c;
          const uint32_t wi=((uint32_t(oc)*op.kh+ky)*op.kw+kx)*a.c;
          for(uint16_t ic=0;ic<a.c;ic++){
            sum+=(int32_t(static_cast<int8_t>(a.data[ai+ic]))-a.zero_point)*int32_t(weights[wi+ic]);
          }
        }
      }
      int32_t q=noodle_i8_multiply_by_quantized_multiplier(sum,mult[oc],shift[oc])+b.zero_point;
      if(op.relu && q<b.zero_point)q=b.zero_point;
      setq(b,(uint32_t(y)*b.w+x)*b.c+oc,q);
    }
  }
  return true;
}
static bool maxpool(const NoodleS3Op &op,const NoodleS3Tensor &a,NoodleS3Tensor &b){
  if(a.c!=b.c || a.is_uint8!=b.is_uint8 || !op.kh || !op.kw || !op.stride_h || !op.stride_w || a.scale!=b.scale || a.zero_point!=b.zero_point)return false;
  const int32_t tot_h=(int32_t(b.h)-1)*op.stride_h+op.kh-a.h;
  const int32_t tot_w=(int32_t(b.w)-1)*op.stride_w+op.kw-a.w;
  const int32_t pad_h=(tot_h>0?tot_h:0)/2, pad_w=(tot_w>0?tot_w:0)/2;
  for(uint16_t y=0;y<b.h;y++)for(uint16_t x=0;x<b.w;x++)for(uint16_t c=0;c<b.c;c++){
    int32_t best=b.is_uint8?0:-128;
    for(uint16_t ky=0;ky<op.kh;ky++)for(uint16_t kx=0;kx<op.kw;kx++){
      const int32_t iy=int32_t(y)*op.stride_h+ky-pad_h;
      const int32_t ix=int32_t(x)*op.stride_w+kx-pad_w;
      if(iy<0||iy>=a.h||ix<0||ix>=a.w)continue;
      int32_t v=getq(a,(uint32_t(iy)*a.w+ix)*a.c+c);if(v>best)best=v;
    }
    setq(b,(uint32_t(y)*b.w+x)*b.c+c,best);
  }
  return true;
}
static bool concat(const NoodleS3Tensor &a,const NoodleS3Tensor &b,NoodleS3Tensor &dst){
  if(a.h!=b.h || a.w!=b.w || dst.h!=a.h || dst.w!=a.w || dst.c!=a.c+b.c ||
     a.is_uint8!=b.is_uint8 || dst.is_uint8!=a.is_uint8 ||
     a.zero_point!=b.zero_point || dst.zero_point!=a.zero_point ||
     a.scale!=b.scale || dst.scale!=a.scale)return false;
  for(uint32_t pix=0;pix<uint32_t(dst.h)*dst.w;pix++){
    memcpy(dst.data+pix*dst.c,a.data+pix*a.c,a.c);
    memcpy(dst.data+pix*dst.c+a.c,b.data+pix*b.c,b.c);
  }
  return true;
}
static bool meanhw(const NoodleS3Tensor &a,NoodleS3Tensor &dst){
  if(a.c!=dst.c || dst.h!=1 || dst.w!=1 || a.is_uint8 || dst.is_uint8)return false;
  const float ratio=a.scale/(float(a.h)*a.w*dst.scale);
  for(uint16_t c=0;c<a.c;c++){
    int32_t sum=0;
    for(uint16_t y=0;y<a.h;y++)for(uint16_t x=0;x<a.w;x++)
      sum+=getq(a,(uint32_t(y)*a.w+x)*a.c+c)-a.zero_point;
    setq(dst,c,roundq(float(sum)*ratio)+dst.zero_point);
  }
  return true;
}
}

bool noodle_s3_i8_execute(const NoodleS3Op &op,NoodleS3Tensor *t,size_t n,
                          const uint8_t *xip,size_t xip_size){
  if(!t || !xip || op.src0>=n || op.dst>=n)return false;
  NoodleS3Tensor &a=t[op.src0],&dst=t[op.dst];
  if(!valid(a)||!valid(dst))return false;
  switch(op.type){
    case NDL_S3_QUANTIZE:return requant(a,dst);
    case NDL_S3_SUB: case NDL_S3_MUL:
      return span(op.weights,a.c,xip_size) && binary(a,dst,reinterpret_cast<const int8_t*>(xip+op.weights),op.constant_scale,op.constant_zero_point,op.type==NDL_S3_MUL);
    case NDL_S3_CONV:return convolution(op,a,dst,xip,xip_size);
    case NDL_S3_MAXPOOL:return maxpool(op,a,dst);
    case NDL_S3_CONCAT:return op.src1<n && valid(t[op.src1]) && concat(a,t[op.src1],dst);
    case NDL_S3_MEAN:return meanhw(a,dst);
    case NDL_S3_RESHAPE:if(count(a)!=count(dst)||a.is_uint8!=dst.is_uint8||a.scale!=dst.scale||a.zero_point!=dst.zero_point)return false;
      memmove(dst.data,a.data,a.bytes);return true;
    default:return false;
  }
}
