#include "noodle_s3_nhwc_fp32.h"
#include <string.h>
#include <float.h>
#include <math.h>
namespace {
constexpr uint32_t NONE=0xffffffffu;
inline bool span(uint32_t off,size_t bytes,size_t cap){return off!=NONE && off<=cap && bytes<=cap-off && !(off&3u);}
inline bool good(const NoodleS3F32Tensor &t){return t.data && t.elements==uint32_t(t.h)*t.w*t.c;}
inline uint32_t idx(const NoodleS3F32Tensor &t,int y,int x,int c){return (uint32_t(y)*t.w+x)*t.c+c;}
static bool conv(const NoodleS3F32Op &op,const NoodleS3F32Tensor &a,NoodleS3F32Tensor &b,const uint8_t *p,size_t limit){
  uint32_t wc=uint32_t(b.c)*op.kh*op.kw*a.c;
  if(!op.kh||!op.kw||!op.sh||!op.sw||!span(op.weights,size_t(wc)*4,limit)||!span(op.bias,size_t(b.c)*4,limit))return false;
  auto weights=reinterpret_cast<const float*>(p+op.weights),bias=reinterpret_cast<const float*>(p+op.bias);
  // SAME and VALID shapes both derive the required spatial left/top padding.
  const int dy=(int(b.h)-1)*op.sh+op.kh-a.h,dx=(int(b.w)-1)*op.sw+op.kw-a.w;
  const int py=(dy>0?dy:0)/2,px=(dx>0?dx:0)/2;
  for(int y=0;y<b.h;++y)for(int x=0;x<b.w;++x){
    int iy0=y*op.sh-py,ix0=x*op.sw-px;
    for(int oc=0;oc<b.c;++oc){
      float v=bias[oc];
      for(int ky=0;ky<op.kh;++ky){int iy=iy0+ky;if(iy<0||iy>=a.h)continue;
        for(int kx=0;kx<op.kw;++kx){int ix=ix0+kx;if(ix<0||ix>=a.w)continue;
          const float* in=a.data+idx(a,iy,ix,0);
          const float* wt=weights+((uint32_t(oc)*op.kh+ky)*op.kw+kx)*a.c;
          for(int ic=0;ic<a.c;++ic)v+=in[ic]*wt[ic];
        }
      }
      b.data[idx(b,y,x,oc)]=op.relu && v<0.f?0.f:v;
    }
  }
  return true;
}
static bool pool(const NoodleS3F32Op &op,const NoodleS3F32Tensor &a,NoodleS3F32Tensor &b){
  if(a.c!=b.c||!op.kh||!op.kw||!op.sh||!op.sw)return false;
  const int dh=(int(b.h)-1)*op.sh+op.kh-a.h,dw=(int(b.w)-1)*op.sw+op.kw-a.w;
  const int py=(dh>0?dh:0)/2,px=(dw>0?dw:0)/2;
  for(int y=0;y<b.h;++y)for(int x=0;x<b.w;++x)for(int c=0;c<b.c;++c){
    float best=-FLT_MAX;
    for(int ky=0;ky<op.kh;++ky)for(int kx=0;kx<op.kw;++kx){
      int iy=y*op.sh+ky-py,ix=x*op.sw+kx-px;
      if(iy>=0&&iy<a.h&&ix>=0&&ix<a.w){float v=a.data[idx(a,iy,ix,c)];if(v>best)best=v;}
    }
    b.data[idx(b,y,x,c)]=best;
  }
  return true;
}
static bool avgpool(const NoodleS3F32Op &op,const NoodleS3F32Tensor &a,NoodleS3F32Tensor &b){
 if(a.c!=b.c||!op.kh||!op.kw||!op.sh||!op.sw)return false;
 const int dh=(int(b.h)-1)*op.sh+op.kh-a.h,dw=(int(b.w)-1)*op.sw+op.kw-a.w;
 const int py=(dh>0?dh:0)/2,px=(dw>0?dw:0)/2;
 for(int y=0;y<b.h;++y)for(int x=0;x<b.w;++x)for(int c=0;c<b.c;++c){
  float sum=0.f;int count=0;
  for(int ky=0;ky<op.kh;++ky)for(int kx=0;kx<op.kw;++kx){
   int iy=y*op.sh+ky-py,ix=x*op.sw+kx-px;
   if(iy>=0&&iy<a.h&&ix>=0&&ix<a.w){sum+=a.data[idx(a,iy,ix,c)];++count;}
  }
  if(count==0)return false;
  float v=sum/float(count);
  b.data[idx(b,y,x,c)]=op.relu && v<0.f?0.f:v;
 }
 return true;
}
static bool fcn(const NoodleS3F32Op &op,const NoodleS3F32Tensor &a,NoodleS3F32Tensor &b,const uint8_t*p,size_t limit){
 if(!b.elements||!a.elements||!span(op.weights,size_t(a.elements)*b.elements*4,limit)||!span(op.bias,size_t(b.elements)*4,limit))return false;
 const float *wt=reinterpret_cast<const float*>(p+op.weights),*bias=reinterpret_cast<const float*>(p+op.bias);
 for(uint32_t j=0;j<b.elements;++j){
  float sum=bias[j];const float*row=wt+size_t(j)*a.elements;
  for(uint32_t i=0;i<a.elements;++i)sum+=row[i]*a.data[i];
  b.data[j]=op.relu && sum<0.f?0.f:sum;
 }
 return true;
}
static bool softmax(const NoodleS3F32Tensor&a,NoodleS3F32Tensor&b){
 if(a.elements!=b.elements||a.elements==0)return false;
 float maximum=-FLT_MAX;
 for(uint32_t i=0;i<a.elements;++i)if(a.data[i]>maximum)maximum=a.data[i];
 float sum=0.f;
 for(uint32_t i=0;i<a.elements;++i){float v=expf(a.data[i]-maximum);b.data[i]=v;sum+=v;}
 if(!(sum>0.f))return false;
 for(uint32_t i=0;i<a.elements;++i)b.data[i]/=sum;
 return true;
}
static bool concat(const NoodleS3F32Tensor&a,const NoodleS3F32Tensor&b,NoodleS3F32Tensor&d){
  if(a.h!=b.h||a.w!=b.w||d.h!=a.h||d.w!=a.w||d.c!=a.c+b.c)return false;
  for(uint32_t i=0;i<uint32_t(d.h)*d.w;++i){
    memcpy(d.data+i*d.c,a.data+i*a.c,size_t(a.c)*4);
    memcpy(d.data+i*d.c+a.c,b.data+i*b.c,size_t(b.c)*4);
  }
  return true;
}
}
bool noodle_s3_f32_execute(const NoodleS3F32Op &op,NoodleS3F32Tensor*t,size_t n,const uint8_t*xip,size_t bytes){
  if(!t||!xip||op.src0>=n||op.dst>=n||!good(t[op.src0])||!good(t[op.dst]))return false;
  const auto&a=t[op.src0];auto&d=t[op.dst];
  switch(op.kind){
    case NDL_F32_CONV:return conv(op,a,d,xip,bytes);
    case NDL_F32_MAXPOOL:return pool(op,a,d);
    case NDL_F32_AVGPOOL:return avgpool(op,a,d);
    case NDL_F32_FCN:return fcn(op,a,d,xip,bytes);
    case NDL_F32_SOFTMAX:return softmax(a,d);
    case NDL_F32_CONCAT:return op.src1<n && good(t[op.src1]) && concat(a,t[op.src1],d);
    case NDL_F32_SUB:case NDL_F32_MUL:{
      if(a.elements!=d.elements||a.c!=3||!span(op.weights,3*4,bytes))return false;
      const float *v=reinterpret_cast<const float*>(xip+op.weights);
      for(uint32_t i=0;i<a.elements;++i)d.data[i]=op.kind==NDL_F32_SUB?a.data[i]-v[i%3]:a.data[i]*v[i%3];
      return true;
    }
    case NDL_F32_MEAN:{
      if(a.c!=d.c||d.h!=1||d.w!=1)return false;
      for(int c=0;c<a.c;++c){float sum=0.f;for(uint32_t pix=0;pix<uint32_t(a.h)*a.w;++pix)sum+=a.data[pix*a.c+c];d.data[c]=sum/(float(a.h)*a.w);}return true;
    }
    case NDL_F32_RESHAPE: if(a.elements!=d.elements)return false;memmove(d.data,a.data,size_t(a.elements)*4);return true;
    default:return false;
  }
}
