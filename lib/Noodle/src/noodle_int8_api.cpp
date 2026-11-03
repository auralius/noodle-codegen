#include "noodle_config.h"
#if defined(NOODLE_USE_INT8)

#include "noodle_int8.h"

#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

void noodle_setup_temp_buffers(void *b1, void *b2) {
  (void)b1;
  (void)b2;
  // The full-int8 kernels own correctly typed int8/int32 scratch storage.
}

void noodle_setup_temp_buffers(void *b2) {
  (void)b2;
}

void noodle_temp_buffers_free(void) {
  noodle_i8_temp_buffers_free();
}

float *noodle_create_buffer(uint16_t size) {
  return (float *)malloc(size);
}

void noodle_delete_buffer(float *buffer) {
  free(buffer);
}


static NoodleI8Pool make_pool(const Pool &pool) {
  NoodleI8Pool p;
  p.M = pool.M;
  p.T = pool.T;
#if NOODLE_POOL_MODE == NOODLE_POOL_MEAN
  p.mode = NOODLE_I8_POOL_MEAN;
#else
  p.mode = NOODLE_I8_POOL_MAX;
#endif
  return p;
}

static bool identity_pool(const Pool &p) {
  return p.M == 1 && p.T == 1;
}

template <typename Layer>
static Layer with_activation_bounds(const Layer &src) {
  Layer dst = src;
  if (dst.act == ACT_RELU) {
    const int32_t qzero = noodle_activation_min_relu(
        dst.output_scale, dst.output_zero_point);
    if (dst.activation_min < qzero) dst.activation_min = qzero;
  }
  return dst;
}

static uint16_t copy_tensor_result(NoodleTensor *src, NoodleTensor *dst) {
  if (!src || !dst || !src->buffer.data) return 0;
  const size_t n = noodle_tensor_size(src);
  NoodleData *p = src->rank == NOODLE_TENSOR_2D
      ? noodle_i8_tensor_require_2d(dst, src->C, src->W, src->scale, src->zero_point)
      : noodle_i8_tensor_require_1d(dst, src->C, src->W, src->scale, src->zero_point);
  if (!p) return 0;
  memmove(p, src->buffer.data, n * sizeof(NoodleData));
  return src->rank == NOODLE_TENSOR_2D ? src->W : src->C;
}

void noodle_tensor_set_quantization(NoodleTensor *t, float scale,
                                    int32_t zero_point) {
  if (!t || !(scale > 0.0f) || zero_point < -128 || zero_point > 127) return;
  t->scale = scale;
  t->zero_point = zero_point;
}

NoodleData *noodle_tensor_require_1d(NoodleTensor *t, uint16_t C, uint16_t W) {
  const float scale = t && t->scale > 0.0f ? t->scale : 1.0f;
  const int32_t zp = t ? t->zero_point : 0;
  return noodle_i8_tensor_require_1d(t, C, W, scale, zp);
}

NoodleData *noodle_tensor_require_2d(NoodleTensor *t, uint16_t C, uint16_t W) {
  const float scale = t && t->scale > 0.0f ? t->scale : 1.0f;
  const int32_t zp = t ? t->zero_point : 0;
  return noodle_i8_tensor_require_2d(t, C, W, scale, zp);
}

NoodleData *noodle_tensor_require_vector(NoodleTensor *t, uint16_t N) {
  return noodle_tensor_require_1d(t, N, 1);
}

template <typename Layer>
static uint16_t conv2d_with_pool(NoodleTensor *input, NoodleTensor *output,
                                 const Layer &layer, const Pool &pool) {
  Layer configured = with_activation_bounds(layer);
  if (identity_pool(pool) && input && output &&
      !input->file_path && !output->file_path)
    return noodle_i8_conv2d(input, configured, output);
  return noodle_i8_conv2d_fused(input, configured, make_pool(pool), output);
}

uint16_t noodle_conv2d(NoodleTensor *input, NoodleTensor *output,
                       const Conv &conv, const Pool &pool) {
  return conv2d_with_pool(input, output, conv, pool);
}

uint16_t noodle_conv2d(NoodleTensor *input, NoodleTensor *output,
                       const ConvMem &conv, const Pool &pool) {
  return conv2d_with_pool(input, output, conv, pool);
}

uint16_t noodle_conv2d(NoodleTensor *input, NoodleTensor *output,
                       const ConvProgmem &conv, const Pool &pool) {
  return conv2d_with_pool(input, output, conv, pool);
}

uint16_t noodle_conv_transpose2d(NoodleTensor *input, NoodleTensor *output,
                                 const ConvMem &conv) {
  ConvMem configured = with_activation_bounds(conv);
  return noodle_i8_conv_transpose2d(input, configured, output);
}

uint16_t noodle_conv1d(NoodleTensor *input, NoodleTensor *output,
                       const ConvMem &conv) {
  ConvMem configured = with_activation_bounds(conv);
  return noodle_i8_conv1d(input, configured, output);
}

uint16_t noodle_conv1d(NoodleTensor *input, NoodleTensor *output,
                       const ConvMem &conv, const Pool &pool) {
  ConvMem configured = with_activation_bounds(conv);
  if (identity_pool(pool)) return noodle_i8_conv1d(input, configured, output);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t v = noodle_i8_conv1d(input, configured, &tmp);
  if (!v) { noodle_tensor_free(&tmp); return 0; }
  const uint16_t out = noodle_i8_pool1d(&tmp, make_pool(pool), output);
  noodle_tensor_free(&tmp);
  return out;
}

template <typename Layer>
static uint16_t dwconv_with_pool(NoodleTensor *input, NoodleTensor *output,
                                 const Layer &layer, const Pool &pool) {
  Layer configured = with_activation_bounds(layer);
  if (identity_pool(pool) && input && output &&
      !input->file_path && !output->file_path)
    return noodle_i8_dwconv2d(input, configured, output);
  return noodle_i8_dwconv2d_fused(input, configured, make_pool(pool), output);
}

uint16_t noodle_dwconv2d(NoodleTensor *input, NoodleTensor *output,
                         const Conv &conv, const Pool &pool) {
  return dwconv_with_pool(input, output, conv, pool);
}

uint16_t noodle_dwconv2d(NoodleTensor *input, NoodleTensor *output,
                         const ConvMem &conv, const Pool &pool) {
  return dwconv_with_pool(input, output, conv, pool);
}

uint16_t noodle_dwconv2d(NoodleTensor *input, NoodleTensor *output,
                         const ConvProgmem &conv, const Pool &pool) {
  return dwconv_with_pool(input, output, conv, pool);
}

uint16_t noodle_pool2d(NoodleTensor *input, NoodleTensor *output,
                       uint16_t K, uint16_t S) {
  Pool p; p.M = K; p.T = S;
  return noodle_i8_pool2d(input, make_pool(p), output);
}

uint16_t noodle_gap(NoodleTensor *inout) {
  if (!inout || inout->rank != NOODLE_TENSOR_2D || !inout->buffer.data) return 0;
  const uint32_t n = (uint32_t)inout->W * inout->W;
  NoodleI8Requant rq;
  rq.output_scale = inout->scale;
  rq.output_zero_point = inout->zero_point;
  rq.activation_min = -128;
  rq.activation_max = 127;
  if (!noodle_quantize_multiplier(1.0 / (double)n, &rq.multiplier, &rq.shift)) return 0;
  return noodle_i8_gap(inout, rq, inout);
}

uint16_t noodle_gmp(NoodleTensor *inout) {
  return noodle_i8_gmp(inout, inout);
}

uint16_t noodle_flat(NoodleTensor *input, NoodleTensor *output) {
  return noodle_i8_flat(input, output);
}

uint16_t noodle_concat(NoodleTensor *A, NoodleTensor *B, NoodleTensor *output) {
  return noodle_i8_concat(A, B, output);
}

uint16_t noodle_fcn(NoodleTensor *input, NoodleTensor *output,
                    const FCNMem &fcn) {
  FCNMem configured = with_activation_bounds(fcn);
  return noodle_i8_fcn(input, configured, output);
}

#if defined(__AVR__)
// AVR: avoid copying FCNFile for already-valid activation bounds. Keep
// the less common clamp-repair path in a separate call frame to protect
// the SRAM-to-SRAM SD-backed kernel's peak stack usage.
__attribute__((noinline))
static uint16_t noodle_fcn_i8_file_clamped(
    NoodleTensor *input, NoodleTensor *output, const FCNFile &fcn) {
  FCNFile configured = with_activation_bounds(fcn);
  return noodle_i8_fcn(input, configured, output);
}
#endif

uint16_t noodle_fcn(NoodleTensor *input, NoodleTensor *output,
                    const FCNFile &fcn) {
#if defined(__AVR__)
  if (fcn.act != ACT_RELU ||
      fcn.activation_min >= noodle_activation_min_relu(
          fcn.output_scale, fcn.output_zero_point)) {
    return noodle_i8_fcn(input, fcn, output);
  }
  return noodle_fcn_i8_file_clamped(input, output, fcn);
#else
  FCNFile configured = with_activation_bounds(fcn);
  return noodle_i8_fcn(input, configured, output);
#endif
}

uint16_t noodle_fcn(NoodleTensor *input, NoodleTensor *output,
                    const FCNProgmem &fcn) {
  FCNProgmem configured = fcn;
  if ((Activation)configured.act == ACT_RELU) {
    const int32_t qzero = noodle_activation_min_relu(
        configured.output_scale, configured.output_zero_point);
    if (configured.activation_min < qzero) configured.activation_min = qzero;
  }
  return noodle_i8_fcn(input, configured, output);
}

uint16_t noodle_soft_max(NoodleTensor *input_output) {
  if (!input_output || input_output->rank != NOODLE_TENSOR_1D) return 0;
  uint16_t lut_data[256];
  noodle_build_softmax_lut(input_output->scale, lut_data);
  NoodleI8SoftmaxLUT lut;
  lut.exp_q15 = lut_data;
  lut.input_scale = input_output->scale;
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t n = noodle_i8_softmax(input_output, lut, &tmp);
  if (n) copy_tensor_result(&tmp, input_output);
  noodle_tensor_free(&tmp);
  return n;
}

uint16_t noodle_sigmoid(NoodleTensor *input_output) {
  if (!input_output || !input_output->buffer.data) return 0;
  int8_t lut[256];
  noodle_build_sigmoid_lut(input_output->scale, input_output->zero_point, lut);
  NoodleTensor tmp;
  noodle_tensor_init(&tmp);
  const uint16_t n = noodle_i8_lut(input_output, lut, 1.0f / 256.0f, -128, &tmp);
  if (n) copy_tensor_result(&tmp, input_output);
  noodle_tensor_free(&tmp);
  return n;
}

uint16_t noodle_relu(NoodleTensor *input_output) {
  return noodle_i8_relu(input_output);
}


// ---- Codegen graph connectivity interface; standard Noodle numerical kernels ----
namespace {
struct NoodleI8GraphView {
  uint8_t *data;
  uint32_t bytes;
  uint16_t h, w, c;
  int32_t zero_point;
  float scale;
  bool is_uint8;
};

static constexpr uint32_t no_param = 0xffffffffu;
static inline int32_t getq(const NoodleI8GraphView &t, uint32_t i) {
  return t.is_uint8 ? t.data[i] : static_cast<int8_t>(t.data[i]);
}
static inline void setq(NoodleI8GraphView &t, uint32_t i, int32_t q) {
  if(t.is_uint8) { if(q<0)q=0; if(q>255)q=255; t.data[i]=static_cast<uint8_t>(q); }
  else { if(q< -128)q=-128; if(q>127)q=127; t.data[i]=static_cast<uint8_t>(static_cast<int8_t>(q)); }
}
static inline int32_t roundq(float v) {
  if(!isfinite(v)) return 0;
  if(v >= float(INT32_MAX)) return INT32_MAX;
  if(v <= float(INT32_MIN)) return INT32_MIN;
  return static_cast<int32_t>(nearbyintf(v));
}
static inline uint32_t count(const NoodleI8GraphView &t) {
  return uint32_t(t.h)*t.w*t.c;
}
static bool valid(const NoodleI8GraphView &t) {return t.data && t.scale > 0.0f && count(t) == t.bytes;}
static bool span(uint32_t off, uint32_t n, size_t size) {
  return off != no_param && off <= size && n <= size - off;
}
static bool requant(const NoodleI8GraphView &a, NoodleI8GraphView &b) {
  if(count(a)!=count(b))return false;
  const float ratio=a.scale/b.scale;
  for(uint32_t i=0;i<count(a);++i)
    setq(b,i,roundq(float(getq(a,i)-a.zero_point)*ratio)+b.zero_point);
  return true;
}
static bool binary(const NoodleI8GraphView &a, NoodleI8GraphView &b,
                   const int8_t *p, float scale, int32_t zp, bool mul) {
  if(count(a)!=count(b) || !p || a.c!=3 || b.c!=3)return false;
  for(uint32_t i=0;i<count(a);++i) {
    const float lhs=float(getq(a,i)-a.zero_point)*a.scale;
    const float rhs=float(int32_t(p[i/(uint32_t(a.h)*a.w)])-zp)*scale;
    const float v=mul?lhs*rhs:lhs-rhs;
    setq(b,i,roundq(v/b.scale)+b.zero_point);
  }
  return true;
}
// Adapt a logical CHW view to the original NoodleTensor API without owning
// another allocation. The existing external binding keeps the lifetime plan intact.
static bool as_native(const NoodleI8GraphView &v, NoodleTensor *out) {
  if (!out || !v.data || v.h != v.w || v.is_uint8) return false;
  if (!noodle_buffer_bind_external(&out->buffer,
        reinterpret_cast<NoodleData*>(v.data),v.bytes)) return false;
  out->C=v.c;out->W=v.w;out->rank=NOODLE_TENSOR_2D;
  out->scale=v.scale;out->zero_point=v.zero_point;
  out->file_path=nullptr;
  return true;
}
static bool convolution(const NoodleI8GraphOp &op, const NoodleI8GraphView &a, NoodleI8GraphView &b,
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
  // Standard Noodle CHW ConvMem kernel for square 3x3 (and other KxK)
  // convolutions. Pointwise4 remains a specialization for 1x1 layers.
  if (op.kh==op.kw && op.stride_h==op.stride_w &&
      a.h==a.w && b.h==b.w) {
    const uint32_t same=(uint32_t(a.h)+op.stride_h-1u)/op.stride_h;
    const uint32_t valid_out=(uint32_t(a.h)>=op.kh)
        ? (uint32_t(a.h)-op.kh)/op.stride_h+1u : 0u;
    if (b.h==same || b.h==valid_out) {
      NoodleTensor src={},dst={};
      if (!as_native(a,&src) || !as_native(b,&dst)) return false;
      ConvMem layer={};
      layer.K=op.kh;layer.P=b.h==same?NOODLE_SAME_PADDING:0;
      layer.S=op.stride_h;layer.O=b.c;
      layer.weight=reinterpret_cast<const NoodleWeight*>(weights);
      layer.bias=reinterpret_cast<const NoodleBias*>(bias);
      layer.multiplier=mult;layer.shift=shift;
      layer.act=op.relu?ACT_RELU:ACT_NONE;
      layer.input_scale=a.scale;layer.input_zero_point=a.zero_point;
      layer.output_scale=b.scale;layer.output_zero_point=b.zero_point;
      layer.activation_min=op.relu?b.zero_point:-128;
      layer.activation_max=127;
      return noodle_i8_conv2d(&src,layer,&dst)==b.h;
    }
  }

  // Unsupported geometry: do not invent a second convolution implementation.
  return false;
}
static bool maxpool(const NoodleI8GraphOp &op,const NoodleI8GraphView &a,NoodleI8GraphView &b){
  if (a.c!=b.c || !op.kh || op.kh!=op.kw || !op.stride_h || op.stride_h!=op.stride_w ||
      a.scale!=b.scale || a.zero_point!=b.zero_point) return false;
  if ((a.h-op.kh)/op.stride_h+1u!=b.h || b.h!=b.w) return false;
  NoodleTensor src={},dst={};
  if (!as_native(a,&src) || !as_native(b,&dst)) return false;
  NoodleI8Pool pool={}; pool.M=op.kh;pool.T=op.stride_h;pool.mode=NOODLE_I8_POOL_MAX;
  // Reuse the existing portable Noodle INT8 max-pooling operator.
  return noodle_i8_pool2d(&src,pool,&dst)==b.h;
}
static bool concat(const NoodleI8GraphView &a,const NoodleI8GraphView &b,NoodleI8GraphView &dst){
  if (a.h!=b.h || a.w!=b.w || dst.h!=a.h || dst.w!=a.w || dst.c!=a.c+b.c ||
      a.is_uint8!=b.is_uint8 || dst.is_uint8!=a.is_uint8 ||
      a.zero_point!=b.zero_point || dst.zero_point!=a.zero_point ||
      a.scale!=b.scale || dst.scale!=a.scale) return false;
  NoodleTensor first={},second={},out={};
  if(!as_native(a,&first)||!as_native(b,&second)||!as_native(dst,&out))return false;
  // Reuse the existing portable Noodle INT8 CHW concatenation operator.
  return noodle_i8_concat(&first,&second,&out)==dst.c;
}
static bool meanhw(const NoodleI8GraphView &a,NoodleI8GraphView &dst){
  if(a.c!=dst.c || dst.h!=1 || dst.w!=1 || a.is_uint8 || dst.is_uint8)return false;
  const float ratio=a.scale/(float(a.h)*a.w*dst.scale);
  for(uint16_t c=0;c<a.c;c++){
    int32_t sum=0;
    for(uint16_t y=0;y<a.h;y++)for(uint16_t x=0;x<a.w;x++)
      sum+=getq(a,uint32_t(c)*a.h*a.w+uint32_t(y)*a.w+x)-a.zero_point;
    setq(dst,c,roundq(float(sum)*ratio)+dst.zero_point);
  }
  return true;
}

static bool noodle_i8_graph_execute_view(const NoodleI8GraphOp &op,NoodleI8GraphView *t,size_t n,
                          const uint8_t *xip,size_t xip_size){
  if(!t || !xip || op.src0>=n || op.dst>=n)return false;
  NoodleI8GraphView &a=t[op.src0],&dst=t[op.dst];
  if(!valid(a)||!valid(dst))return false;
  switch(op.type){
    case NDL_I8_GRAPH_QUANTIZE:return requant(a,dst);
    case NDL_I8_GRAPH_SUB: case NDL_I8_GRAPH_MUL:
      return span(op.weights,a.c,xip_size) && binary(a,dst,reinterpret_cast<const int8_t*>(xip+op.weights),op.constant_scale,op.constant_zero_point,op.type==NDL_I8_GRAPH_MUL);
    case NDL_I8_GRAPH_CONV:return convolution(op,a,dst,xip,xip_size);
    case NDL_I8_GRAPH_MAXPOOL:return maxpool(op,a,dst);
    case NDL_I8_GRAPH_CONCAT:return op.src1<n && valid(t[op.src1]) && concat(a,t[op.src1],dst);
    case NDL_I8_GRAPH_MEAN:return meanhw(a,dst);
    case NDL_I8_GRAPH_RESHAPE:if(count(a)!=count(dst)||a.is_uint8!=dst.is_uint8||a.scale!=dst.scale||a.zero_point!=dst.zero_point)return false;
      memmove(dst.data,a.data,a.bytes);return true;
    default:return false;
  }
}

} // anonymous namespace

// Resolve logical metadata to shared NoodleTensor storage before calling core INT8 operators.
// Only the operator's 2-3 referenced views are assembled, never the full graph.
// No persistent data pointers: underlying buffers are resolved on each call.
bool noodle_i8_graph_execute(const NoodleI8GraphOp &op, NoodleI8GraphTensor *t,
                            size_t n, const uint8_t *parameters, size_t parameter_bytes) {
  static_assert(sizeof(NoodleData)==1, "CHW INT8 needs one-byte NoodleData");
  if (!t || op.src0>=n || op.dst>=n || (op.type==NDL_I8_GRAPH_CONCAT && op.src1>=n)) return false;
  auto view = [](const NoodleI8GraphTensor &src) -> NoodleI8GraphView {
    if (!src.storage || src.storage->buffer.capacity < src.bytes)
      return {nullptr, src.bytes, src.h, src.w, src.c, src.zero_point, src.scale, src.is_uint8};
    return {const_cast<uint8_t*>(noodle_i8_graph_data(src)), src.bytes, src.h, src.w, src.c, src.zero_point, src.scale, src.is_uint8};
  };
  NoodleI8GraphView views[3] = {view(t[op.src0]), {}, view(t[op.dst])};
  if (op.type==NDL_I8_GRAPH_CONCAT) views[1]=view(t[op.src1]);
  NoodleI8GraphOp local=op;
  local.src0=0; local.src1=1; local.dst=2;
  return noodle_i8_graph_execute_view(local, views, 3, parameters, parameter_bytes);
}

#endif  // NOODLE_USE_INT8
