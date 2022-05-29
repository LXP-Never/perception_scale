#include <stdio.h>
#include <string.h>
#include <math.h>
#include "rnnoise.h"
#include "kiss_fft.h"
#include "pitch.h"
#include "rnn.h"
#include "rnnoise.h"
#include "rnn_data.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)      // 帧长 480
#define WINDOW_SIZE (2*FRAME_SIZE)              // 窗长 960
#define FREQ_SIZE (FRAME_SIZE + 1)              // 频点数 481

#define PITCH_MIN_PERIOD 60     // 基音最小周期
#define PITCH_MAX_PERIOD 768    // 基音最大周期
#define PITCH_FRAME_SIZE 960    // 帧长
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)  // PITCH_BUF_SIZE=1728

#define SQUARE(x) ((x)*(x))

#define NB_BANDS 22     // 窄带频带数

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)    // 窄带特征 42

#define TRAINING 1
#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

/* The built-in model, used if no file is given as input */
extern const struct RNNModel rnnoise_model_orig;

// 22个点，opus band相关，和频率有一定对应关系，用来进行三角滤波
static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};


/* 创建一个Denoise实例并对其进行初始化 */
DenoiseState *rnnoise_create(RNNModel *model) {
    DenoiseState *st;
    st = malloc(rnnoise_get_size());
    rnnoise_init(st, model);
    return st;
}


struct DenoiseState {
    float analysis_mem[FRAME_SIZE];             // 分析成员(member) [480]
    float cepstral_mem[CEPS_MEM][NB_BANDS];     // 倒谱成员 [8][22]
    int memid;
    float synthesis_mem[FRAME_SIZE];            // 合成 成员
    float pitch_buf[PITCH_BUF_SIZE];            // [1728]
    float pitch_enh_buf[PITCH_BUF_SIZE];        // [1728]
    float last_gain;                            // gian值
    int last_period;                            // 最后一个周期
    float mem_hp_x[2];
    float lastg[NB_BANDS];                  // [22]
    RNNState rnn;
};

typedef struct {
    int init;
    kiss_fft_state *kfft;
    float half_window[FRAME_SIZE];
    float dct_table[NB_BANDS * NB_BANDS];
} CommonState;

/* 获取denoise的大小（字节数） */
int rnnoise_get_size() {
    return sizeof(DenoiseState);
}



/*函数声明*/
/**
 * biquad滤波，双二阶滤波器
 * @param y 输出信号
 * @param mem 调节参数，从训练模型中获取
 * @param x 输入信号
 * @param b biquad系数b值
 * @param a biquad系数a值
 * @param N x信号数组长度
 */
static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N);

/**
 * 帧分析
 * @param st
 * @param X 对x进行FFT后的X
 * @param Ex X的频带能量Ex[22]
 * @param in 长度为480的输入数据
 */
static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in);

/**
 * 快速傅里叶变换，out = fft(in)
 * @param out   频域频谱 X[481]
 * @param in    时域帧 加窗后的x[960]
 */
static void forward_transform(kiss_fft_cpx *out, const float *in);

/**
 * 计算每个band的能量
 * @param bandE     bandE[22] 储存每个band的能量
 * @param X         频域 X[481]
 */
void compute_band_energy(float *bandE, const kiss_fft_cpx *X);

/**
 * 加窗
 * @param x 前后两帧x[960]=[：后480]
 */
static void apply_window(float *x);

// 检查初始化状态，即要对fft运算分配内存空间，然后生成需要使用的dct table
static void check_init();

/**
 * 输入在后，输出在前
 * 计算42维特征
 * @param st    全局context，存储一些上下文用的变量和结构体
 * @param X     对x进行FFT后的X
 * @param P     对p进行FFT后的P
 * @param Ex    对X进行三角滤波得到22个频带的能量Ex
 * @param Ep    对P进行三角滤波得到22个频带的能量Ep
 * @param Exp   利用P和X计算得到的基音相关度Exp
 * @param features  输出42点特征(RNN的输入)
 * @param in    输入信号 x[480],biquad 的输出
 * @return      如果是训练数据有返回值，返回是否有语音
 */
static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features,
                                  const float *in);

/**
 * 计算band之间的相关度
 * @param 输出频带相关性 Exp[22]
 * @param X X[481]
 * @param P P[481]
 */
void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P);

// 对输入数据 in 做离散余弦变换
static void dct(float *out, const float *in);

/**
 * 论文中的pitch filter部分，下面的r[i]实际上就是论文中的alpha
 * @param X     对x进行FFT后的X
 * @param P     对p进行FFT后的P
 * @param Ex    对X进行三角滤波得到22个频带的能量Ex
 * @param Ep    对P进行三角滤波得到22个频带的能量Ep
 * @param Exp   利用P和X计算得到的进行基音相关度Exp
 * @param g     神经网络的输出
 */
void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep, const float *Exp,
                  const float *g);

/*
 * 插值，即22点数据插值得到481点数据
 */
void interp_band_gain(float *g, const float *bandE);

CommonState common;

int main() {
    int i;
    FILE *f1, *f2;
    float x[FRAME_SIZE];
    float n[FRAME_SIZE];
    float xn[FRAME_SIZE];           // 声明480点带噪语音
    short tmp[FRAME_SIZE];
    float speech_gain = 1, noise_gain = 1;
    DenoiseState *st;
    DenoiseState *noise_state;
    DenoiseState *noisy;
    st = rnnoise_create(NULL);
    noise_state = rnnoise_create(NULL);
    noisy = rnnoise_create(NULL);


    f1 = fopen("../clean.pcm", "r");
    if (f1 == NULL) {
        printf("!Error: Cant't open file.\n");
        exit(1);
    }
    f2 = fopen("../noise.pcm", "r");
    if (f2 == NULL) {
        printf("!Error: Cant't open file.\n");
        exit(1);
    }

    /* 进入循环 看要读取多少个特征 */
    // while (1) {
    float E = 0;
    /* 纯净语音读取 */
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) {
        rewind(f1);
        fread(tmp, sizeof(short), FRAME_SIZE, f1);
    }
    for (i = 0; i < FRAME_SIZE; i++) x[i] = speech_gain * tmp[i];   // 纯净语音 * 语音增益
    for (i = 0; i < FRAME_SIZE; i++) E += tmp[i] * (float) tmp[i];  // 纯净语音的 能量

    /* 噪声读取 */
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
    if (feof(f2)) {
        rewind(f2);
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
    }
    for (i = 0; i < FRAME_SIZE; i++) n[i] = noise_gain * tmp[i];    // 噪声 * 噪声增益

    /* 纯净语音经过 双二阶滤波器 */
    float mem_hp_x[2] = {0};
    float mem_hp_n[2] = {0};
    float mem_resp_x[2] = {0};
    float mem_resp_n[2] = {0};
    static const float a_hp[2] = {-1.99599, 0.99600};   // 定义全局数组常量
    static const float b_hp[2] = {-2, 1};           // 定义全局数组常量
    float a_noise[2] = {0};
    float b_noise[2] = {0};
    float a_sig[2] = {0};
    float b_sig[2] = {0};

    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
    for (i = 0; i < FRAME_SIZE; i++) xn[i] = x[i] + n[i];       // 带噪语音 = 纯净语音 + 噪声

    // VAD
    int vad_cnt = 0;
    float vad = 0;
    if (E > 1e9f) {
        vad_cnt = 0;
    } else if (E > 1e8f) {
        vad_cnt -= 5;
    } else if (E > 1e7f) {
        vad_cnt++;
    } else {
        vad_cnt += 2;
    }
    if (vad_cnt < 0) vad_cnt = 0;
    if (vad_cnt > 15) vad_cnt = 15;

    if (vad_cnt >= 10) vad = 0;
    else if (vad_cnt > 0) vad = 0.5f;
    else vad = 1.f;


    /* 加窗分析 */
    kiss_fft_cpx Y[FRAME_SIZE], N[FREQ_SIZE];   // 声明纯净语音和噪声的的FFT
    float Ey[NB_BANDS], En[NB_BANDS];       // 22个点的频带能量(纯净语音和噪声的)
    float Ln[NB_BANDS];         // 对数频带能量(噪声的)

    frame_analysis(st, Y, Ey, x);       // 纯净语音 分帧加窗 FFT
    frame_analysis(st, Y, Ey, x);       // 纯净语音 分帧加窗 FFT
    frame_analysis(noise_state, N, En, n);  // 噪声 分帧加窗 FFT
    for (i = 0; i < NB_BANDS; i++) Ln[i] = log10(1e-2 + En[i]);

    /* 提取42点特征，*/
    kiss_fft_cpx X[FRAME_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ep[NB_BANDS];       // 22个点的频带能量(带噪语音和基音的)
    float Exp[NB_BANDS];                // 带噪语音频谱X和基音频谱P的 相关性
    float features[NB_FEATURES];    // 声明42点特征

    int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);   // 提取带噪语音的特征

    /* 基音滤波器 */
    float g[NB_BANDS];
    pitch_filter(X, P, Ex, Ep, Exp, g);
    for (i = 0; i < NB_BANDS; i++) {
        g[i] = sqrt((Ey[i] + 1e-3) / (Ex[i] + 1e-3));       // 增益 mask
        if (g[i] > 1) g[i] = 1;
        if (silence || i > band_lp) g[i] = -1;      // 如果静音 g[i]=-1
        if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;    // 如果纯净语音和带噪语音的音量都很小，则g[i]=-1
        if (vad == 0 && noise_gain == 0) g[i] = -1;     // 如果VAD为0，则g[i]=-1
    }

    fwrite(features, sizeof(float), NB_FEATURES, stdout);        // 42 (特征提取) RNN输入
    fwrite(g, sizeof(float), NB_BANDS, stdout);                     // 22 (期望增益)    label,RNN输出
    fwrite(Ln, sizeof(float), NB_BANDS, stdout);                    // 22 (噪声对数谱)
    fwrite(&vad, sizeof(float), 1, stdout);



    return 0;
}


/**
 * biquad滤波，双二阶滤波器
 * @param y 输出信号
 * @param mem 调节参数，从训练模型中获取
 * @param x 输入信号
 * @param b biquad系数b值
 * @param a biquad系数a值
 * @param N x信号数组长度
 */
static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
    int i;
    for (i = 0; i < N; i++) {
        float xi, yi;
        xi = x[i];
        yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0] * (double) xi - a[0] * (double) yi);
        mem[1] = (b[1] * (double) xi - a[1] * (double) yi);
        y[i] = yi;
    }
}

/**
 * 帧分析
 * @param st
 * @param X 对x进行FFT后的X
 * @param Ex X的频带能量Ex[22]
 * @param in 长度为480的输入数据
 */
static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
    int i;
    float x[WINDOW_SIZE];       // x[960]
    // 从 analysis_mem拷贝FRAME_SIZE个数据赋给x的前半段，analysis_mem是上一次的输入
    RNN_COPY(x, st->analysis_mem, FRAME_SIZE);      // 把老数据往前移
    for (i = 0; i < FRAME_SIZE; i++)
        x[FRAME_SIZE + i] = in[i];                  // 把新数据 放后面
    RNN_COPY(st->analysis_mem, in, FRAME_SIZE);     // 然后再将in拷贝给 analysis_mem
    apply_window(x);                // 对x进行加窗
    forward_transform(X, x);        // 对x进行fft并储存在X中
#if TRAINING
    for (i = lowpass; i < FREQ_SIZE; i++)       // 第481个频点实部和虚部置零
        X[i].r = X[i].i = 0;
#endif
    compute_band_energy(Ex, X);     // 计算22个频带能量
}

/**
 * 加窗
 * @param x 前后两帧x[960]=[：后480]
 */
static void apply_window(float *x) {
    int i;
    check_init();
    for (i = 0; i < FRAME_SIZE; i++) {
        x[i] *= common.half_window[i];                      // 前480
        x[WINDOW_SIZE - 1 - i] *= common.half_window[i];    // 后480 倒序
    }
}

/**
 * 快速傅里叶变换，out = fft(in)
 * @param out   频域频谱 X[481]
 * @param in    时域帧 加窗后的x[960]
 */
static void forward_transform(kiss_fft_cpx *out, const float *in) {
    int i;
    kiss_fft_cpx x[WINDOW_SIZE];    // x[960]
    kiss_fft_cpx y[WINDOW_SIZE];    // y[960]
    check_init();          // 生成fft的存储数据
    //将输入数据赋值给x
    for (i = 0; i < WINDOW_SIZE; i++) {
        x[i].r = in[i];     // 实部？
        x[i].i = 0;         // 虚部？
    }
    opus_fft(common.kfft, x, y, 0);   // 对输入数据x进行fft，计算结果给y
    for (i = 0; i < FREQ_SIZE; i++) {
        out[i] = y[i];          // 储存计算结果
    }
}

/**
 * 计算每个band的能量
 * @param bandE     bandE[22] 储存每个band的能量
 * @param X         频域 X[481]
 */
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
    int i;
    float sum[NB_BANDS] = {0};
    //对22个band进行操作
    for (i = 0; i < NB_BANDS - 1; i++) {
        int j;
        int band_size;
        // band_size 每个频带的频点数
        // band的点数为eband5ms每两个值之间的差乘以4
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        for (j = 0; j < band_size; j++) {       // 每个band内部的计算
            float tmp;
            // frac实际上就是论文中的wb，将frac作用于X实际上就是将opus band作用于频谱
            float frac = (float) j / band_size;
            tmp = SQUARE(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r);   // 实部
            tmp += SQUARE(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i);  // +虚部
            // 每两个三角窗都有一半区域是重叠的，因而对于某一段频点，其既被算入该频带，也被算入下一频带 ？？
            sum[i] += (1 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    // 第一个band和最后一个band的窗只有一半因而能量乘以2
    sum[0] *= 2;
    sum[NB_BANDS - 1] *= 2;
    for (i = 0; i < NB_BANDS; i++) {
        bandE[i] = sum[i];
    }
}


// 检查初始化状态，即要对fft运算分配内存空间，然后生成需要使用的dct table
static void check_init() {
    int i;
    if (common.init) return;
    common.kfft = opus_fft_alloc_twiddles(2 * FRAME_SIZE, NULL, NULL, NULL, 0);
    for (i = 0; i < FRAME_SIZE; i++)
        common.half_window[i] = sin(
                .5 * M_PI * sin(.5 * M_PI * (i + .5) / FRAME_SIZE) * sin(.5 * M_PI * (i + .5) / FRAME_SIZE));
    for (i = 0; i < NB_BANDS; i++) {
        int j;
        for (j = 0; j < NB_BANDS; j++) {
            common.dct_table[i * NB_BANDS + j] = cos((i + .5) * j * M_PI / NB_BANDS);
            if (j == 0) common.dct_table[i * NB_BANDS + j] *= sqrt(.5);
        }
    }
    common.init = 1;
}

/* 对DenoiseState整个结构体进行初始化 */
int rnnoise_init(DenoiseState *st, RNNModel *model) {
    memset(st, 0, sizeof(*st));
    if (model)
        st->rnn.model = model;
    else
        st->rnn.model = &rnnoise_model_orig;
    st->rnn.vad_gru_state = calloc(sizeof(float), st->rnn.model->vad_gru_size);
    st->rnn.noise_gru_state = calloc(sizeof(float), st->rnn.model->noise_gru_size);
    st->rnn.denoise_gru_state = calloc(sizeof(float), st->rnn.model->denoise_gru_size);
    return 0;
}

/**
 * 输入在后，输出在前
 * 计算42维特征
 * @param st    全局context，存储一些上下文用的变量和结构体
 * @param X     对x进行FFT后的X
 * @param P     对p进行FFT后的P
 * @param Ex    对X进行三角滤波得到22个频带的能量Ex
 * @param Ep    对P进行三角滤波得到22个频带的能量Ep
 * @param Exp   利用P和X计算得到的基音相关度Exp
 * @param features  输出42点特征(RNN的输入)
 * @param in    输入信号 x[480],biquad 的输出
 * @return      如果是训练数据有返回值，返回是否有语音
 */
static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features,
                                  const float *in) {
    int i;
    float E = 0;
    float *ceps_0, *ceps_1, *ceps_2;
    float spec_variability = 0;
    float Ly[NB_BANDS];
    float p[WINDOW_SIZE];
    float pitch_buf[PITCH_BUF_SIZE >> 1];
    int pitch_index;      // 这里设置的是整型，但下面的参数是其地址，则可改变原来的值
    float gain;             // 增益
    float *(pre[1]);
    float tmp[NB_BANDS];
    float follow, logMax;
    frame_analysis(st, X, Ex, in);      // 带噪语音做分析窗，X是做完fft的数据， Ex是22个频带的能量

    // 从src拷贝n个字节给dst，不同的是，若src和dst内存有重叠，也能顺利拷贝
    // 将pitch_buf[480:1728]个字节数据，拷贝到 pitch_buf[0:(1728-480)]. 相当于将数据前移480位
    RNN_MOVE(st->pitch_buf, &(st->pitch_buf[FRAME_SIZE]), PITCH_BUF_SIZE - FRAME_SIZE);
    // 将in[480]复制到pitch_buf[(1728-480):1728]的位置，对其后面的480个数据进行更新
    RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE], in, FRAME_SIZE);
    pre[0] = &st->pitch_buf[0];     // 指针指向pitch_buf[0]的首地址

    // 降采样，对pitch_buf平滑降采样，求自相关，利用自相关求lpc系数，然后进行lpc滤波，即得到lpc残差
    pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
    // 寻找基音周期   存入pitch_index80*2
    pitch_search(pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
                 PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, &pitch_index);
    pitch_index = PITCH_MAX_PERIOD - pitch_index;
    // 去除高阶谐波影响
    gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD, PITCH_FRAME_SIZE, &pitch_index,
                           st->last_period, st->last_gain);
    printf("gain:%f\n", gain);
    st->last_period = pitch_index;
    st->last_gain = gain;
    // 根据index得到p[i]
    for (i = 0; i < WINDOW_SIZE; i++)
        p[i] = st->pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index + i];       // 768-pitch_index + i
    apply_window(p);                          // pitch数据应用window
    forward_transform(P, p);                  // 对pitch数据进行傅里叶变换
    compute_band_energy(Ep, P);               // 计算pitch部分band能量
    compute_band_corr(Exp, X, P);             // 计算信号频域与pitch频域的相关band能量系数
    for (i = 0; i < NB_BANDS; i++)
        Exp[i] = Exp[i] / sqrt(.001 + Ex[i] * Ep[i]);    // Exp进行标准化
    dct(tmp, Exp);                          // 然后再做一次dct，实际上就是信号与pitch相关BFCC了
    // 然后填充feature的 NB_BANDS+2NB_DELTA_CEPS （22+26=34）到NB_BANDS+3*NB_DELTA_CEPS （40），填充后实际上是做了一些参数调整的
    for (i = 0; i < NB_DELTA_CEPS; i++) {
        features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];    // features[22+6*2+i<6]  基音相关度
    }
    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01 * (pitch_index - 300);     // features[22+3*6]  基音周期
    //而feature的1-NB_BANDS（22）是由log10(Ex)再做一次DCT后填充的，代码如下
    logMax = -2;
    follow = -2;
    for (i = 0; i < NB_BANDS; i++) {
        Ly[i] = log10(1e-2 + Ex[i]);
        Ly[i] = MAX16(logMax - 7, MAX16(follow - 1.5, Ly[i]));
        logMax = MAX16(logMax, Ly[i]);
        follow = MAX16(follow - 1.5, Ly[i]);
        E += Ex[i];
    }
    if (!TRAINING && E < 0.04) {
        /* 如果没有声音，避免把state搞乱. */
        RNN_CLEAR(features, NB_FEATURES);
        return 1;
    }
    dct(features, Ly);  // features[0-22] 就是带噪语音22频带对数能量

    /*
     * cepstral_mem 是一个8*22的数组,每一次feature填充到ceps_0,然后这个数组会往下再做一次。
     * ceps_0 是float指针，它指向的是ceptral_mem第一个NB_BANDS数组，然后每次与相邻的band数组相减，做出一个delta差值。
     */
    features[0] -= 12;
    features[1] -= 4;
    ceps_0 = st->cepstral_mem[st->memid];
    ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM + st->memid - 1] : st->cepstral_mem[st->memid - 1];
    ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM + st->memid - 2] : st->cepstral_mem[st->memid - 2];
    for (i = 0; i < NB_BANDS; i++) ceps_0[i] = features[i];
    st->memid++;
    for (i = 0; i < NB_DELTA_CEPS; i++) {
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];        // features[i<6]  BFCC
        // features[22+i<6] BFCC一阶导
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
        // features[22+6+i<6] BFCC二阶导
        features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2 * ceps_1[i] + ceps_2[i];
    }
    /* Spectral variability features. */
    if (st->memid == CEPS_MEM) st->memid = 0;
    // 最后一个特性值 平稳性度量
    for (i = 0; i < CEPS_MEM; i++) {
        int j;
        float mindist = 1e15f;
        for (j = 0; j < CEPS_MEM; j++) {
            int k;
            float dist = 0;
            for (k = 0; k < NB_BANDS; k++) {
                float tmp;
                tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            if (j != i)
                mindist = MIN32(mindist, dist);
        }
        spec_variability += mindist;
    }
    //features[41] 平稳性度量
    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM - 2.1;
    // 返回值为判断是否有语音
    return TRAINING && E < 0.1;
}

/**
 * 计算band之间的相关度
 * @param 输出频带相关性 Exp[22]
 * @param X X[481]
 * @param P P[481]
 */
void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
    int i;
    float sum[NB_BANDS] = {0};
    for (i = 0; i < NB_BANDS - 1; i++) {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        for (j = 0; j < band_size; j++) {
            float tmp;
            float frac = (float) j / band_size;     // 求解wb
            tmp = X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r;
            tmp += X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i;
            sum[i] += (1 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2;
    sum[NB_BANDS - 1] *= 2;
    for (i = 0; i < NB_BANDS; i++) {
        bandE[i] = sum[i];
    }
}

/* 对输入数据 in 做离散余弦变换 */
static void dct(float *out, const float *in) {
    int i;
    check_init();
    for (i = 0; i < NB_BANDS; i++) {
        int j;
        float sum = 0;
        for (j = 0; j < NB_BANDS; j++) {
            sum += in[j] * common.dct_table[j * NB_BANDS + i];
        }
        out[i] = sum * sqrt(2. / 22);
    }
}

/* ============================================================= */

/**
 * 论文中的pitch filter部分，下面的r[i]实际上就是论文中的alpha
 * @param X     对x进行FFT后的X[481]
 * @param P     对p进行FFT后的P[481]
 * @param Ex    对X进行三角滤波得到22个频带的能量Ex[22]
 * @param Ep    对P进行三角滤波得到22个频带的能量Ep[22]
 * @param Exp   利用P和X计算得到的进行基音相关度Exp[22]
 * @param g     神经网络的输出
 */
void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep, const float *Exp,
                  const float *g) {
    int i;
    float r[NB_BANDS];
    float rf[FREQ_SIZE] = {0};
    for (i = 0; i < NB_BANDS; i++) {
#if 0
        if (Exp[i]>g[i]) r[i] = 1;
        else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
        r[i] = MIN16(1, MAX16(0, r[i]));
#else
        if (Exp[i] > g[i]) {
            r[i] = 1;
        } else {
            r[i] = SQUARE(Exp[i]) * (1 - SQUARE(g[i])) / (.001 + SQUARE(g[i]) * (1 - SQUARE(Exp[i])));
        }
        r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
        r[i] *= sqrt(Ex[i] / (1e-8 + Ep[i]));
    }
    interp_band_gain(rf, r);        // 22个点插值到481个点
    for (i = 0; i < FREQ_SIZE; i++) {
        X[i].r += rf[i] * P[i].r;
        X[i].i += rf[i] * P[i].i;
    }
    float newE[NB_BANDS];
    compute_band_energy(newE, X);

    float norm[NB_BANDS];
    for (i = 0; i < NB_BANDS; i++) {
        norm[i] = sqrt(Ex[i] / (1e-8 + newE[i]));
    }

    float normf[FREQ_SIZE] = {0};
    interp_band_gain(normf, norm);
    for (i = 0; i < FREQ_SIZE; i++) {
        X[i].r *= normf[i];
        X[i].i *= normf[i];
    }
}
/**
 * 插值，即22点数据插值得到481点数据
 * @param g bandE[22]
 * @param bandE g[481]
 */
void interp_band_gain(float *g, const float *bandE) {
    int i;
    memset(g, 0, FREQ_SIZE);
    for (i = 0; i < NB_BANDS - 1; i++) {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        for (j = 0; j < band_size; j++) {
            float frac = (float) j / band_size;
            g[(eband5ms[i] << FRAME_SIZE_SHIFT) + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1];
        }
    }
}

/* ============================================================= */