
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "meow_hash_x64_aesni.h"

using namespace std;
#define prefetcht0(A)           _mm_prefetch((char *)(A), _MM_HINT_T0)
#define movdqu(A, B)        A = _mm_loadu_si128((__m128i *)(B))
#define movdqu_mem(A, B)        _mm_storeu_si128((__m128i *)(A), B)
#define movq(A, B)          A = _mm_set_epi64x(0, B);
#define aesdec(A, B)        A = _mm_aesdec_si128(A, B)
#define aesenc(A, B)	    A = _mm_aesenc_si128(A, B)
#define pshufb(A, B)        A = _mm_shuffle_epi8(A, B)
#define pxor(A, B)          A = _mm_xor_si128(A, B)
#define paddq(A, B)         A = _mm_add_epi64(A, B)
#define psubq(A, B)         A = _mm_sub_epi64(A,B)
#define pand(A, B)          A = _mm_and_si128(A, B)
#define palignr(A, B, i)    A = _mm_alignr_epi8(A, B, i)
#define pxor_clear(A, B)    A = _mm_setzero_si128(); // NOTE(casey): pxor_clear is a nonsense thing that is only here because compilers don't detect xor(a, a) is clearing a :(

#define INSTRUCTION_REORDER_BARRIER _ReadWriteBarrier()

#define MEOW_MIX_REG(r1, r2, r3, r4, r5,  i1, i2, i3, i4) \
aesdec(r1, r2);              \
INSTRUCTION_REORDER_BARRIER; \
paddq(r3, i1);               \
pxor(r2, i2);                \
aesdec(r2, r4);              \
INSTRUCTION_REORDER_BARRIER; \
paddq(r5, i3);               \
pxor(r4, i4);

#define MEOW_MIX(r1, r2, r3, r4, r5,  ptr) \
MEOW_MIX_REG(r1, r2, r3, r4, r5, _mm_loadu_si128( (__m128i *) ((ptr) + 15) ), _mm_loadu_si128( (__m128i *) ((ptr) + 0)  ), _mm_loadu_si128( (__m128i *) ((ptr) + 1)  ), _mm_loadu_si128( (__m128i *) ((ptr) + 16) ))

#define MEOW_SHUFFLE(r1, r2, r3, r4, r5, r6) \
aesdec(r1, r4); \
paddq(r2, r5);  \
pxor(r4, r6);   \
aesdec(r4, r2); \
paddq(r5, r6);  \
pxor(r2, r3)

static meow_u128 xmm_allzero = _mm_setzero_si128(); // All zero

// 逆序函数
#define inv_mixcol(A)		A = _mm_aesimc_si128(A) // AES-128-ECB invert mix columns
#define MixColumns(A)		A = _mm_aesdeclast_si128(A, xmm_allzero); A = _mm_aesenc_si128(A, xmm_allzero) // AES-128-ECB mix columns
#define SubBytes(A)			A = _mm_shuffle_epi8(A, xmm_SubBytes); A = _mm_aesenc_si128(A, xmm_allzero) // AES-128-ECB sub bytes
#define ShiftRows(A)		A = _mm_shuffle_epi8(A, xmm_ShiftRows) // AES-128-ECB shift rows
//AES部件逆序
#define inv_aesdec(A, B) \
pxor(A, B);              \
MixColumns(A);           \
aesenc(A, xmm_allzero);  \
inv_mixcol(A)

//Finalization 逆序
#define INV_MEOW_SHUFFLE(r0, r1, r2, r4, r5, r6) \
pxor(r1, r2);         \
inv_aesdec(r4, r1);   \
psubq(r5, r6);        \
pxor(r4, r6);         \
psubq(r1, r5);        \
inv_aesdec(r0, r4);

//Absorb 逆序
#define INV_MEOW_MIX_REG(r1, r2, r3, r4, r5,  i1, i2, i3, i4) \
pxor(r4, i4);                \
psubq(r5, i3);               \
INSTRUCTION_REORDER_BARRIER; \
inv_aesdec(r2, r4);          \
pxor(r2, i2);                \
psubq(r3, i1);               \
INSTRUCTION_REORDER_BARRIER; \
inv_aesdec(r1, r2);          \


static void InvToGetKey(meow_umm Len, void* h, void* msg, void* get_key) {

	meow_u128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
	meow_u128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

	meow_u8* rcx = (meow_u8*)h;
	movdqu(xmm0, rcx+0x00);
	movdqu(xmm1, rcx+0x10);
	movdqu(xmm2, rcx+0x20);
	movdqu(xmm3, rcx+0x30);
	movdqu(xmm4, rcx+0x40);
	movdqu(xmm5, rcx+0x50);
	movdqu(xmm6, rcx+0x60);
	movdqu(xmm7, rcx+0x70);
	//Squzze 逆序
	psubq(xmm0, xmm4);
	pxor(xmm0, xmm1);
	pxor(xmm4, xmm5);
	psubq(xmm0, xmm2);
	psubq(xmm1, xmm3);
	psubq(xmm4, xmm6);
	psubq(xmm5, xmm7);
	//finalization 逆序
	INV_MEOW_SHUFFLE(xmm3, xmm4, xmm5, xmm7, xmm0, xmm1);
	INV_MEOW_SHUFFLE(xmm2, xmm3, xmm4, xmm6, xmm7, xmm0);
	INV_MEOW_SHUFFLE(xmm1, xmm2, xmm3, xmm5, xmm6, xmm7);
	INV_MEOW_SHUFFLE(xmm0, xmm1, xmm2, xmm4, xmm5, xmm6);
	INV_MEOW_SHUFFLE(xmm7, xmm0, xmm1, xmm3, xmm4, xmm5);
	INV_MEOW_SHUFFLE(xmm6, xmm7, xmm0, xmm2, xmm3, xmm4);
	INV_MEOW_SHUFFLE(xmm5, xmm6, xmm7, xmm1, xmm2, xmm3);
	INV_MEOW_SHUFFLE(xmm4, xmm5, xmm6, xmm0, xmm1, xmm2);
	INV_MEOW_SHUFFLE(xmm3, xmm4, xmm5, xmm7, xmm0, xmm1);
	INV_MEOW_SHUFFLE(xmm2, xmm3, xmm4, xmm6, xmm7, xmm0);
	INV_MEOW_SHUFFLE(xmm1, xmm2, xmm3, xmm5, xmm6, xmm7);
	INV_MEOW_SHUFFLE(xmm0, xmm1, xmm2, xmm4, xmm5, xmm6);
//*********************************************************
//以下xmm赋值代码引用自meow.h文件,特此说明
	pxor_clear(xmm9, xmm9);
	pxor_clear(xmm11, xmm11);
    meow_u8* Last = (meow_u8*)msg + (Len & ~0xf);
    int unsigned Len8 = (Len & 0xf);
    if (Len8)
    {
        // NOTE(casey): Load the mask early
        movdqu(xmm8, &MeowMaskLen[0x10 - Len8]);

        meow_u8* LastOk = (meow_u8*)((((meow_umm)(((meow_u8*)msg) + Len - 1)) | (MEOW_PAGESIZE - 1)) - 16);
        int Align = (Last > LastOk) ? ((int)(meow_umm)Last) & 0xf : 0;
        movdqu(xmm10, &MeowShiftAdjust[Align]);
        movdqu(xmm9, Last - Align);
        pshufb(xmm9, xmm10);

        // NOTE(jeffr): and off the extra bytes
        pand(xmm9, xmm8);
    }

    // NOTE(casey): Next, we have to load the part that _is_ 16-byte aligned
    if (Len & 0x10)
    {
        xmm11 = xmm9;
        movdqu(xmm9, Last - 0x10);
    }

    //
    // NOTE(casey): Construct the residual and length injests
    //

    xmm8 = xmm9;
    xmm10 = xmm9;
    palignr(xmm8, xmm11, 15);
    palignr(xmm10, xmm11, 1);

    // NOTE(casey): We have room for a 128-bit nonce and a 64-bit none here, but
    // the decision was made to leave them zero'd so as not to confuse people
    // about hwo to use them or what security implications they had.
    pxor_clear(xmm12, xmm12);
    pxor_clear(xmm13, xmm13);
    pxor_clear(xmm14, xmm14);
    movq(xmm15, Len);
    palignr(xmm12, xmm15, 15);
    palignr(xmm14, xmm15, 1);

	INV_MEOW_MIX_REG(xmm1, xmm5, xmm7, xmm2, xmm3, xmm12, xmm13, xmm14, xmm15);
	INV_MEOW_MIX_REG(xmm0, xmm4, xmm6, xmm1, xmm2, xmm8, xmm9, xmm10, xmm11);

	meow_u8* rax = (meow_u8*)get_key;
	movdqu_mem(rax + 0x00, xmm0);
	movdqu_mem(rax + 0x10, xmm1);
	movdqu_mem(rax + 0x20, xmm2);
	movdqu_mem(rax + 0x30, xmm3);
	movdqu_mem(rax + 0x40, xmm4);
	movdqu_mem(rax + 0x50, xmm5);
	movdqu_mem(rax + 0x60, xmm6);
	movdqu_mem(rax + 0x70, xmm7);
	get_key = rax;

	return;
}

const char* MSG = "Yang Wentao 202000460052";
const char* h = "sdu_cst_20220610";
int main()
{
	int MsgLen = strlen(MSG); 
	char* message = new char[MsgLen+1];
	memset(message, 0, MsgLen+1);
	memcpy(message, MSG, MsgLen);

	cout << "m: " << message << endl;
	int Hashed_MsgLen = strlen(h);
	char* Hashed_message = new char[Hashed_MsgLen+1 ];
	memset(Hashed_message, 0, Hashed_MsgLen +1);
	memcpy(Hashed_message, h, Hashed_MsgLen);

	cout << "h: " << Hashed_message << endl;

	meow_u8 get_key[129];
	memset(get_key, 0, 129);
	InvToGetKey(MsgLen, Hashed_message, message, get_key); 
	cout << "生成的k值:";
	for (int i = 0; i < 128; i++) {
		printf("%02X", (int)get_key[i]);
		if (i % 4 == 3) {
			cout << " ";
		}
		if (i % 16 == 15) {
			cout << endl;
		}
	}
	meow_u128 Hash = MeowHash(get_key, MsgLen, message);

	meow_u8 temp[17];
	memset(temp, 0, 17);
	movdqu_mem(temp + 0x00, Hash);
	cout << "根据生成的k计算的meow_hash值:";
	printf("\t%s\n", temp);

	return 0;
}