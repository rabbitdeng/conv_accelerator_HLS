#ifndef __CONV__
#include "ap_int.h"

#define CHout 4
#define R 14
#define C 14
#define S 1
#define Rin 16
#define Cin 16
#define K 3
#define  CHin 4

typedef ap_fixed<8,4> data_t;
typedef ap_fixed<16,8> out_t;

void conv_v2(out_t result[CHout][R][C],
	data_t W[CHout][CHin][K][K],
	data_t In[CHin][Rin][Cin]);
#endif


#define CHout 4
#define R 14
#define C 14
#define S 1
#define Rin 16
#define Cin 16
#define K 3
#define  CHin 4



void conv_v2(out_t Out[CHout][R][C],
	data_t W[CHout][CHin][K][K],
	data_t In[CHin][Rin][Cin])
{
//Out array can't be partitioned or will cause none output and systhesis wrong result.
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=W
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=W
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=In
#pragma HLS TOP name=conv_v2

	Kernel_Row:
		for(int kr=0;kr<K;kr++)
		{
			Kernel_Column:
			for(int kc=0;kc<K;kc++)
			{
				Row:
				for(int r = 0;r < R;r+=S)
				{
					Column:
					for(int c=0;c<C;c+=S)
					{
#pragma HLS PIPELINE II=1
					Output_Channel:
					for(int cho=0;cho<CHout;cho++)
					{
						Input_Channel:
						for(int chi=0;chi<CHin;chi++)
						{
							Out[CHout][R][C] += In[chi][r+kr][c+kc] * W[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}
