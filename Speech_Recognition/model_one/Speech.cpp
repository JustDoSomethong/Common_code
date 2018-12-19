#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "./include/qisr.h"
#include "./include/msp_cmn.h"
#include "./include/msp_errors.h"
#include "./samples/asr_record_sample/speech_recognizer.h"
#define MAX_GRAMMARID_LEN   (32)

using namespace std;

typedef struct _UserData {
	int     build_fini; //标识语法构建是否完成
	int     update_fini; //标识更新词典是否完成
	int     errcode; //记录语法构建或更新词典回调错误码
	char    grammar_id[MAX_GRAMMARID_LEN]; //保存语法构建返回的语法ID
}UserData;

int main(int argc, char* argv[])
{
    const char *login_config    = "appid = 598422f9"; //登录参数
	UserData asr_data; 
	int ret                     = 0 ;
    int Control_Character;

    ret = MSPLogin(NULL, NULL, login_config); //第一个参数为用户名，第二个参数为密码，传NULL即可，第三个参数是登录参数
	if (MSP_SUCCESS != ret) {
		printf("登录失败：%d\n", ret);
		return 0;
	}

	memset(&asr_data, 0, sizeof(UserData));
	printf("构建离线识别语法网络...\n");

    while(1)
    {
         if(Control_Character = getchar() =='q')
         {
             return 0;
         } 
         
    }
    return 0;
}