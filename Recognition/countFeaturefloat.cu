#include "countFeature.cuh"
#define L2HYS_EPSILON 		0.01f
#define L2HYS_EPSILONHYS	1.0f
#define L2HYS_CLIP			0.2f
#define data_h2y            30
__global__ void countCell(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int ImageHeight,int Imagewidth)
{

    int xx=blockIdx.x*blockDim.x+threadIdx.x;
    int yy=blockIdx.y*blockDim.y+threadIdx.y;
    int tidx=threadIdx.x;
    int tidy=threadIdx.y;
    long id=xx+yy*Imagewidth;
    int idx=tidy*Windowx+tidx;
     __shared__  float histo[1260 ];//һ��Բ��18������(max(0~17))*����Ŀ��70(ÿ������7��cellÿ��cell 10��bin)+������ţ�max(0~6)��*bin����10��+�����ĸ�bin(max(0~9))=17*70+6*10+9=1259
    if(c_Mag[tidx+tidy*Windowx]>64)
        return;
    memset(histo,0,1260*sizeof(float));//ÿ��������һ��cell������ֱ��ͼ��ʱ����Ҫ��ֱ��ͼ����
    __syncthreads();
    //for(int stridex=0,stridey=0;stridex<gridDim.x, )
        if(xx<Imagewidth&&yy<ImageHeight)
        {
            float t_fm_nbin=p_ANG[yy*ImageHeight+xx]-c_ANG[tidy*Windowy+tidx];
            while(t_fm_nbin<0)
            t_fm_nbin+=Pi;
            int t_nm_nbin=(int)(t_fm_nbin*10/Pi);
            if(tidx<128&&tidy<128)
                atomicAdd(& (histo[d_ANG[tidy*Windowy+tidx]*70+d_Mag[tidy*Windowy+tidx]*10+t_nm_nbin]),p_Mag[yy*ImageHeight+xx]);
            __syncthreads();
            out[d_ANG[tidy*Windowy+tidx]*70+d_Mag[tidy*Windowy+tidx]*10+t_nm_nbin+(blockIdx.x+blockIdx.y*gridDim.x)*1260]=histo[d_ANG[tidy*Windowy+tidx]*70+d_Mag[tidy*Windowy+tidx]*10+t_nm_nbin];

            }
        
    
        }

__global__ void smoothcell(float *in,float *out){
    int t_nleft,t_nright;
    t_nleft=(threadIdx.x-1+10)%10;
    t_nright=(threadIdx.x+1)%10;
    float *t_ptemp,t_ftemp;
    t_ptemp=in+blockIdx.x*70+blockIdx.y*10;//+threadIdx.y)*0.8f+0.1f*(in+blockIdx.x*70+threadIdx.x*10+t_left)
    t_ftemp=t_ptemp[threadIdx.x]*0.8f+0.1f*t_ptemp[t_nleft]+0.1f*t_ptemp[t_nright];
    __syncthreads();
    out[blockIdx.x*70+blockIdx.y*10+threadIdx.x]=t_ptemp[threadIdx.x];
    __syncthreads();
}

__global__ void countblock(float *in ,float *out)
{
    if(in+70*blockIdx.x+(blockIdx.y+blockIdx.x)*10!=NULL)
   { float *ptr_in=in+70*blockIdx.x+(blockIdx.y+blockIdx.x)*10;//threadIdx.x;//70=һ���Ƕȷ���7��cell��ÿ��cell 10��bin,
    float *ptr_out=out+120*blockIdx.x+30*blockIdx.y+10*blockDim.x;//threadIdx.x;//һ���Ƕȷ���4��block��һ��block3��cell��һ��cell 10��bin,
    //һ��block3��cell��һ��cell 10��bin, 
    ptr_out[threadIdx.x]=ptr_in[threadIdx.x];
	}
    }
    
    


 
__global__ void normalizeL2Hys(float *in,float *out)
{
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    // Sum the vector
    float sum = 0;
    
    float *t_ftemp=in+bid*30;
    float *t_foutemp=out+bid*30;
    sum+=t_ftemp[tid]*t_ftemp[tid];
    __syncthreads();
    // Compute the normalization term
    float norm = 1.0f/(rsqrt(sum) + L2HYS_EPSILONHYS * 30);
    t_foutemp[tid]=t_ftemp[tid]*norm;
    __syncthreads();


}
 extern "C" void countFeaturesfloat(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight)
{
    int *device_d_ANG,*device_d_Mag;
    float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG,*device_out,*device_smooth_out,*device_block_out,*device_out_norm;
    uchar *device_in;
    long size_d_window=sizeof(int)*Windowx*Windowy;
    long size_c_window=sizeof(float)*Windowx*Windowy;
    long size_c_pixel=sizeof(float)*ImageHeight*Imagewidth;
    long size_uc_pixel=sizeof(uchar)*ImageHeight*Imagewidth;
    long size_c_cell=sizeof(float)*1260*(ImageHeight/Windowy)*(Imagewidth/Windowx);
    long size_s_cell=sizeof(float)*1260;
    long size_c_block=sizeof(float)*2160;

    checkCudaErrors(cudaMalloc((void **)&device_c_ANG,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_c_Mag,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_d_ANG,size_d_window));
    checkCudaErrors(cudaMalloc((void **)&device_d_Mag,size_d_window));
    checkCudaErrors(cudaMalloc((void **)&device_p_ANG,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_p_Mag,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_in,size_uc_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_out,size_c_cell));
    checkCudaErrors(cudaMalloc((void **)&device_smooth_out,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_block_out,size_c_block));
    checkCudaErrors(cudaMalloc((void **)&device_out_norm,size_c_block));

    checkCudaErrors(cudaMemcpy(device_c_ANG,c_ANG,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_c_Mag,c_Mag,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_d_Mag,d_Mag,size_d_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_d_ANG,d_ANG,size_d_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_Mag,p_Mag,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_ANG,p_ANG,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_in,in,size_uc_pixel,cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(device_out,0,size_c_cell));
    checkCudaErrors(cudaMemset(device_smooth_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_block_out,0,size_c_block));
    checkCudaErrors(cudaMemset(device_out_norm,0,size_c_block));

    long shared=sizeof(int)*1260;
    long h_windowx=iDivUp(Imagewidth,Windowx);
    long h_windowy=iDivUp( ImageHeight,Windowy);
    dim3 blocks(h_windowx,h_windowy);//h_windowx=ImageWidth/Windowx,h_windowy=ImageHeight/Windowy
    dim3 threads(Windowx,Windowy);//ÿһ���߳̿����һ��cell��������
    countCell<<<blocks,threads>>>(device_in, device_out, device_d_ANG,device_d_Mag,device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, Imagewidth,ImageHeight);
    
    dim3 block_smooth(18,7);//һ��cell��18���Ƕȷ���,һ������7��cell��
    dim3 threads_smooth(10);//ÿ��cell 10 ��bin

    dim3 block_b(18,4);//18=m_nANGһ�����ڷ�18������4=һ���Ƕȷ���4��block
    dim3 thread_b(3,10);//3=һ��block����3��cell,10=һ��cell10��bin

    dim3 block_norm(72);//blob������ 18*4=72
    dim3 thread_norm(30);//block�����������ȣ�m_nBIN��
    
    for(int i=0;i<h_windowx;i++)
        for(int j=0;j<h_windowy;j++)
        {       smoothcell<<<block_smooth,threads_smooth>>>(device_out+(i+h_windowx*j)*1260,device_smooth_out);
                countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);
                normalizeL2Hys<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);

				checkCudaErrors(cudaMemcpy(out+(i+h_windowx*j)*2160*sizeof(float),device_out_norm,size_c_block,cudaMemcpyDeviceToHost));
                cudaDeviceSynchronize();
    }

    cudaFree(device_c_ANG);
    cudaFree(device_c_Mag);
    cudaFree(device_d_ANG);
    cudaFree(device_d_Mag);
    cudaFree(device_p_ANG);
    cudaFree(device_p_Mag);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_smooth_out);
    cudaFree(device_block_out);
    cudaFree(device_out_norm);



    cudaDeviceReset();
    
    
}