#include "countFeature.cuh"
#define stride 12





__global__ void countCell(float *out,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_histo_mask,int offset_X,int offset_Y)
{
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=blockIdx.y*blockDim.y+threadIdx.y;
	
	int tidx=threadIdx.x;
    int tidy=threadIdx.y;
    
	int off_X=xx+offset_X;
	int off_Y=yy+offset_Y;
	
	__shared__  float histo[1280];//һ��Բ��18������(max(0~17))*����Ŀ��70(ÿ������7��cellÿ��cell 10��bin)+������ţ�max(0~6)��*bin����10��+�����ĸ�bin(max(0~9))=17*70+6*10+9=1259
	
	__shared__  float t_fm_nbin[Windowy][Windowx];
	//__shared__  float temp[Windowy][Windowx];
	__shared__  int  t_nm_nbin[Windowy][Windowx];
	
	      __syncthreads();
		  t_fm_nbin[tidy][tidx]=device_p_ANG[off_Y*Imagewidth+off_X]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[tidy][tidx]<0)
         t_fm_nbin[tidy][tidx]+=Pi; 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
         t_fm_nbin[tidy][tidx]+=Pi; 
		  
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
		atomicAdd(& (histo[d_histo_mask[yy*m_nImage+xx]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(off_Y)*Imagewidth+ off_X]*d_mask[xx+(yy)*m_nImage]); 

		__syncthreads();
		
		atomicAdd(&out[tidy*32+tidx],histo[tidy*32+tidx]);
		
		if(tidy%4==0)
				atomicAdd(&out[1024+(tidy/4)*32+tidx],histo[1024+(tidy/4)*32+tidx]);
		
}

__global__ void smoothcell(float *in,float *out){
    int t_nleft,t_nright;
    t_nleft=(threadIdx.x-1+10)%10;
    t_nright=(threadIdx.x+1)%10;
    float *t_ptemp,t_ftemp[10];
    t_ptemp=in+blockIdx.x*70+blockIdx.y*10;//+threadIdx.y)*0.8f+0.1f*(in+blockIdx.x*70+threadIdx.x*10+t_left)
	/*__syncthreads();*/
	if(t_ptemp)
	t_ftemp[threadIdx.x]=t_ptemp[threadIdx.x]*0.8f+0.1f*t_ptemp[t_nleft]+0.1f*t_ptemp[t_nright];
    __syncthreads();
	out[blockIdx.x*70+blockIdx.y*10+threadIdx.x]=t_ftemp[threadIdx.x];
    __syncthreads();
}

__global__ void countblock(float *in ,float *out)
{
    //if(in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10!=NULL)
   //{ 
	float *ptr_in=in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10;//threadIdx.x;//70=һ���Ƕȷ���7��cell��ÿ��cell 10��bin,
    float *ptr_out=out+120*blockIdx.x+30*blockIdx.y+10*threadIdx.x;//threadIdx.x;//һ���Ƕȷ���4��block��һ��block3��cell��һ��cell 10��bin,
    //һ��block3��cell��һ��cell 10��bin, 
    ptr_out[threadIdx.y]=ptr_in[threadIdx.y];
	////}
    }

__global__ void normalizeL2Hys(float *in,float *out)
{
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    // Sum the vector
	__shared__ float sum[72][15];//15*72
   //memset(sum[15],0,15*sizeof(float));
   __syncthreads();
    float *t_ftemp=in+bid*30;
    float *t_foutemp=out+bid*30;
    if(tid<15) sum[bid][tid]=t_ftemp[tid+15]*t_ftemp[tid+15]+t_ftemp[tid]*t_ftemp[tid];

    __syncthreads();

	if(tid<7) sum[bid][tid]+=sum[bid][tid+7];
	 __syncthreads();

	 if(tid<3) sum[bid][tid]+=sum[bid][tid+3];
	 __syncthreads();
	/* if(tid<2) sum[bid][tid]+=sum[bid][tid+2];
	 __syncthreads();*/
	 if(tid==0) sum[bid][tid]=sum[bid][tid]+sum[bid][tid+1]+sum[bid][14]+sum[bid][6]+sum[bid][2];
	 __syncthreads();
    // Compute the normalization term
	
	 float norm = (rsqrt(sum[bid][0]));
	/*if(sum[1]-0<0.000001) norm=0;*/
	 //printf(" %f ",sum[bid][0]);
	//printf(" %f,%f ",sum[7],norm);
	t_foutemp[tid]=t_ftemp[tid]*norm;
    __syncthreads();


}


 extern "C" void countFeaturesfloat(float *out,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight,int *mask,int *histo_mask,int off_x,int off_y)
{
	 float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG;
     int *d_mask,*d_histo_mask;
    //int *device_d_ANG,*device_d_Mag,*d_mask;
    //float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG,*device_out,*device_smooth_out,*device_block_out,*device_out_norm,*device_smooth_in;
    float *device_out,*device_smooth_out,*device_block_out,*device_out_norm,*device_smooth_in;
	//uchar *device_in;
	//void * m_pClassifier;//������ָ��
	//float t_nRes;//SVM�����ĸ���
	//CvMat *t_FeatureMat;
	//CvSVM * t_pSVM = new CvSVM;
	//	t_pSVM->load( "C:\\Users\\Cyj\\Desktop\\123.xml" );
	//	/*m_pClassifier = (void *)t_pSVM;*/
	//t_FeatureMat = cvCreateMat(  1, 2160,CV_32FC1 );

      long size_d_window=sizeof(int)*m_nImage*m_nImage;
    long size_c_window=sizeof(float)*m_nImage*m_nImage;
    long size_c_pixel=sizeof(float)*ImageHeight*Imagewidth;
    
     long size_s_cell=sizeof(float)*1280;
    long size_c_block=sizeof(float)*2160;

    checkCudaErrors(cudaMalloc((void **)&device_c_ANG,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_c_Mag,size_c_window));

    checkCudaErrors(cudaMalloc((void **)&device_p_ANG,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_p_Mag,size_c_pixel));
	checkCudaErrors(cudaMalloc((void **)&d_mask,size_d_window));
	checkCudaErrors(cudaMalloc((void **)&d_histo_mask,size_d_window));
   
    checkCudaErrors(cudaMalloc((void **)&device_out,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_smooth_out,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_block_out,size_c_block));
    checkCudaErrors(cudaMalloc((void **)&device_out_norm,size_c_block));
	/* checkCudaErrors(cudaMalloc((void **)&device_in,size_uc_pixel));*/

    checkCudaErrors(cudaMemcpy(device_c_ANG,c_ANG,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_c_Mag,c_Mag,size_c_window,cudaMemcpyHostToDevice));
 
    checkCudaErrors(cudaMemcpy(device_p_Mag,p_Mag,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_ANG,p_ANG,size_c_pixel,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask,mask,size_d_window,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_histo_mask,histo_mask,size_d_window,cudaMemcpyHostToDevice));
	
    checkCudaErrors(cudaMemset(device_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_smooth_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_block_out,0,size_c_block));
    checkCudaErrors(cudaMemset(device_out_norm,0,size_c_block));

   
	int h_windowx=4;
	int h_windowy=4;
	dim3 blocks(4,4);
	dim3 threads(Windowx,Windowy);//ÿһ���߳̿����һ��cell��������
	//countCell<<<blocks,threads>>>(device_in, device_out, device_d_ANG,device_d_Mag,device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, ImageHeight,Imagewidth,d_mask);

	
	dim3 block_right(32);
	dim3 thread_right(4,128);

    dim3 block_smooth(18,7);//һ��cell��18���Ƕȷ���,һ������7��cell��
    dim3 threads_smooth(10);//ÿ��cell 10 ��bin

	
    dim3 block_b(18,4);//18=m_nANGһ�����ڷ�18������4=һ���Ƕȷ���4��block
    dim3 thread_b(3,10);//3=һ��block����3��cell,10=һ��cell10��bin

    dim3 block_norm(72);//blob������ 18*4=72
    dim3 thread_norm(30);//block�����������ȣ�m_nBIN��
    


	countCell<<<blocks,threads>>>( device_out, device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, ImageHeight,Imagewidth,d_mask,d_histo_mask,off_x,off_y);
	
	smoothcell<<<block_smooth,threads_smooth>>>(device_out,device_smooth_out);
	countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);
	normalizeL2Hys<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);

	checkCudaErrors(cudaMemcpy(out,device_out_norm,size_c_block,cudaMemcpyDeviceToHost));
		 
			
   /* cudaFreeArray(cuArray_Mag);
	   cudaFreeArray(cuArray_ANG);*/
    cudaFree(device_c_ANG);
    cudaFree(device_c_Mag);
    /*cudaFree(device_d_ANG);
    cudaFree(device_d_Mag);*/
    cudaFree(device_p_ANG);
    cudaFree(device_p_Mag);
   
    cudaFree(device_out);
    cudaFree(device_smooth_out);
    cudaFree(device_block_out);
    cudaFree(device_out_norm);

	cudaFree(d_mask);
	cudaFree(d_histo_mask);

    cudaDeviceReset();
    
    
}