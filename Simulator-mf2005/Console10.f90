MODULE GLOBAL
implicit none

integer :: nx,ny,nz,nsummary,nnode,nobs,npar 
integer :: nstep,Ne,nwelldata,nd,ninjector,nproducer
real*8 :: lx,dx,ly,dy,lz,dz,k1,k2,p1,p2
real*8 :: corlen_x,corlen_z,meanlnk,STD_lnk
real*8 :: mean_k,std_k,mean_s,std_s,mean_L,std_L,mean_R,std_R
real*8, allocatable :: x(:),y(:),z(:)


End MODULE 



!*********************************************************
!------MAIN Program
!**********************************************************

PROGRAM ENKF_facies
use GLOBAL
implicit none
real*8 :: tempf, perm,l,tv,g,h1,temp
integer :: i,j,ii,n,nn,n1,n2,q,nc,vv1,vv2,vv3,vv4
integer :: ierror,system
character*100 :: flname,fs1,keyword1,fs2,fs3
real*8, allocatable :: x1(:),x2(:),variable(:),var1(:,:),ksi(:),var(:),head(:,:)
real*8, allocatable ::   Kr(:),poro_ref(:),outdata1(:,:),observation(:,:)
integer :: status=0
integer, allocatable :: ind_xy(:,:)


call INIT
write(*,*) 'end Initial'


nc=nz*nx*ny      


 allocate(Kr(nc),var1(ny,nx),var(nx),head(ny,nx))              
 allocate(outdata1(nstep,nwelldata),observation(nstep,ny*nx))  

   kr=1
   open(10,file='transmissivity.dat')
   do j=1,ny
     write(10,"(<nx>f16.7)") kr((j-1)*nx+1:(j-1)*nx+nx)
   enddo
   close(10)

   head=0.0
   head(:,1)=1.0
   open(10,file='initial_head.dat')
   do j=1,ny
     write(10,"(<nx>f16.7)") head(j,1:nx)
   enddo
   close(10)

   ierror = system("mf2005.exe singlephase.MFN")
   if(ierror == -1) then
     write(*,*) ' Error in calling modflow ', ierror
     stop
   endif  
    
  
 
   open(10,file='final_head.dat')
   do nn=1,nstep
    read(10,"(A80)") fs1
    do i=1,ny
    read(10,"(<nx>f16.7)") var1(i,1:nx)
    enddo 

    do n1=1,ny
     do n2=1,nx
      observation(nn,(n1-1)*nx+n2)=var1(n1,n2)
     enddo
    enddo    
  
   enddo
   close(10)  

  
   
   open(10,file='observation.dat')
   do i=1,nstep
     write(10,"(<nx*ny>f16.7)") observation(i,1:ny*nx)
   enddo
   close(10) 

    open(10,file='observation1.dat')
   do i=1,nx*ny
     write(10,"(<nstep>f16.7)") observation(1:nstep,i)
   enddo
   close(10) 

END PROGRAM

!___________________________________________________________________
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Subroutine INIT
use global
implicit none
integer :: i, j

nx = 51
dx = 20
lx = dx*nx
ny = 51
dy = 20
ly = dy*ny
nz = 1
dz = 1
lz = dz*nz 
Ne = 1
nwelldata = 16  
nobs=16        
nstep=50                                ! do not contains time zero
nd = nwelldata                          ! the number of observation data
nnode=nx*ny*nz                         ! the total node of the model
npar=13
mean_k=0.0
std_k=1.0
 


allocate (x(nx))
do i = 1,nx
  x(i) = dx*i-dx/2
enddo
allocate (y(ny))
do i = 1,ny
  y(i) = dy*i-dy/2
enddo
allocate (z(nz))
do i = 1,nz
  z(i) = dz*i-dz/2
enddo

open(10,file="coordinate.dat")
do j=1,ny
 do i=1,nx
  write(10,*) x(i), y(j)
 enddo
enddo
close(10)


RETURN 
End Subroutine init 


